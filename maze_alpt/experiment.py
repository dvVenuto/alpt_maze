import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.alpt import ALPT

from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer

from dice_rl.environments.gridworld import maze

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = "maze", "standard"
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    env_seed = variant['seed']
    size = variant['maze_size']
    wall_type=variant['wall_type']
    env = env = maze.Maze(size, wall_type, maze_seed=env_seed)
    env.seed(env_seed)

    max_ep_len = 500
    env_targets = [6.0, 6.0]  # evaluation conditioning targets
    scale = 1000.  # normalization for rewards/returns


    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = 1

    # load dataset
    path_tar = variant['path_tar']
    path_src = [variant['path_src']]


    src_list_idm = list()
    src_list_dt = list()

    for path in path_src:
        with open(path, 'rb') as f:
            loader = pickle.load(f)
        numpy_loader_i = dict()
        numpy_loader_i['rewards'] = loader['rewards'].numpy()
        numpy_loader_i['observations'] = loader['observations'].numpy()
        numpy_loader_i['actions'] = loader['actions'].numpy()
        src_list_dt.append(numpy_loader_i)
        src_list_idm.append(numpy_loader_i)


    with open(path_tar, 'rb') as f:
        loader = pickle.load(f)
    numpy_loader_tar = dict()
    numpy_loader_tar['rewards'] = loader['rewards'].numpy()
    numpy_loader_tar['observations'] = loader['observations'].numpy()
    numpy_loader_tar['actions'] = loader['actions'].numpy()

    action_limited = variant['action_limited']
    keep_portion=variant['keep_portion']+1
    if action_limited:
        indices = np.random.choice(numpy_loader_tar['actions'].shape[0],
                                   replace=False, size=int(numpy_loader_tar['actions'].shape[0]-keep_portion))
        loader_idm = dict()
        loader_idm['actions'] = np.delete(numpy_loader_tar['actions'], indices, axis=0)
        loader_idm['observations'] = np.delete(numpy_loader_tar['observations'], indices, axis=0)
        loader_idm['rewards'] = np.delete(numpy_loader_tar['rewards'], indices, axis=0)
        numpy_loader_tar['actions'][indices,:] = 4 #NULL ACTION

    src_list_dt.append(numpy_loader_tar)

    src_list_idm.append(loader_idm)


    numpy_loader = {}
    for k in numpy_loader_i.keys():
        numpy_loader[k] = np.concatenate(list(d[k] for d in src_list_dt))

    numpy_loader_idm = {}
    for k in numpy_loader_i.keys():
        numpy_loader_idm[k] = np.concatenate(list(d[k] for d in src_list_idm))

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
        numpy_loader['rewards'][-1] = numpy_loader['rewards'].sum(axis=1)
        numpy_loader['rewards'][:-1] = 0.
    states = (numpy_loader['observations'])
    traj_lens = np.repeat(numpy_loader['observations'].shape[1], numpy_loader['observations'].shape[0])
    returns = (numpy_loader['rewards'].sum(axis=1))
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states_idm, traj_lens_idm, returns_idm = [], [], []
    if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
        numpy_loader_idm['rewards'][-1] = numpy_loader_idm['rewards'].sum(axis=1)
        numpy_loader_idm['rewards'][:-1] = 0.
    states_idm = (numpy_loader_idm['observations'])
    traj_lens_idm = np.repeat(numpy_loader_idm['observations'].shape[1], numpy_loader_idm['observations'].shape[0])
    returns_idm = (numpy_loader_idm['rewards'].sum(axis=1))
    traj_len_idm, returns_idm = np.array(traj_lens_idm), np.array(returns_idm)

    # used for input normalization
    states_idm = np.concatenate(states_idm, axis=0)
    state_mean_idm, state_std_idm = np.mean(states_idm, axis=0), np.std(states_idm, axis=0) + 1e-6

    num_timesteps_idm = sum(traj_lens_idm)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)


    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]

    trajectories = list()

    for i in range(numpy_loader['observations'].shape[0]):
        s = numpy_loader['observations']
        state = s[i]
        a = numpy_loader['actions']
        actions = a[i]
        r = numpy_loader['rewards']
        reward = r[i]
        traj = dict()
        traj['observations'] = state
        traj['actions'] = actions
        traj['rewards'] = reward
        traj['dones'] = np.repeat(0, reward.shape[0])
        trajectories.append(traj)

    ind = len(numpy_loader) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps_idm = max(int(pct_traj*num_timesteps_idm), 1)
    sorted_inds_idm = np.argsort(returns_idm)  # lowest to highest
    num_trajectories_idm = 1
    timesteps_idm = traj_lens_idm[sorted_inds_idm[-1]]

    trajectories_idm = list()

    for i in range(numpy_loader_idm['observations'].shape[0]):
        s = numpy_loader_idm['observations']
        state = s[i]
        a = numpy_loader_idm['actions']
        actions = a[i]
        r = numpy_loader_idm['rewards']
        reward = r[i]
        traj = dict()
        traj['observations'] = state
        traj['actions'] = actions
        traj['rewards'] = reward
        traj['dones'] = np.repeat(0, reward.shape[0])
        trajectories_idm.append(traj)

    ind_idm = len(numpy_loader_idm) - 2
    while ind_idm >= 0 and timesteps_idm + traj_lens_idm[sorted_inds_idm[ind_idm]] <= num_timesteps_idm:
        timesteps_idm += traj_lens_idm[sorted_inds_idm[ind_idm]]
        num_trajectories_idm += 1
        ind_idm -= 1
    sorted_inds_idm = sorted_inds_idm[-num_trajectories_idm:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample_idm = traj_lens_idm[sorted_inds_idm] / sum(traj_lens_idm[sorted_inds_idm])

    def get_batch(batch_size=256, max_len=K):
        if model_type=="alpt":
            batch_inds_idm = np.random.choice(
                np.arange(num_trajectories_idm),
                size=batch_size,
                replace=True,
                p=p_sample_idm,  # reweights so we sample according to timesteps
            )

            s_idm, a_idm, r_idm, d_idm, rtg_idm, timesteps_idm, mask_idm = [], [], [], [], [], [], []
            for i in range(batch_size):
                traj_idm = trajectories_idm[int(sorted_inds_idm[batch_inds_idm[i]])]
                si_idm = random.randint(0, traj_idm['rewards'].shape[0] - 1)

                # get sequences from dataset
                s_idm.append(traj_idm['observations'][si_idm:si_idm + max_len].reshape(1, -1, state_dim))
                a_idm.append(traj_idm['actions'][si_idm:si_idm + max_len].reshape(1, -1, act_dim))
                r_idm.append(traj_idm['rewards'][si_idm:si_idm + max_len].reshape(1, -1, 1))
                if 'terminals' in traj_idm:
                    d_idm.append(traj_idm['terminals'][si_idm:si_idm + max_len].reshape(1, -1))
                else:
                    d_idm.append(traj_idm['dones'][si_idm:si_idm + max_len].reshape(1, -1))
                timesteps_idm.append(np.arange(si_idm, si_idm + s_idm[-1].shape[1]).reshape(1, -1))
                timesteps_idm[-1][timesteps_idm[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
                rtg_idm.append(discount_cumsum(traj_idm['rewards'][si_idm:], gamma=1.)[:s_idm[-1].shape[1] + 1].reshape(1, -1, 1))
                if rtg_idm[-1].shape[1] <= s_idm[-1].shape[1]:
                    rtg_idm[-1] = np.concatenate([rtg_idm[-1], np.zeros((1, 1, 1))], axis=1)

                # padding and state + reward normalization
                tlen_idm = s_idm[-1].shape[1]
                s_idm[-1] = np.concatenate([np.zeros((1, max_len - tlen_idm, state_dim)), s_idm[-1]], axis=1)
                s_idm[-1] = (s_idm[-1] - state_mean) / state_std
                a_idm[-1] = np.concatenate([np.ones((1, max_len - tlen_idm, act_dim)), a_idm[-1]], axis=1)
                r_idm[-1] = np.concatenate([np.zeros((1, max_len - tlen_idm, 1)), r_idm[-1]], axis=1)
                d_idm[-1] = np.concatenate([np.ones((1, max_len - tlen_idm)) * 2, d_idm[-1]], axis=1)
                rtg_idm[-1] = np.concatenate([np.zeros((1, max_len - tlen_idm, 1)), rtg_idm[-1]], axis=1) / scale
                timesteps_idm[-1] = np.concatenate([np.zeros((1, max_len - tlen_idm)), timesteps_idm[-1]], axis=1)
                mask_idm.append(np.concatenate([np.zeros((1, max_len - tlen_idm)), np.ones((1, tlen_idm))], axis=1))

        s_idm = torch.from_numpy(np.concatenate(s_idm, axis=0)).to(dtype=torch.float32, device=device)
        a_idm = torch.from_numpy(np.concatenate(a_idm, axis=0)).to(dtype=torch.long, device=device)
        r_idm = torch.from_numpy(np.concatenate(r_idm, axis=0)).to(dtype=torch.float32, device=device)
        d_idm = torch.from_numpy(np.concatenate(d_idm, axis=0)).to(dtype=torch.long, device=device)
        rtg_idm = torch.from_numpy(np.concatenate(rtg_idm, axis=0)).to(dtype=torch.float32, device=device)
        timesteps_idm = torch.from_numpy(np.concatenate(timesteps_idm, axis=0)).to(dtype=torch.long, device=device)
        mask_idm = torch.from_numpy(np.concatenate(mask_idm, axis=0)).to(device=device)

        #DT
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.long, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        if model_type=="alpt":
            return s, a, r, d, rtg, timesteps, mask, s_idm, a_idm, r_idm, d_idm, rtg_idm, timesteps_idm
        else:
            return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'alpt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    elif model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'alpt':
        model = ALPT(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )

    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            alpt=False,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'alpt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            alpt=True,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--wall_type', type=str, default="blocks:10")  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--maze_size', type=int, default='10')  # normal for standard setting, delayed for sparse
    parser.add_argument('--path_src', type=str, default='dice_rl/maze_pickle')  # normal for standard setting, delayed for sparse
    parser.add_argument('--path_tar', type=str, default='dice_rl/maze_pickle')  # normal for standard setting, delayed for sparse

    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--keep_portion', type=int, default=10)
    parser.add_argument('--seed', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='alpt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=1)
    parser.add_argument('--num_steps_per_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--action_limited', type=bool, default=True)

    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
