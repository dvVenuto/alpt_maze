import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        if self.alpt:
            states, actions, rewards, dones, rtg, timesteps, attention_mask, states_idm, actions_idm, rewards_idm, dones_idm, returns_to_go_idm, timesteps_idm = self.get_batch(self.batch_size)
        else:
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)

        action_target_idm = torch.clone(actions_idm)
        if self.alpt:
            state_preds, action_preds, reward_preds, action_target, action_preds_idm = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps,
                states_idm, actions_idm, rewards_idm, returns_to_go_idm[:,:-1], timesteps_idm,
                attention_mask=attention_mask,
            )
        else:
            state_preds, action_preds, reward_preds, action_target = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
            )

        act_dim = action_preds.shape[2]

        if self.alpt:
            action_target = torch.nn.functional.one_hot(action_target)
            action_target = action_target.to(torch.float32)

            action_preds = action_preds.unsqueeze(2)

            action_target_idm = torch.nn.functional.one_hot(action_target_idm)
            action_target_idm = action_target_idm.to(torch.float32)

            action_preds_idm = action_preds_idm.unsqueeze(2)


            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(action_preds, action_target) + loss_fn(action_preds_idm, action_target_idm)
        else:
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            loss = self.loss_fn(
                None, action_preds, None,
                None, action_target, None,
            )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
