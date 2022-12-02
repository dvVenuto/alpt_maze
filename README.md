
# Maze DT

This codebase was built upon:

https://github.com/kzl/decision-transformer

and

https://github.com/google-research/dice_rl

Please follow the respective installation instructions on each of the pages.  First run:

```
cd maze
conda env create -f conda_env.yml
cd dice_rl
pip3 install -e .
```

To create a maze dataset:
```
cd maze 
cd dice_rl
python create_dataset.py --env_name 'maze:10-blocks:20' --seed 0 --num_trajectory 10000 --save_dir "."
python create_dataset.py --env_name 'maze:tunnel:20' --seed 0 --num_trajectory 10000 --save_dir "."
```

maze:10-blocks:20 refers to a maze with 10 blocks and shape 20x20.

maze:tunnel:20 refers to the tunneled maze of shape 20x20.

You can create multiple maze configurations and save them in seperate directories for later use with ALPT.

To run ALPT on a target environment of maze:10-blocks:20 with a source environment of maze:tunnel:20, you can run:

```
python experiment.py --wall_type 'blocks:10' --maze_size 20 --path_tar 'path/to/your/maze_1.pkl' --path_src 'path/to/your/maze_2.pkl' --model_type 'alpt'
```

## License