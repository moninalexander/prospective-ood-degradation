from definitions.train_eval import run_experiment
from configs.configs import (make_experiment_config, 
                     clean_dataset,
                     perfect_shortcut_dataset,
                     cnn_95, 
                     mlp_95, 
                     clean_training,
                     perfect_shortcut_training)
from pathlib import Path
from configs.local_config import EXPERIMENTS_DIR


################################### CNN ###################################

random_number=5250
run_name='cnn_shortcut_run'
for seed in range(random_number, random_number+2):
    config=make_experiment_config(cnn_95, perfect_shortcut_dataset, perfect_shortcut_training, seed)
    run_experiment(config,EXPERIMENTS_DIR,run_name)


### Clean

seed=1234567890
run_name='cnn_clean_run'
config=make_experiment_config(cnn_95, clean_dataset, clean_training, seed)
run_experiment(config,EXPERIMENTS_DIR,run_name)


################################### MLP ###################################

random_number=5250
run_name='mlp_shortcut_run'
for seed in range(random_number, random_number+2):
    config=make_experiment_config(mlp_95, perfect_shortcut_dataset, perfect_shortcut_training, seed)
    run_experiment(config,EXPERIMENTS_DIR,run_name)


### Clean

seed=1234567890
run_name='mlp_clean_run'
config=make_experiment_config(mlp_95, clean_dataset, clean_training, seed)
run_experiment(config,EXPERIMENTS_DIR,run_name)








