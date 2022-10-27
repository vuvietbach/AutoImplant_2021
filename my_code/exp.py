"""
Experiment description:

Training for Task 1 only (with only geometric augmentation).
"""

import os
import sys
import getopt
sys.path.append('./')

import ori_code.utils as u
import paths as p
import ori_code.datasets as ds
import ori_code.training_implant_reconstruction as tr
import ori_code.augmentation as aug
import pytorch_lightning as pl


def experiment(current_run):
    current_run = str(current_run)

    experiment_name = "Task1_Exp1"
    experiments_family = "Task1_Experiments"
    ### Experiment Parameters Defined Below ###

    log_dir = os.path.join(p.log_path, experiments_family, experiment_name, str("cp_" + current_run))
    comment = "Training for Task 1 only (with only geometric augmentation)."
    logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, comment=comment)
    ###########################################


    log_dir = os.path.join(p.log_path, experiments_family, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    
    training_params = dict()
    training_params['gpus'] = [0, 1]
    training_params['num_workers'] = 4
    training_params['num_iters'] = 100
    training_params['cases_per_iter'] = 500
    training_params['learning_rate'] = 0.003
    training_params['decay_rate'] = 0.99
    training_params['batch_size'] = 2
    training_params['cost_function'] = u.dice_loss
    training_params['transforms'] = aug.generate_transforms_flip_affine(scales=(0.97, 1.03), degrees=(-6, 6), translation=(-5, 5))
    training_params['training_mode'] = "defect_implant"
    training_params['checkpoints_path'] = log_dir
    training_params['to_load_checkpoint'] = None if int(current_run) <= 1 else os.path.join(p.log_path, experiments_family, experiment_name, str("cp" + str(int(current_run) - 1)))
    training_params['to_save_checkpoint'] = os.path.join(p.log_path, experiments_family, experiment_name, str("cp" + str(current_run)+'.ckpt'))
    training_params['save_best'] = True
    task = p.preprocessed_task1
    training_params['data_path'] = task.train_path
    training_params['training_csv'] = task.train_csv
    training_params['validation_csv'] = task.val_csv
    training_params['model_save_path'] = os.path.join(p.log_path, experiments_family, experiment_name, str("model_cp" + current_run +'.ckpt'))
    training_params['logger'] = logger

    ############################
    tr.training(training_params)
    ############################

def run(arguments):
    try:
        opts, args = getopt.getopt(arguments ,"c:",)
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-c":
            current_run = str(arg)
    try:
        if current_run == current_run:
            pass
    except:
        current_run = str(1)
    experiment(current_run)

if __name__ == "__main__":
    experiment(1)
    pass
