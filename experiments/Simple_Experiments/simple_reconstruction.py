"""
Experiment description:

Just a verification whether the training script works correctly.
"""

import os
import sys
import getopt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import utils as u
import paths as p
import datasets as ds
import training_implant_reconstruction as tr
import augmentation as aug
import pytorch_lightning as pl


def experiment(current_run):
    current_run = str(current_run)

    ### Experiment Parameters Defined Below ###
    gpus = [0]
    num_workers = 4
    num_iters = 3
    cases_per_iter = 50
    learning_rate = 0.002
    decay_rate = 0.95
    batch_size = 1
    cost_function = u.dice_loss
    transforms = aug.generate_transforms_flip_affine()
    training_mode = "defect_implant"
    checkpoints_path = p.checkpoints_path / "Simple_Experiments" / "simple_reconstruction"
    to_load_checkpoint = None if int(current_run) <= 1 else p.checkpoints_path / "Simple_Experiments" / "simple_reconstruction" / str("cp" + str(int(current_run) - 1))
    to_save_checkpoint = p.checkpoints_path / "Simple_Experiments" / "simple_reconstruction" / str("cp" + str(current_run))
    data_path = p.task_1_training_preprocessed_path
    training_csv = p.task_1_training_csv_path
    validation_csv = p.task_1_validation_csv_path
    model_save_path = p.simple_exp1_save_path / str("model_cp" + current_run)

    log_dir = p.logs_path / "Simple_Experiments" / "simple_reconstruction" / str("cp_" + current_run)
    comment = "Verification whether the training works correctly."
    logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, comment=comment)
    ###########################################

    training_params = dict()
    training_params['gpus'] = gpus
    training_params['num_workers'] = num_workers
    training_params['num_iters'] = num_iters
    training_params['cases_per_iter'] = cases_per_iter
    training_params['learning_rate'] = learning_rate
    training_params['decay_rate'] = decay_rate
    training_params['batch_size'] = batch_size
    training_params['cost_function'] = cost_function
    training_params['transforms'] = transforms
    training_params['training_mode'] = training_mode
    training_params['checkpoints_path'] = checkpoints_path
    training_params['to_load_checkpoint'] = to_load_checkpoint
    training_params['to_save_checkpoint'] = to_save_checkpoint
    training_params['data_path'] = data_path
    training_params['training_csv'] = training_csv
    training_params['validation_csv'] = validation_csv
    training_params['model_save_path'] = model_save_path
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
    experiment(2)
    experiment(3)
    pass

