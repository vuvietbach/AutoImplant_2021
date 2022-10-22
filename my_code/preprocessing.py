import sys
sys.path.append('code')
import utils as u
import paths as p
import pandas as pd
import numpy as np
import os
import pathlib
def preprocess_training_set(csv_path, **preprocessing_params):
    output_spacing = preprocessing_params['output_spacing']
    output_size = preprocessing_params['output_size']
    pad_size = preprocessing_params['pad_size']
    offset = preprocessing_params['offset']

    dataframe = pd.read_csv(csv_path)
    print("Dataset size: ", len(dataframe))
    
    errors = list()
    for (id, complete_skull_path, defective_skull_path, implant_path) in dataframe.itertuples():
        print("Current ID: ", id)

        complete_skull, defective_skull, implant, spacing = u.load_training_case(complete_skull_path, defective_skull_path, implant_path)
        print("Original Complete Skull Shape: ", complete_skull.shape)
        print("Original Defective Skull Shape: ", defective_skull.shape)
        print("Original Implant Shape: ", implant.shape)
        print("Initial spacing: ", spacing)

        preprocessed_complete_skull, preprocessed_defective_skull, preprocessed_implant, to_pad, internal_shape, padding = u.preprocess_training_case(defective_skull, complete_skull, implant, spacing, output_spacing, pad_size, output_size, offset)
        print("Preprocessed Complete Skull Shape: ", preprocessed_complete_skull.shape)
        print("Preprocessed Defective Skull Shape: ", preprocessed_defective_skull.shape)
        print("Preprocessed Implant Shape: ", preprocessed_implant.shape)

        recovered_complete_skull = u.postprocess_case(preprocessed_complete_skull, spacing, output_spacing, padding, to_pad, internal_shape, pad_size)
        mse = lambda a, b: np.mean((a-b)**2)
        error = mse(complete_skull, recovered_complete_skull)
        print("MSE: ", error)
        errors.append(error)

        preprocessed_complete_skull_path = complete_skull_path.replace('raw_data', 'preprocessed_data')
        preprocessed_defective_skull_path = defective_skull_path.replace('raw_data', 'preprocessed_data')
        preprocessed_implant_path = implant_path.replace('raw_data', 'preprocessed_data')

        for path in [preprocessed_complete_skull_path, preprocessed_defective_skull_path, preprocessed_implant_path]:
            dir = '/'.join(path.split('/')[:-1])
            os.makedirs(dir, exist_ok=True)
    
        u.save_volume(preprocessed_complete_skull, output_spacing, pathlib.Path(preprocessed_complete_skull_path))
        u.save_volume(preprocessed_defective_skull, output_spacing, pathlib.Path(preprocessed_defective_skull_path))
        u.save_volume(preprocessed_implant, output_spacing, pathlib.Path(preprocessed_implant_path))
    
    print("Mean error: ", np.mean(errors))
    print("Max error: ", np.max(errors))

def preprocess_testing_set(csv_path, **preprocessing_params):
    output_spacing = preprocessing_params['output_spacing']
    output_size = preprocessing_params['output_size']
    pad_size = preprocessing_params['pad_size']
    offset = preprocessing_params['offset']

    dataframe = pd.read_csv(csv_path)
    print("Dataset size: ", len(dataframe))
    
    errors = list()
    for (current_id, defective_skull_path) in dataframe.itertuples():
        print("Current ID: ", current_id)
        defective_skull, spacing = u.load_testing_case(defective_skull_path)
        print("Original Defective Skull Shape: ", defective_skull.shape)
        print("Initial spacing: ", spacing)

        preprocessed_defective_skull, to_pad, internal_shape, padding = u.preprocess_testing_case(defective_skull, spacing, output_spacing, pad_size, output_size, offset)
        print("Preprocessed Defective Skull Shape: ", preprocessed_defective_skull.shape)

        recovered_defective_skull = u.postprocess_case(preprocessed_defective_skull, spacing, output_spacing, padding, to_pad, internal_shape, pad_size)
        mse = lambda a, b: np.mean((a-b)**2)
        error = mse(defective_skull, recovered_defective_skull)
        print("MSE: ", error)
        errors.append(error)

        preprocessed_defective_skull_path = defective_skull_path.replace('raw_data', 'preprocessed_data')
        
        os.makedirs('/'.join(preprocessed_defective_skull_path.split('/')[:-1]), exist_ok=True)
        
        u.save_volume(preprocessed_defective_skull, output_spacing, pathlib.Path(preprocessed_defective_skull_path))
    
    print("Mean error: ", np.mean(errors))
    print("Max error: ", np.max(errors))
def run():
    preprocessing_params = dict()
    preprocessing_params['output_spacing'] = (1.0, 1.0, 1.0)
    preprocessing_params['output_size'] = (240, 200, 240)
    preprocessing_params['pad_size'] = 3
    preprocessing_params['offset'] = 35

    preprocess_training_set(p.task1_train_path, **preprocessing_params)
    # preprocess_training_set(p.task_1_training_path, p.task_1_training_preprocessed_path, p.task_1_validation_csv_path, **preprocessing_params)

    # preprocess_task_3_training_set(p.task_3_training_path, p.task_3_training_preprocessed_path, p.task_3_training_csv_path, **preprocessing_params)
    # preprocess_task_3_training_set(p.task_3_training_path, p.task_3_training_preprocessed_path, p.task_3_validation_csv_path, **preprocessing_params)

    # preprocess_task_1_testing_set(p.task_1_testing_path, p.task_1_testing_preprocessed_path, p.task_1_testing_csv_path, **preprocessing_params)
    # preprocess_task_3_testing_set(p.task_3_testing_path, p.task_3_testing_preprocessed_path, p.task_3_testing_csv_path, **preprocessing_params)
if __name__ == '__main__':
    run()