import sys
import ori_code.utils as u
from paths import preprocessed_task1, preprocessed_task3
import paths as p
import os
import csv

from registration import instance_optimization
from registration import utils_tc
from registration import cost_functions
from registration import regularizers

def register_single(source, target):
    """
    Registration (purely nonrigid) of the given source and target with fixed registration parameters.
    """
    device = "cuda:0"
    y_size, x_size, z_size = source.shape
    source = tc.from_numpy(source.astype(np.float32))
    target = tc.from_numpy(target.astype(np.float32))

    source = source.view(1, 1, y_size, x_size, z_size).to(device)
    target = target.view(1, 1, y_size, x_size, z_size).to(device)

    num_levels = 4
    used_levels = 4
    num_iters = 50
    learning_rate = 0.01
    alpha = 100000
    cost_function = cost_functions.mse_tc
    regularization_function = regularizers.diffusion_tc
    cost_function_params = {}
    regularization_function_params = {}
    
    transformation = instance_optimization.affine_registration(source, target, num_levels, used_levels-1, num_iters, learning_rate, cost_function, device=device)
    displacement_field_tc = utils_tc.tc_transform_to_tc_df(transformation, (1, 1, y_size, x_size, z_size), device=device)
    displacement_field_tc = instance_optimization.nonrigid_registration(source, target, num_levels, used_levels, num_iters, learning_rate,
    alpha, cost_function, regularization_function, cost_function_params, regularization_function_params, initial_displacement_field=displacement_field_tc, device=device)
    displacement_field = utils_tc.tc_df_to_np_df(displacement_field_tc)
    return displacement_field

def register_dataset(src_task, tar_task, defect_types):
    src_skulls = os.listdir(src_task.train_complete_skull)
    tar_skulls = os.listdir(tar_task.train_complete_skull)
    
    save_dir = os.path.join(p.registered_data_dir, f"{src_task.id}_{tar_task.id}")
    os.makedirs(save_dir, exist_ok=True)

    save_dir_complete_skull = save_dir + '/complete_skull'
    save_dir_defect_skull = save_dir + '/defective_skull'
    save_dir_implant = save_dir + '/implant'
    
    train_csv = save_dir + '/train.csv'
    
    f = open(train_csv, 'w')
    writer = csv.writer(f)
    writer.writerow(['complete_skull','defect_skull','implant'])
    
    for s_skull_id in src_skulls:
        for t_skull_id in tar_skulls:
            s_id = s_skull_id.rstrip('.nrrd')
            t_id = t_skull_id.rstrip('.nrrd')
            save_name = f'{s_id}_{t_id}.nrrd'
            src_path = os.path.join(src_task.train_complete_skull, s_skull_id)
            tar_path = os.path.join(tar_task.train_complete_skull, t_skull_id)
            
            source, spacing, _ = u.load_volume(src_path)
            target, _, _ = u.load_volume(tar_path)
            displacement_field = register_single(source, target)

            warped_complete_skull = u.image_warping(source, displacement_field, order=0)
            complete_skull_save_path = os.path.join(save_dir_complete_skull, save_name)
            u.save_volume(warped_complete_skull, spacing, complete_skull_save_path)
            
            for defect in defect_types:
                defect_skull, _, _ = u.load_volume(os.path.join(src_task.train_defect_skull, defect, s_skull_id)) 
                implant, _, _ =  u.load_volume(os.path.join(src_task.train_implant, defect, s_skull_id))        
                
                warped_defect_skull =  u.image_warping(defect_skull, displacement_field, order=0)
                warped_implant =  u.image_warping(implant, displacement_field, order=0)

                warped_defect_save = os.path.join(save_dir_defect_skull, defect, save_name)
                warped_implant_save = os.path.join(save_dir_implant, defect, save_name)

                u.save_volume(warped_defect_skull, spacing, warped_defect_save)
                u.save_volume(warped_implant, spacing, warped_implant_save)

                writer.writerow([complete_skull_save_path, warped_defect_save, warped_implant_save])

    f.close()


def run():
    register_dataset(preprocessed_task1, preprocessed_task1)
    register_dataset(preprocessed_task1, preprocessed_task3)
    register_dataset(preprocessed_task3, preprocessed_task1)
    register_dataset(preprocessed_task3, preprocessed_task3)
if __name__ == '__main__':
    run()