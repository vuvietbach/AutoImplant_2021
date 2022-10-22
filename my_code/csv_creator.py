import os
import csv
import random
import paths as p
def create_train_val_csv(complete_skull_path, defect_skull_paths, implant_paths, save_dir, ratio=0.9):
    # get file name
    complete_skull_files = os.listdir(complete_skull_path)
    defect_skull_files = [os.listdir(path) for path in defect_skull_paths]
    implant_files = [os.listdir(path) for path in implant_paths]
    # get file path
    complete_skulls = [os.path.join(complete_skull_path, file) for file in complete_skull_files]
    # for dir, files in zip(defect_skull_paths, defect_skull_files):
    #     print(dir, files)
    # import pdb; pdb.set_trace()
    defect_skulls = [[os.path.join(dir, file) for file in files] for dir, files in zip(defect_skull_paths, defect_skull_files)]
    implants = [[os.path.join(dir, file) for file in files] for dir, files in zip(implant_paths, implant_files)]
    
    f_train = open(os.path.join(save_dir, 'train.csv'), 'w')
    f_val = open(os.path.join(save_dir, 'val.csv'), 'w')
    
    #init csv writer
    train_writer = csv.writer(f_train)
    val_writer = csv.writer(f_val)
    
    #write header
    train_writer.writerow(['complete_skull', 'defect_skull', 'implant'])
    val_writer.writerow(['complete_skull', 'defect_skull', 'implant'])
    
    total_data = len(complete_skulls)
    train_data = int(total_data * ratio)
    val_data = total_data - train_data
    for defect_type, implant_type in zip(defect_skulls, implants):
        val_idx = random.sample(range(total_data), val_data)
        train_idx = [idx for idx in range(total_data) if idx not in val_idx]
        for idx in train_idx:
            train_writer.writerow([complete_skulls[idx], defect_type[idx], implant_type[idx]])
        for idx in val_idx:
            val_writer.writerow([complete_skulls[idx], defect_type[idx], implant_type[idx]])
    
    f_train.close()
    f_val.close()

def create_test_csv(test_dirs, defect_types, save_dir):
    with open(os.path.join(save_dir, 'test.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['defect_skull'])
        for dir in test_dirs:
            files = get_children_path(dir)
            for path in files:
                writer.writerow([path])


def get_children_path(par_dir):
    files = os.listdir(par_dir)
    paths = [os.path.join(par_dir, file) for file in files]
    return paths

def run():
    # task 1
    create_train_val_csv(p.task1_train_complete_skull, get_children_path(p.task1_train_defect_skull), get_children_path(p.task1_train_implant), p.task1_path)
    create_test_csv(get_children_path(p.task1_test_path), os.listdir(p.task1_test_path), p.task1_path)
    # task 3
    create_train_val_csv(p.task3_train_complete_skull, [p.task3_train_defect_skull], [p.task3_train_implant], p.task3_path)
    create_test_csv([p.task3_test_path], [""], p.task3_path)
if __name__ == '__main__':
    run()