from atexit import register
from easydict import EasyDict as edict
########## raw data ############
# task 1
task1_r = edict({
    'dir': 'dataset/raw_data/task1'
}
)
task1_r.train_csv = task1_r.dir + '/train.csv'
task1_r.val_csv = task1_r.dir + '/val.csv'
task1_r.test_csv = task1_r.dir + '/test.csv'

task3_r = edict({
    'dir': 'dataset/raw_data/task1'
}
)
task3_r.train_csv = task3_r.dir + '/train.csv'
task3_r.val_csv = task3_r.dir + '/val.csv'
task3_r.test_csv = task3_r.dir + '/test.csv'

task1_path = 'dataset/raw_data/task1'
task1_train_path = task1_path + '/training-set'
task1_train_complete_skull = task1_train_path + '/complete_skull'
task1_train_defect_skull = task1_train_path + '/defective_skull'
task1_train_implant = task1_train_path + '/implant'
task1_test_path = task1_path + '/test-set'
# task 3 
task3_path = 'dataset/raw_data/task3'
task3_train_path = task3_path + '/training-set'
task3_train_complete_skull = task3_train_path + '/complete_skull'
task3_train_defect_skull = task3_train_path + '/defective_skull'
task3_train_implant = task3_train_path + '/implant'
task3_test_path = task3_path + '/test-set'

task1_train_csv = 'dataset/raw_data/task1/train.csv'
task1_val_csv = 'dataset/raw_data/task1/val.csv'
task3_train_csv = 'dataset/raw_data/task3/train.csv'
task3_val_csv = 'dataset/raw_data/task3/val.csv'

preprocessed_task1 = edict({
    "train_path": 'dataset/preprocessed_data/task1',
    'id':1
})
preprocessed_task1.train_complete_skull = preprocessed_task1.train_path + '/complete_skull'
preprocessed_task1.train_defect_skull = preprocessed_task1.train_path + '/defective_skul'
preprocessed_task1.train_implant = preprocessed_task1.train_path + '/implant'
preprocessed_task1.train_csv = preprocessed_task1.train_path + '/train.csv'
preprocessed_task1.val_csv = preprocessed_task1.train_path + '/val.csv'

preprocessed_task3 = edict({
    "train_path": 'dataset/preprocessed_data/task3',
    'id':3
})
preprocessed_task3.train_complete_skull = preprocessed_task3.train_path + '/complete_skull'
preprocessed_task3.train_defect_skull = preprocessed_task3.train_path + '/defective_skul'
preprocessed_task3.train_implant = preprocessed_task3.train_path + '/implant'
task1 = edict({
    'defect_type': ['bilateral', 'frontoorbital', 'parietotemporal', 'random_1', 'random_2']
})
task3 = edict({
    'defect_type': [""]
})


registered_data_dir = 'dataset/registered_data'


log_path = 'log'

test_task = edict(
    {
        'train_csv': 'dataset/raw_data/task1/train.csv',
        'val_csv':'dataset/raw_data/task1/val.csv',
        'train_path': 'dataset/raw_data/task1/training-set'
    }
)