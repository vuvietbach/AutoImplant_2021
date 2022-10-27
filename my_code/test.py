import sys
sys.path.append('./')
import ori_code.datasets as ds

def main():
    data_path = './'
    val_csv = 'dataset/preprocessed_data/task1/val.csv'
    training_mode = 'defect_implant'
    batch_size = 2
    dl = ds.create_dataloader(data_path, val_csv, training_mode, batch_size=2, transforms=None, shuffle=False)
    for batch in dl:
        input, gt, _ = batch
        print(input.size())
        print(gt.size())
        break
if __name__ == '__main__':
    main()