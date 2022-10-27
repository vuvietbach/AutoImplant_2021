import sys
sys.path.append('./')
import pandas as pd
import ori_code.utils as u
import ori_code.pipeline as pp
import ori_code.evaluation_metrics as metrics
import csv
import time
import os
import numpy as np
from networks import unet
def run(val_csv, reconstruction_params, echo=False):
    # read evaluation csv file
    val_csv = 'dataset/raw_data/task1/val.csv'
    df = pd.read_csv(val_csv)
    # write result to csv file
    f = open('result/task1/val_res.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['Case', 'DC', 'BDC', 'HD95'])
    
    # mean_reconstruction time
    mtime = []
    for _, defect_skull_path, implant_path in df.itertuples(index=False):
        start = time.time()
        defective_skull, spacing, _ = u.load_volume(defect_skull_path)
        implant, _, _ = u.load_volume(implant_path)

        to_save_path = implant_path.replace('/raw_data', '').replace('dataset', 'result')
        os.makedirs('/'.join(to_save_path.split('/')[:-1]), exist_ok=True)
        
        print(to_save_path)
        reconstructed_implant = pp.defect_reconstruction(defect_skull_path, to_save_path, echo=echo, **reconstruction_params)
        
        dc = metrics.dc(reconstructed_implant, implant)
        bdc = metrics.bdc(reconstructed_implant, implant, defective_skull, spacing)
        hd95 = metrics.hd95(reconstructed_implant, implant, spacing)

        writer.writerow([defect_skull_path, dc, bdc, hd95])
        elapsed = time.time() - start
        
        mtime.append(elapsed)
        print('reconstruction time:')
        print(elapsed)

    f.close()
    
    mtime = np.mean(np.array(mtime))
    print("mean reconstruction time:")
    print(mtime)
if __name__ == '__main__':
    val_csv = 'dataset/raw_data/task1/val.csv'
    reconstruction_params = {}
    reconstruction_params['device'] = 'cuda:0'
    reconstruction_params['reconstruction_model'] = unet
    reconstruction_params['reconstruction_weights'] = 'log/Task1_Experiments/Task1_Exp1/model_cp1.ckpt'
    reconstruction_params['defect_refinement'] = False
    reconstruction_params['implant_modeling'] = False
    run(val_csv, reconstruction_params)
