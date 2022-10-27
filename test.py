import nrrd
import ori_code.utils as u
import pandas as pd
def test(path):
    output_spacing = (1.0, 1.0, 1.0)
    output_size = (240, 200, 240)
    pad_size = 3
    offset = 35
    df = pd.read_csv('dataset/raw_data/task1/train.csv')
    for cs, ds, i in df.itertuples(index=False):
        cs, ds, i, spacing = u.load_training_case(cs, ds, i)
        cs, ds, i, to_pad, internal_shape, padding = u.preprocess_training_case(cs, ds, i, spacing, output_spacing, pad_size, output_size, offset)
        out_path = 'test_data/p.nrrd'
        for v in [cs, ds, i]:
            u.save_volume(v, output_spacing, out_path)
            t, spacing, _ = u.load_volume(out_path)
            print(spacing)
        break


if __name__ == '__main__':
    import os
    print(os.path.dirname(__file__))