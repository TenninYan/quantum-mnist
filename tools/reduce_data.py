import sys
import pickle
import numpy as np
from ptrace import ptrace
from tqdm import tqdm


def generate_reduced(feature_data):
    new_feature_data = []
    for i in tqdm(range(len(feature_data))):
        new_feature_data.append(
            ptrace(
                16,
                [6, 7, 8, 9],
                # [0, 4, 8, 12],
                feature_data[i]
            )
        )
    return np.array(new_feature_data)
        

if __name__ == '__main__':
    print(
        'usage: python tools/reduce_data.py in_fname.pickle out_fname.pickle'
    )
    in_filename = sys.argv[1]
    out_filename = sys.argv[2]
    print('loading data from %s' % in_filename)
    with open(in_filename, 'rb') as infile:
        data = pickle.load(infile)

    print('reducing data...')
    data[0] = generate_reduced(data[0])

    print('writing data to %s' % out_filename)
    with open(out_filename, 'wb') as outfile:
        pickle.dump(data, outfile)
        