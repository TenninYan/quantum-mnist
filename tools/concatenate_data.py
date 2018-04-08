import sys
import pickle
import numpy as np


if __name__ == '__main__':
    print(
        'usage: python concatenate_data.py out_fname.pickle '
        'in0_fname.pickle in1_fname.pickle ...'
    )

    total_data = [
        None,
        None
    ]        

    for in_filename in sys.argv[2:]:
        print('opening %s' % in_filename)
        with open(in_filename, 'rb') as infile:
            data_in = pickle.load(infile)

        for i in range(2):
            if total_data[i] is None:
                total_data[i] = data_in[i]
            else:
                total_data[i] = np.concatenate((
                    total_data[i],
                    data_in[i]
                ))

        print('total # samples %s' % len(total_data[0]))

    out_filename = sys.argv[1]
    print('writing all data to %s' % out_filename)
    with open(out_filename, 'wb') as outfile:
        pickle.dump(total_data, outfile)
