import numpy as np
import os

def loadNpz(filename=os.path.join('data','data.npz')):
    """
    This function returns the content of the NPZ file passed as parameter.
    The NPZ file passed must been build according to the first Notebook process of this project
    The value returned as a structure described above
    """
    
    # initialize returned variable
    data_dict=dict()
    
    # Load the NPZ file
    with np.load(filename) as npz_file:
        
        # Load constant
        data_dict={key: value for key, value in zip(npz_file['constant_label'], npz_file['constant_value'])}    
        
        for name in data_dict['DATASET_NAME']:
            print("Loading '{}' set".format(name))
            data_dict[name]=dict()
            
            for data_type in ['data', 'features', 'filenames']:
                print("  loading ", data_type)
                data_dict[name][data_type]=npz_file['{}_{}'.format(name, data_type)]
                print('     shape: {} - dtype: {}'.format(data_dict[name][data_type].shape, data_dict[name][data_type].dtype))
            #print("Loading features from {} set".format(name))
            #features[name]=npz_file['{}_features'.format(name)]
            #print('  shape: {} - dtype: {}'.format(features[name].shape, features[name].dtype))
            #print("Loading filenames from {} set".format(name))
            #filenames[name]=npz_file['{}_filenames'.format(name)]
            #print('  shape: {} - dtype: {}'.format(filenames[name].shape, filenames[name].dtype))

            print("\n")
        return data_dict
    


