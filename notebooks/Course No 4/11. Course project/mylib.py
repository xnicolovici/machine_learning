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
        data_dict={key: value for key, value in zip(npz_file['constant_labels'], npz_file['constant_values'])}
        
        # Load class_name
        data_dict['class_name']=npz_file['class_name']
        
        # Load dataset
        for name in data_dict['DATASET_NAME']:
            print("Loading '{}' set".format(name))
            data_dict[name]=dict()
            
            for data_type in ['data', 'features', 'filenames', 'labels']:
                print("  loading ", data_type)
                data_dict[name][data_type]=npz_file['{}_{}'.format(name, data_type)]
                print('     shape: {} - dtype: {}'.format(data_dict[name][data_type].shape, data_dict[name][data_type].dtype))

            print("\n")
            
        # Return the structure prepared by this function
        return data_dict
    


