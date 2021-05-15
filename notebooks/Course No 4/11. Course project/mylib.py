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
    

def loadXy(data=None, concatenate=[]):
    """
    This function returns the data, X_train, y_train, X_valid, y_valid, X_test and y_test vectors
    from the data passed as parameter.
    If the data parameter is set to None, this function uses the loadNpz() function with default parameter
    to get the data.
    Note that the data passed as parameter must comply with the structure passed in Notebook number 1
    """

    if data==None:
        data=loadNpz()

    X=dict()
    y=dict()
    # Get X_train from high level feateurs and y_train from labels
    for name in data['DATASET_NAME']:
        X[name]=data[name]['features']
        y[name]=data[name]['labels']
        print("X {} shape:".format(name),X[name].shape)
        print("y {} shape:".format(name),X[name].shape)

    return (data, X, y)
