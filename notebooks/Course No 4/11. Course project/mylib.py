import numpy as np
import os

def loadNpz(filename=os.path.join('data','data.npz')):
    """
    This function returns the content of the NPZ file passed as parameter.
    The NPZ file passed must been build according to the first Notebook process of this project
    The value returned a structure that is described in the Notebook 02 - Data exploration.ipynb
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

        name='trainX'
        print("building '{}' set".format(name))
        data_dict[name]=dict()
            
        for data_type in ['data', 'features', 'filenames', 'labels']:
            print("  building ", data_type)
            data_dict[name][data_type]=np.concatenate((data_dict['train'][data_type], data_dict['valid'][data_type]), axis=0)
            print('     shape: {} - dtype: {}'.format(data_dict[name][data_type].shape, data_dict[name][data_type].dtype))

        print("\n")


            
        # Return the structure prepared by this function
        return data_dict
    

def loadXy(data=None, concatenate=[]):
    """
    This function returns the data, an X and a y dict containing 'train', 'valid', 'test' and 'trainX' vectors
    from the data passed as parameter.
    The 'trainX' is the merge of the 'train' and 'valid' dataset.
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
        print("X['{}'] shape:".format(name),X[name].shape)
        print("y['{}'] shape:".format(name),X[name].shape)

    X['trainX']=np.concatenate((X['train'], X['valid']), axis=0)
    y['trainX']=np.concatenate((y['train'], y['valid']), axis=0)
    print("X['trainX'] shape:", X['trainX'].shape)
    print("y['trainX'] shape:", X['trainX'].shape)
    
    return (data, X, y)
