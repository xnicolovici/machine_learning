import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle


def loadNpz(filename=os.path.join('data','data.npz'), verbose=True):
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
            if verbose:
                print("Loading '{}' set".format(name))
            data_dict[name]=dict()
            
            for data_type in ['data', 'features', 'filenames', 'labels']:
                if verbose:
                    print("  loading ", data_type)
                data_dict[name][data_type]=npz_file['{}_{}'.format(name, data_type)]
                if verbose:
                    print('     shape: {} - dtype: {}'.format(data_dict[name][data_type].shape, data_dict[name][data_type].dtype))
            if verbose:
                print("\n")

        if verbose:
            print("building '{}' set".format(name))
            
        name='trainX'
        data_dict[name]=dict()
            
        for data_type in ['data', 'features', 'filenames', 'labels']:
            if verbose:
                print("  building ", data_type)
            data_dict[name][data_type]=np.concatenate((data_dict['train'][data_type], data_dict['valid'][data_type]), axis=0)
            if verbose:
                print('     shape: {} - dtype: {}'.format(data_dict[name][data_type].shape, data_dict[name][data_type].dtype))

        if verbose:
            print("\n")
            
        # Return the structure prepared by this function
        return data_dict
    

def loadXy(data=None, concatenate=[], verbose=True):
    """
    This function returns the data, an X and a y dict containing 'train', 'valid', 'test' and 'trainX' vectors
    from the data passed as parameter.
    The 'trainX' is the merge of the 'train' and 'valid' dataset.
    If the data parameter is set to None, this function uses the loadNpz() function with default parameter
    to get the data.
    Note that the data passed as parameter must comply with the structure passed in Notebook number 1
    """

    if data==None:
        data=loadNpz(verbose=verbose)

    X=dict()
    y=dict()
    # Get X_train from high level feateurs and y_train from labels
    for name in data['DATASET_NAME']:
        X[name]=data[name]['features']
        y[name]=data[name]['labels']
        if(verbose==True):
            print("X['{}'] shape:".format(name),X[name].shape)
            print("y['{}'] shape:".format(name),X[name].shape)

    X['trainX']=np.concatenate((X['train'], X['valid']), axis=0)
    y['trainX']=np.concatenate((y['train'], y['valid']), axis=0)
    if(verbose==True):
        print("X['trainX'] shape:", X['trainX'].shape)
        print("y['trainX'] shape:", X['trainX'].shape)
    
    return (data, X, y)


def getModelFilename(model_name):
    """
    Basic function that will return the filename used to store on disk the model
    passed as parameter
    """
    return 'model-{}.sav'.format(model_name)

def saveModel(model, name):
    """
    Function that saves on disk the model passed as first parameter.
    It uses the function getModelFilename() with the 'name' parameter
    to get the filename where to save the model
    """
    filename=getModelFilename(name)
    # Save model to disk
    print("Saving model {} to {}".format(name, filename))
    pickle.dump(model, open(filename, 'wb'))

def loadModel(name):
    """
    Function that loads from disk the model of which name is passed as first parameter.
    It uses the function getModelFilename() with the 'name' parameter
    to get the filename from where to load the model
    """
    filename=getModelFilename(name)
    # load the model from disk
    print("Loading model from ", filename)
    return pickle.load(open(filename, 'rb'))



def plotGridSearchResults(results_df, x_param, y_param=[], semilogx=True, xlabel='', ylabel='accuracy (%)', title='', figsize=(15,10), std_param={}):
    """
    Function to graph data points from GridSearchCV results. Can be used to graph the mean test and train
    score of a GridSearchCV fitted object.
    Note that the graph built expects % values on the Y axis (mean_test_score, mean_train_score for example) 
    Mandatory parameters are:
        results_df: A dataframe built from GridSearchCV.cv_results_
        x_param: The column name of the results_df dataframe to be used as X axis
        y_param: An array of column to be plotted on the Y axis. Those values must be %.
    Optionnal parameters:
        semilogx: If True, the X data points are plotted using a log10 scale
        xlabel: Label of the X axis
        ylabel: Label of the Y axis
        title: Title of the graph
        figsize: Size ot the graph
        std_param: A dict with key=y_param element and value the corresponding std deviation column name.
            This parameters is used to draw the std deviation of the y_params as a filled area around the data plot
            
    The function will also determine, for each of the y_param to be plotted, which is the plot with the highest
    y_param value, and use the coordinates to draw a red cross on the plotted line, along with horizontal and
    vertical lines to the X and Y axis.
    For that purpose, the function first sort the results_df dataframe using the x_param column in ascending order.
    """
    # Order dataframe by xparam value
    temp_df=results_df.sort_values(x_param, ascending=True)

    # Define figsize
    plt.figure(figsize=figsize)

    # Store x_min, x_max, y_min and y_max values to set xlimit and ylimit of the graph
    x_min=0
    y_min=100
    x_max=0
    y_max=0
    
    # Loop for each yparam plot
    for i in y_param:
        # Find indices of  the best y value
        best_idx=temp_df[i].idxmax()

        # Get best x information
        best_x = temp_df[x_param][best_idx]
        # Get x plots
        x_values=temp_df[x_param]
        # Store x_min and x_max if needed
        if x_min>np.min(x_values):
            x_min=np.min(x_values)
        if x_max<np.max(x_values):
            x_max=np.max(x_values)



        # get best y information
        best_y=temp_df[i][best_idx]*100 # Multiply by 100 to get %
        # Get y plots
        y_values=temp_df[i]*100
        # Store y_min and y_max if needed
        if y_min>np.min(y_values):
            y_min=np.min(y_values)-y_values.std()
        if y_max<np.max(y_values):
            y_max=np.max(y_values)+y_values.std()

        if semilogx:
            plt.semilogx(x_values, y_values, label=i)
        else:
            plt.plot(x_values, y_values, label=i)

        # Draw a cross on the best_x/best_accuracy point
        plt.scatter(best_x, best_y, marker='x', c='red', zorder=10)
        # Write near of the cross the best_y/best_y value
        plt.text(best_x, best_y+0.5, 'x:{:.3f} y:{:.1f}'.format(best_x, best_y))
        plt.plot([best_x, best_x], [0, best_y], c='red', alpha=0.5, linestyle='--')
        plt.plot([np.min(x_values), best_x], [best_y, best_y], c='red', alpha=0.5, linestyle='--')
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.ylim(bottom=y_min, top=y_max+1)
    # plt.xlim(left=x_min, right=x_max)
    plt.legend()
    plt.show()

