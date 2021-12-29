# Import some globally used libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# Set some default display for the pandas lib
pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 15)

def load_csv(dataset = 'travel', low_memory=False) -> pd.DataFrame:
    '''
    Open csv file passed as parameter from ./data/ folder.
    
    Returns
    -------
    pandas.DataFrame
    '''    
    return pd.read_csv(os.path.join('data', 'nyc_{}.csv'.format(dataset)), low_memory=low_memory)