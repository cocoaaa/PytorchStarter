import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from collections import Counter
from sklearn.externals import joblib

#Globals
#ROOT = plib.Path("/Users/hjsong/Workspace/Luna")
#nRuns = 5 #number of test runs per unit
#RANDOM_SEED = 15;

class DataGenerator:
    def __init__(self, n, xmin=-20, xmax=20):
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.X = None
        self.Y = None
        
    def getX(n, xmin=-20, xmax=20):
        """ 
        Creates (n,2) data matrix for Exercise 3-2
        - First column is a vector of `n` points equally spaces in (-20, 20)
        - Second column is a vector of elementwise squared of the first column
        """
        X = np.zeros((n,2))
        X[:,0] = np.linspace(-20, 20, num=n)
        X[:,1] = X[:,0]**2
        self.X = X
        return self.X

    def getY(true_w, true_b):
        """ 
        Creates (n,1) matrix for target of X
        with true weights, w = [w1, w2] and b
        """
        w = np.r_[true_b, true_w]
        X = np.c_[np.ones((X.shape[0], 1)), X]
        #self.Y =
        return np.dot(X, w).reshape((X.shape[0],-1)) 
    

class LunaCsvDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        - csv_file (string): path to the csv file
        - root_dir (string or pathlib.Path object): directory with all the images
        - transform (callable, optional): optional transform to be applied to a sample
        """
        all_data = pd.read_csv(csv_file, header=0, index_col=0)
        is_not_fraud = [False if 'Fraud' in all_data.state[i] else True for i in range(len(all_data))]
        all_data = all_data[is_not_fraud]        
        self.data = np.array(all_data.drop('state', axis=1)) 
        self.target = np.array(all_data['state'])
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        "returns a sample of format (feature 1,...,featureD, class)"
        return (self.data[idx, :], self.target[idx])
      

class LunaPickleDataset(Dataset):
    def __init__(self, pk_file, root_dir, transform=None):
        """
        - pk_file (string): path to the pickled file
        - root_dir (string): directory with all the images
        - transform (callable, optional): optional transform to be applied to a sample
        """
        all_data = pd.read_csv(csv_file, header=None) #todo
        is_not_fraud = [False if 'Fraud' in all_data.state[i] else True for i in range(len(all_data))]
        all_data = all_data[is_not_fraud]
        self.data = np.array(all_data.iloc[:, 1:-1]) #ignore first column (patient ID)
        self.label_dict = {'KGB':0, 'Cracked':1, 'Repaired':2, 'NoPSA':0, }
        self.target = np.array(all_data.state.map())
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        "returns a sample of format (feature 1,...,featureD, class)"
        return (self.data[idx, :], self.target[idx])
    
    def set_label_dict(self, new_dict):
        self.label_dict = new_dict
            