import numpy as np

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
        self.Y =
        return np.dot(X, w).reshape((X.shape[0],-1)) 