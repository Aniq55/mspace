import numpy as np

class TemporalGraph:
    def __init__(self, X, A, name=None, adj_static=True, ):
        self.name = name
        self.X = X
        self.A = A
        
        self.n = A.shape[0]
        self.T = X.shape[0]
        self.d = X.shape[2]
        
        self.adj_static = adj_static

    def info(self):
        """prints the dataset information
        """
        print(f'Temporal graph\n name: \t {self.name} \n n: \t {self.n} \n d: \t {self.d}\n T: \t {self.T}')
