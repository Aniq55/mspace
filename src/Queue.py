import numpy as np
from src.utils import *

class Queue:
    def __init__(self, max_size, state, arm, n_U, d, sample_mode):
        self.queue = []
        
        self.max_size = max_size
        self.state = state # can either be (s) or (s,s')
        self.arm = arm
        self.n_U = n_U 
        self.d = d
        self.index = {u:i for i,u in enumerate(self.arm)}
        
        self.distribution = distribution[sample_mode]()
        
    def dequeue(self):
        if not self.is_empty():
            _ = self.queue.pop(0)

    def enqueue(self, x):
        if len(self.queue) < self.max_size:
            self.queue.append(x)
        else:
            self.dequeue()
            self.queue.append(x)   

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)
    
    def get_numpy(self):
        return np.array(self.queue).T
    
    def update_distribution(self):
        self.distribution.update(self.get_numpy())
