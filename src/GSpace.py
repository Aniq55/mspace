import numpy as np
from tqdm import tqdm

from src.Arm import Arm
from src.utils import *


class GSpace:
    def __init__(self, dataset, train_ratio, state_mode, sample_mode, M_max, T_period=None, K=1):
        self.n = dataset.n
        self.d = dataset.d
        self.T = dataset.T -1
        
        self.T_train = int(train_ratio*self.T)
        self.state_mode = state_mode
        self.sample_mode = sample_mode
        
        self.K = K
        self.M_max = M_max
        
        self.T_period = T_period
        
        self.shock_chain = self.create_shock_chain(dataset.X)
        self.arm_list = self.create_arm_list(dataset)
        
        
        
    def create_shock_chain(self, X):
        shocks_mtx = X[1:]-X[0:self.T]
        # if self.bin_class:
        #     shocks_mtx = X
        return np.array([x.flatten() for x in shocks_mtx])
        
    def create_arm_list(self, dataset):
        # A = (np.abs(dataset.A) > 0).astype('float')
        
        arm_list = [tuple(range(dataset.n))]
        
        # V = set(arm_list) # remove duplicates
        return [Arm(list(U), self.d, self.M_max, self.state_mode, self.sample_mode, self.T_period) for U in arm_list] # Arm objects
    
    def train(self):
        for t in tqdm(range(self.T_train)):
            for ARM in self.arm_list: # update the set and sample from the list
                ARM.update(self.shock_chain[t], self.shock_chain[t+1], t)
                
    def test(self, q):
        # Online learning / Testing
        error_mtx = np.zeros((self.T-1-q-self.T_train, self.n, self.d, q))
        # r_square_mtx = np.zeros((self.T-1-q-self.T_train, self.n, self.d))
        # cosine_mtx = np.zeros((self.T-1-q-self.T_train, self.n, self.d))
        # corr_mtx = np.zeros((self.T-1-q-self.T_train, self.n, self.d))

        for t in tqdm(range(self.T_train, self.T-q-1)):
            shock_now = self.shock_chain[t]
            shock_next = self.shock_chain[t+1]
            
            for i, ARM in enumerate(self.arm_list):
                # do not need extract at this point
                # shock_next_arm = np.array([ARM.extract(ARM.preprocess(self.shock_chain[t+m+1]), i) for m in range(q)]).T
                # shock_next_pred = ARM.extract(ARM.sample_multistep(shock_now, q, t), i)
                
                shock_next_arm = np.array([ARM.preprocess(self.shock_chain[t+m+1]) for m in range(q)]).T
                shock_next_pred = ARM.sample_multistep(shock_now, q, t)
                
                obs_arm = vec_cumsum(shock_next_arm)
                obs_pred =  vec_cumsum(shock_next_pred)
                error_mtx[t-self.T_train] = (np.abs(obs_arm - obs_pred)).reshape(self.n, self.d, q)
                # r_square_mtx[t-self.T_train][i] = r_square(obs_arm, obs_pred)
                # cosine_mtx[t-self.T_train][i] = cos_similarity(obs_arm, obs_pred)
                # corr_mtx[t-self.T_train][i] = correlation(obs_arm, obs_pred)

            for ARM in self.arm_list:
                ARM.update(self.shock_chain[t], self.shock_chain[t+1], t)
            

        # Metrics
        MAE = np.mean(error_mtx)
        RMSE = np.mean(np.sqrt(np.mean(np.square(error_mtx), axis=1)))
        # RSQ = np.mean(r_square_mtx)
        # COS = np.mean(cosine_mtx)
        # COR = np.mean(corr_mtx)
        
        return {'RMSE': RMSE, 'MAE': MAE} # , 'RSQ': RSQ, 'COS': COS, 'COR': COR}
    
    def run(self, q):
        self.train()
        
        return self.test(q)