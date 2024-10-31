import numpy as np
from src.distributions import Gaussian, Mean, Bernoulli
import pickle
from src.TemporalGraph import *

DATA_LOC = '/home/chri6578/Documents/DATA/'

FILES = {
    'pems03': ['PEMS03/PEMS03.npz', 'pems03_spatial_distance.npy'],
    'pems04': ['PEMS04/PEMS04.npz', 'pems04_spatial_distance.npy'],
    'pems07': ['PEMS07/PEMS07.npz', 'pems07_spatial_distance.npy'],
    'pems08': ['PEMS08/PEMS08.npz', 'pems08_spatial_distance.npy'],
    'pemsbay': ['PEMSBAY/pems_bay.npz', 'pemsbay_spatial_distance.npy'],
    'metrla': ['METRLA/metr_la.npz', 'metrla_spatial_distance.npy'],
    'stocks': ['STOCKS/stocks.npz', 'STOCKS/stock.csv'],
}

data_keys = ['pems03', 'pems04', 'pems07', 'pems08', 'pemsbay', 'metrla', 'chickenpox', 'pedalme', 'wikimath']
data_keys_dyn = ['tennis', 'engcovid']


def data_loader(data_key):
    print(f"Loading from /home/chri6578/Documents/mspace/dataset/{data_key}.pkl ")
    with open(f"/home/chri6578/Documents/mspace/dataset/{data_key}.pkl", 'rb') as file:
        G_ = pickle.load(file)
    
    if len(G_['A'].shape)==3:
        return TemporalGraph(X = G_['X'], A = G_['A'][0], name=data_key)
    
    return TemporalGraph(X = G_['X'], A = G_['A'], name=data_key)

def get_data(dataset, sampling=None):
    X = np.load(DATA_LOC + FILES[dataset][0])
    # from GRAM-ODE
    dist_matrix = np.load(DATA_LOC + FILES[dataset][1])
    num_node = dist_matrix.shape[0]
    std = np.std(dist_matrix[dist_matrix != np.float32('inf')])
    mean = np.mean(dist_matrix[dist_matrix != np.float32('inf')])
    
    # works OK for METR-LA and PEMS-BAY
    A = (((dist_matrix < 0.3*mean).astype('float') + np.eye(num_node) ) > 0).astype('float')
    
    # works OK for PEMS0*:
    # dist_matrix = (dist_matrix - mean) / std
    # sigma = 1e20
    # sp_matrix = np.exp(- dist_matrix**2 / sigma**2)
    # A = ((sp_matrix > 0.99).astype('float') + np.eye(num_node) > 0).astype('float')

    return TemporalGraph(X = X['data'], A = A, name=dataset)


def str2vec(X):
    return np.array([int(x) for x in X])

def distance_binary(v, w):
    return np.sum(np.abs(str2vec(v)-str2vec(w)))

def vec_cumsum(x):
    T = x.T.shape[0]
    y = np.zeros(x.T.shape)
    for t in range(T):
        y[t] = np.sum(x.T[0:t+1], axis=0) 
    return y.T

# distributions



distribution = {
    'normal': Gaussian,
    'mean': Mean,
    'bern': Bernoulli
}

# state functions

def fn_Psi_spatial(x, t, T_period):
    x = (x>0).astype(int)
    return ''.join([str(y) for y in x.flatten()])

def fn_Psi_temporal(x, t, T_period): 
    t_string = str(int(t%T_period))
    t_string = ''.join(["0"]*(len(str(T_period)) - len(t_string)) ) + t_string
    return t_string 

def fn_Psi_spatiotemporal(x, t, T_period):
    x = (x>0).astype(int)  
    t_string = str(int(t%T_period))
    t_string = ''.join(["0"]*(len(str(T_period)) - len(t_string)) ) + t_string
    return ''.join([str(y) for y in x.flatten()]) + t_string 


fn_Psi = {
    'S': fn_Psi_spatial,
    'T': fn_Psi_temporal,
    'ST': fn_Psi_spatiotemporal
}


# state distance functions

def find_closest_spatial(A, x, T_period):
    # find an element y in A such that distance(x,y) is minimum
    A_vec = np.array([str2vec(y) for y in A])
    A_vec_diff = A_vec - str2vec(x)
    
    diff = np.sum(np.abs(A_vec_diff), axis=1)
    return A[np.argmin(diff)]

def find_closest_temporal(A, x, T_period): # uses T_PERIOD
    T_period_len = len(str(T_period))
    
    A_vec_time = np.array([np.int32(y) for y in A])
    A_vec_diff_time = np.abs(A_vec_time - np.int32(x))
    
    return A[np.argmin(A_vec_diff_time)]

def find_closest_spatiotemporal(A, x, T_period): # uses T_PERIOD
    T_period_len = len(str(T_period))
    # find an element y in A such that distance(x,y) is minimum
    A_vec = np.array([str2vec(y[:-T_period_len]) for y in A])
    A_vec_diff = A_vec - str2vec(x[:-T_period_len])
    
    A_vec_time = np.array([np.int32(y[-T_period_len:]) for y in A])
    A_vec_diff_time = np.abs(A_vec_time - np.int32(x[-T_period_len:]))
    
    mask = (1e5*(A_vec_diff_time < 1) + 1) + np.abs(A_vec_diff_time) 
    #makes all the indices that are beyond a time difference of 5, really high (1e5)
    
    diff = np.sum(np.abs(A_vec_diff), axis=1)*mask
    
    return A[np.argmin(diff)]

find_closest = {
    'S': find_closest_spatial,
    'T': find_closest_temporal,
    'ST': find_closest_spatiotemporal
}

def r_square(x, y):
    
    if len(x.shape) == 2:
        (d, q) = x.shape
        
        x_mean = np.mean(x, axis=1)
        x_mean = np.repeat(x_mean[:, np.newaxis], q, axis=1)
        z = np.square(x-x_mean)
        z_sum = np.sum(z, axis=1)
        
        w = np.square(x-y)
        w_sum = np.sum(w, axis=1)        
        
    if len(x.shape) == 3:
        (n, d, q) = x.shape
        
        x_mean = np.mean(x, axis=2)
        x_mean = np.repeat(x_mean[:, :, np.newaxis], q, axis=2)
        z = np.square(x-x_mean)
        z_sum = np.sum(z, axis=2)
        
        w = np.square(x-y)
        w_sum = np.sum(w, axis=2)
    
    if len(x.shape) == 4:
        (tau, n, d, q) = x.shape

        x_mean = np.mean(x, axis=3)
        x_mean = np.repeat(x_mean[:, :, :, np.newaxis], q, axis=3)
        z = np.square(x-x_mean)
        z_sum = np.sum(z, axis=3)
        
        w = np.square(x-y)
        w_sum = np.sum(w, axis=3)
    

    v = 1-  w_sum/z_sum
    
    # return v.flatten()
    return np.mean(v.flatten())

def cos_similarity(x,y):
    (d, q) = x.shape
    
    cos = np.sum(x*y, axis=1)/(np.sqrt(np.sum(x*x, axis=1))*np.sqrt(np.sum(y*y, axis=1)))
    cos = (1/q)*np.mean(cos)
    
    return cos

def correlation(x,y):
    (d, q) = x.shape
    x_mean = np.mean(x, axis=1)
    x_mean = np.repeat(x_mean[:, np.newaxis], q, axis=1)
    
    y_mean = np.mean(y, axis=1)
    y_mean = np.repeat(y_mean[:, np.newaxis], q, axis=1)
    
    corr = np.sum((x - x_mean)*(y - y_mean), axis=1)/(np.sqrt(np.sum((x - x_mean)*(x - x_mean), axis=1))*np.sqrt(np.sum((y - y_mean)*(y - y_mean), axis=1)))
    corr = (1/q)*np.mean(corr)
    
    return corr
