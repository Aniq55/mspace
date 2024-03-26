import argparse
from tqdm import tqdm
import logging
import time
from pykalman import KalmanFilter

from src.utils import *


# ARGPARSER
parser = argparse.ArgumentParser(description='Argument Parser for your program.')

# Define arguments
parser.add_argument('--dataset', type=str, default='chickenpox', help='Name of the dataset')
parser.add_argument('--n_subset', type=int, default=0, help='Number of subsets')
parser.add_argument('--train_ratio', type=float, default=0.7, help='Training ratio')
parser.add_argument('--shock', action='store_true', help='Whether to include shock')
parser.add_argument('--n_iters', type=int, default=10, help='Number of iterations')
parser.add_argument('--q', type=int, default=1, help='Value for q')

# Parse arguments
args = parser.parse_args()

# Accessing the values
dataset_name = args.dataset
n_subset = args.n_subset
train_ratio = args.train_ratio
shock = args.shock
n_iters = args.n_iters
q = args.q

def create_shock_chain(X, T):
    shocks_mtx = X[1:]-X[0:T]
    return np.array([x.flatten() for x in shocks_mtx])

def create_feature_chain(X):
    return np.array([x.flatten() for x in X])

# load dataset
dataset = data_loader(dataset_name)
T_train = int(train_ratio*dataset.T)

Z = np.reshape(dataset.X, (dataset.T, dataset.n*dataset.d, 1 ))

if shock:
    E = create_shock_chain(Z, dataset.T-1)
else:
    E = create_feature_chain(Z)
    
# Train-test split
dataset.n = dataset.n*dataset.d
if n_subset > 0:
    E = E[:, 0:n_subset]
    dataset.n = n_subset
E_train = E[0:T_train]
E_test = E[T_train:-1]

# print(dataset.X.shape)
# print(E_train.shape)
# print(E.shape, dataset.n, dataset.T)

EM_VARS = ['initial_state_mean', 'initial_state_covariance', 'transition_matrices', 
        'transition_covariance', 'observation_matrices', 'observation_covariance']

KF = []
for i in range(dataset.n):
    KF.append(KalmanFilter(em_vars=EM_VARS, n_dim_obs= 1, n_dim_state= 1))

# TRAINING
for i in tqdm(range(dataset.n)):
    measurements = [e[i].T for e in E_train]
    KF[i].em(measurements, n_iter=n_iters)

error_mtx = np.zeros((dataset.T-1-q-T_train, dataset.n, q, 1))

# TESTING
for t in tqdm(range(T_train, dataset.T -q-1)):
    E_t = np.expand_dims(E[t:t+q].T, axis=2)
    E_t_z = E[t-q-1:t].T
    
    E_t_pred = np.zeros((dataset.n,q,1))
    
    # print(E_t.shape, E_t_pred.shape )

    for i in range(dataset.n):
        z = KF[i].filter(E_t_z[i])[0][-1]
        E_t_pred[i] = KF[i].sample(n_timesteps=q, initial_state=z)[1].data
    
    if shock:
        if q == 1:
            error_mtx[t-T_train] = np.abs( np.cumsum(E_t, axis=0) - np.cumsum(E_t_pred,axis=0) )
        else:
            error_mtx[t-T_train] = np.abs( np.cumsum(E_t, axis=0) - np.cumsum(E_t_pred,axis=0) ) # shock
    else:
        error_mtx[t-T_train] = np.abs( E_t - E_t_pred ) # feature

# RESULTS
MAE = np.mean(error_mtx)
RMSE = np.mean(np.sqrt(np.mean(np.square(error_mtx), axis=1)))


logging.basicConfig(filename=f"/home/chri6578/Documents/GG_SPP/markovspace/logs/kalman.txt",
                    level=logging.INFO)

shock_vec = ['feature', 'shock']
log_data = [
    'kalman-ind-'+shock_vec[args.shock],
    args.dataset,
    str(args.n_iters),
    str(args.q),
    str(args.train_ratio),
    f"{MAE:.2f}",
    f"{RMSE:.2f}"
]

log_string = '\t'.join(log_data)
logging.info(log_string)

