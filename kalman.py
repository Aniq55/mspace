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
parser.add_argument('--n_subset', type=int, default=0, help='Number of nodes in the subsets')
parser.add_argument('--train_ratio', type=float, default=0.7, help='Training ratio')
parser.add_argument('--latent_dim', type=float, default=0.8, help='Latent dimension')
parser.add_argument('--shock', action='store_true', help='Whether to include shock')
parser.add_argument('--n_iters', type=int, default=10, help='Number of iterations')
parser.add_argument('--q', type=int, default=1, help='Value for q')

# Parse arguments
args = parser.parse_args()

# Accessing the values
dataset_name = args.dataset
n_subset = args.n_subset
train_ratio = args.train_ratio
latent_dim = args.latent_dim
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

if shock:
    E = create_shock_chain(dataset.X, dataset.T-1)
else:
    E = create_feature_chain(dataset.X)
    
# Train-test split
if n_subset > 0:
    E = E[:, 0:n_subset]
    dataset.n = n_subset
E_train = E[0:T_train]
E_test = E[T_train:-1]


EM_VARS = ['initial_state_mean', 'initial_state_covariance', 'transition_matrices', 
        'transition_covariance', 'observation_matrices', 'observation_covariance']

kf = KalmanFilter(em_vars=EM_VARS, n_dim_obs= dataset.n, n_dim_state= int(latent_dim*dataset.n))

# TRAINING
measurements = [list(e.T) for e in E_train]
kf.em(measurements, n_iter=n_iters)

error_mtx = np.zeros((dataset.T-1-q-T_train, dataset.n, q))

# TESTING
for t in tqdm(range(T_train, dataset.T -q-1)):
    E_t = E[t:t+q].T
    
    E_t_z = [list(e.T) for e in E[t-q-1:t]]
    z = kf.filter(E_t_z)[0][-1]
    
    E_t_pred = kf.sample(n_timesteps=q, initial_state=z)[1].data
    E_t_pred = np.array(E_t_pred).T
    
    # print(E_t.shape, E_t_pred.shape)
    
    if shock:
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
    'kalman-'+shock_vec[args.shock],
    args.dataset,
    str(args.n_iters),
    str(args.q),
    str(args.train_ratio),
    str(args.latent_dim),
    f"{MAE:.2f}",
    f"{RMSE:.2f}"
]

log_string = '\t'.join(log_data)
logging.info(log_string)

