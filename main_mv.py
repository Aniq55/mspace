from src.MSpace import MSpace
from src.GSpace import GSpace
from src.utils import *
from src.args import *

import logging
import time


# load dataset
dataset = data_loader(args.dataset)
# dataset = get_data(args.dataset)

if args.ind:
    dataset.A = np.copy(np.eye(dataset.n))

# supply the args to the algorithm
if args.algo: # not reporting
    algo = GSpace(dataset, args.train_ratio, args.state_mode, args.sample_mode, args.M_max, args.T_period, args.K)
else:
    algo = MSpace(dataset, args.train_ratio, args.state_mode, args.sample_mode, args.M_max, args.T_period, args.K)

algo.train()
        

t_start = time.perf_counter()
result = algo.test_mv(args.q)
t_end = time.perf_counter()

t_test = (t_end - t_start)/(((1-args.train_ratio)*dataset.T)*dataset.n)


# print results
print('Data:\t',dataset.name)
print('Steps\t', args.q)
print('MAE\t', result["MAE"])
print('RMSE\t', result["RMSE"])
# print('RSQ\t', result["RSQ"])
# print('COS\t', result["COS"])
# print('COR\t', result["COR"])


logging.basicConfig(filename=f"/home/chri6578/Documents/GG_SPP/markovspace/logs/{args.filename}.txt",
                    level=logging.INFO)


algo_name = args.state_mode + '-' + args.sample_mode
if args.ind:
    algo_name = algo_name + '-I'

log_data = [
    algo_name,
    str(args.q),
    str(args.train_ratio),
    args.dataset,
    f"{result['MAE']:.2f}",
    f"{result['RMSE']:.2f}",
    f"{np.log10(t_test):.2f}"
]

log_string = '\t'.join(log_data)
logging.info(log_string)


# logging.info(f"\t{args.state_mode}-{args.sample_mode}\t{args.q}\t{args.train_ratio}\t{args.dataset}\t{result['MAE']:.3f}\t{result['RMSE']:.3f}\t{np.log10(t_test):.2f}")





