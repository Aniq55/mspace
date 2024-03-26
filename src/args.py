import argparse

parser = argparse.ArgumentParser(description="MSPACE")

parser.add_argument('-d', '--dataset', dest='dataset', action='store', type=str, required=True, help='Dataset name')
                    # choices=['tennis', 'wikimath', 'pedalme', 'chickenpox', 'pems03', 'pems04', 'pems07', 'pems08', 'pemsbay', 'metrla'])
parser.add_argument('-q', '--steps', dest='q', action='store', type=int, default= 1, required=True, help='Forecast steps')
parser.add_argument('-K', '--hops', dest='K', action='store', type=int, default=1, required=True, help='Hop count')
parser.add_argument('-M', '--queuesize', dest='M_max', action='store', type=int, default=10, required=True, help='Size of the queue')
parser.add_argument('-r', '--ratio', dest='train_ratio', action='store', type=float, default=0.8, required=True, help='Ratio of the train split')
parser.add_argument('-T', '--period', dest='T_period', action='store', type=int, required=False, help='Time period')
parser.add_argument('--statemode', dest='state_mode', action='store', type=str, default='S', required=True, help='State function',
                    choices=['S', 'T', 'ST'])
parser.add_argument('--samplemode', dest='sample_mode', action='store', type=str, default='mean', required=True, help='Sampling function',
                    choices=['mean', 'normal'])
parser.add_argument('-f', '--file', dest='filename', action='store', type=str, required=True, help='Save file')
parser.add_argument('-I', '--ind', dest='ind', action='store', type=int, default=0, required=False, help='Save file')
parser.add_argument('-a', '--algo', dest='algo', action='store', type=int, default=0, required=False, help='MSpace/GSpace')

args = parser.parse_args()
