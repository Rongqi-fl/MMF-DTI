
import argparse

from RunModel import run_model
from model import MMF

parser = argparse.ArgumentParser(prog='MMF')

parser.add_argument('--dataSetName', default='Davis')
parser.add_argument('-m', '--model', default='MMF')
parser.add_argument('-s', '--seed', type=int, default=114514,
                    help='Set the random seed, the default is 114514')
parser.add_argument('-f', '--fold', type=int, default=5,
                    help='Set the K-Fold number, the default is 5')
args = parser.parse_args()


run_model(SEED=args.seed, DATASET=args.dataSetName,
            MODEL=MMF, K_Fold=args.fold, LOSS='PolyLoss' )


