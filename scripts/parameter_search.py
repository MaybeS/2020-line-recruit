#!/bin/python3
# -*- coding: utf-8 -*-

"""Parameter search (2020 Line recruit test)

This is code for 2020 line recruit test problem B.
Author: Bae Jiun, Maybe
"""
from typing import Tuple
import sys
import argparse
from multiprocessing import Pool, cpu_count
from itertools import product
from pathlib import Path
import json

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from lib import criterion, data, seed
from lib.recommender import Recommender

# Using as global constant variable
param_space = None
train, test = None, None
test_header = None
critic = None


def runner(args: argparse.Namespace) \
        -> Tuple[argparse.Namespace, float]:
    global train, test, test_header, critic

    # Fit model, using train data
    model = Recommender(factors=args.factor, epochs=args.epoch,
                        mean=args.mean, derivation=args.dev,
                        lr=args.lr, reg=args.reg)
    model.fit(train[:, :2], train[:, 2])

    # Predict by test data and calculate error
    predictions = model.predict(test[:, :2])
    error = critic(predictions, test[:, 2])
    print(f'RMSE: {error}')

    # Save predictions
    result = test.copy()
    result[:, 2] = predictions
    data.to_csv(args.result, result, header=test_header)
    return args, error


def wrapper(*args) \
        -> Tuple[argparse.Namespace, float]:
    param = dict(zip(param_space.keys(), *args))
    param['result'] = str(result_prefix.joinpath(f"{'-'.join(map(str, param.values()))}.csv"))

    args = argparse.Namespace(**param)

    print(f'Testing param as {args}')
    return runner(args)


def main(args: argparse.Namespace):
    global train, test, test_header

    # Reproducible (Important)
    # An experiment that can not be reproduced can not make any conclusions.
    # So fix random seed before anything else.
    seed(args.seed)

    # Load dataset
    # Provides two dataset loading methods
    # - Load from whole csv and split train, test by condition (slow)
    # - Load each train, test csv (faster)
    #   (Using scripts/split.py to split train, test by condition)
    if args.dataset:
        dataset = data.Dataset(args.dataset)
        train, test = dataset.split_train_test(args.mode)
        test_header = dataset.rating_headers

    else:
        train, train_header = data.read_csv(args.train)
        test, test_header = data.read_csv(args.test)

    # Find param in search space
    params = list(product(*param_space.values()))

    if args.size:
        indexes = np.random.choice(len(params), args.size, replace=False)
        params = [params[i] for i in indexes]

    print(f'Search space: {len(params)}')
    print(f'Param: {param_space}')

    with Pool(args.cpu or cpu_count()) as pool:
        results = pool.map(wrapper, params)

    best, *_ = sorted(results, key=lambda x: x[1])
    print(f'Best RMSE: {best[1]}')
    print(f'param: {best[0]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Provide single csv file and split automatically
    parser.add_argument("--dataset", type=str, default='', required=False,
                        help="Dataset path")
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'test', 'tiny'],
                        help="Dataset load mode")

    # Provide each train, test dataset
    parser.add_argument("--train", type=str, default='', required=False,
                        help="Train dataset directory path")
    parser.add_argument("--test", type=str, default='', required=False,
                        help="Test dataset directory path")

    parser.add_argument("--search", type=str, default='./data/search.json', required=True,
                        help="Path to search parameters")
    parser.add_argument("--result", type=str, default='./results', required=False,
                        help="Result directory path")

    parser.add_argument("--cpu", type=int, default=0, required=False,
                        help="# of processors to use")
    parser.add_argument("--size", type=int, default=0, required=False,
                        help="Size of search space")

    parser.add_argument('-s', '--seed', required=False,
                        default=42,
                        help="The answer to life the universe and everything")
    parser.add_argument('--criterion', type=str, default='RMSE', choices=['RMSE'],
                        help="The answer to life the universe and everything")

    default_args = parser.parse_args()

    with open(default_args.search) as f:
        param_space = json.load(f)

    result_prefix = Path(default_args.result)
    result_prefix.mkdir(exist_ok=True, parents=True)

    # Set criterion as RMSE
    critic = criterion.get(default_args.criterion)()

    main(default_args)
