#!/bin/python3
# -*- coding: utf-8 -*-

"""Recommendation (2020 Line recruit test)

This is code for 2020 line recruit test problem B.
Author: Bae Jiun, Maybe
"""

import argparse

from lib import measure, data, seed
from lib.recommender import Recommender


def main(args: argparse.Namespace):
    seed(args.seed)

    # Load dataset
    if args.dataset:
        dataset = data.Dataset(args.dataset)
        train, test = dataset.split_train_test(args.mode)
        test_header = dataset.rating_headers

    else:
        train, train_header = data.read_csv(args.train)
        test, test_header = data.read_csv(args.test)

    # define criterion as RMSE
    criterion = measure.RMSE()
    # fit model, using train data
    model = Recommender(factors=args.factor, epochs=args.epoch,
                        mean=args.mean, derivation=args.dev,
                        lr=args.lr, reg=args.reg)
    model.fit(train[:, :2], train[:, 2])

    # predict and calculate error
    predictions = model.predict(test[:, :2])
    error = criterion(predictions, test[:, 2])
    print(f'RMSE: {error}')

    # save predictions
    test[:, 2] = predictions
    data.to_csv(args.result, test, header=test_header)


if __name__ == '__main__':
    # argument parser
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

    parser.add_argument("--result", type=str, default='result.csv', required=False,
                        help="Result csv file")

    parser.add_argument('-s', '--seed', required=False,
                        default=42,
                        help="The answer to life the universe and everything")

    parser.add_argument("--factor", type=int, default=100,
                        help="size of factor")
    parser.add_argument("--epoch", type=int, default=20,
                        help="epoch size")
    parser.add_argument("--mean", type=float, default=.0,
                        help="initial mean value")
    parser.add_argument("--dev", type=float, default=.1,
                        help="initial derivation value")
    parser.add_argument("--lr", type=float, default=.005,
                        help="learning rate")
    parser.add_argument("--reg", type=float, default=.02,
                        help="regression rate")

    main(parser.parse_args())
