#!/bin/python3
# -*- coding: utf-8 -*-

"""Recommendation (2020 Line recruit test)

This is code for 2020 line recruit test problem B.
Author: Bae Jiun, Maybe
"""

import argparse

from lib.data import Dataset
from lib.measure import RMSE
from lib.recommender import Recommender


def main(args: argparse.Namespace):
    dataset = Dataset(args.dataset)
    train, test = dataset.split_train_test()

    # fit model, using train data
    model = Recommender(factors=args.factor, epochs=args.epoch,
                        mean=args.mean, derivation=args.dev,
                        lr=args.lr, reg=args.reg)
    model.fit(train[:, :2], train[:, 2])

    # predict and calculate RMSE
    predicts = model.predict(test[:, :2])
    print('RMSE:', RMSE()(predicts, test[:, 2]))

    # save predictions
    # test = test.drop(test.columns[-1], axis=1)
    # test[test.columns[-1]] = pd.Series(predicts)
    # test.to_csv(splitext(test_file)[0] + '.base_prediction.txt', sep='\t', index=None, header=None)


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset directory path")

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
