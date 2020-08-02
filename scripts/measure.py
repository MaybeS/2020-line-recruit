#!/bin/python3
# -*- coding: utf-8 -*-

"""scripts/measure.py (2020 Line recruit test)

This is code for 2020 line recruit test problem B.
Author: Bae Jiun, Maybe
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from lib import measure, data


def main(args: argparse.Namespace):
    label, *_ = data.read_csv(args.label)
    pred, *_ = data.read_csv(args.prediction)

    # define criterion as RMSE
    criterion = measure.get(args.method)()
    result = criterion(label[:, 2], pred[:, 2])

    print(f'{args.method}: {result}')


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default='./results/answer.csv', required=False,
                        help="Label csv file")
    parser.add_argument("--prediction", type=str, default='./results/result.csv', required=False,
                        help="Prediction csv file")

    parser.add_argument("--method", type=str, default='RMSE', required=False,
                        help="Measure method")

    main(parser.parse_args())
