#!/bin/python3
# -*- coding: utf-8 -*-

"""scripts/split.py (2020 Line recruit test)

This is code for 2020 line recruit test problem B.
Author: Bae Jiun, Maybe
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from lib import data


def main(args: argparse.Namespace):
    result = Path(args.result)
    result.mkdir(exist_ok=True, parents=True)

    # Read whole csv file
    content, header = data.read_csv(args.dataset)

    # Split query
    for query in args.split:
        try:
            # Each query must include {name, begin condition, end condition}
            name, begin, end = query.split(',')
            begin, end = map(int, (begin, end))

            cropped = content[(begin <= content[:, -1]) & (content[:, -1] <= end)],
            data.to_csv(str(result.joinpath(f'{name}.csv')), cropped, header=header)

        except ValueError:
            print('format mismatch, each query must be [name,begin_condition,end_condition]')
            print('But got `{query}`')
            continue


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='./data/ml-20m/ratings.csv',
                        help="Dataset directory path")
    parser.add_argument("--result", type=str, default='./data/dataset',
                        help="Spited dataset output path")
    parser.add_argument("--axis", type=int, default=-1,
                        help="Axis for condition")
    parser.add_argument("--split", nargs='+', default=[
                            'dataset1_train,1104505203,1230735592',
                            'dataset1_test,1230735600,1262271552',
                            'dataset2_train,789652004,1388502017',
                            'dataset2_test,1388502017,1427784002',
                            'tiny_train,1104505203,1104805203',
                            'tiny_test,1230735600,1230835600',
                        ], help="Split data range as list. train,2,3 split 2<timestamp<3 as train.csv")

    main(parser.parse_args())
