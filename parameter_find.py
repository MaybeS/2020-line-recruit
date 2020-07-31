import argparse
from multiprocessing import Pool
from itertools import product
from collections import OrderedDict
from pathlib import Path

from lib import measure, data
from lib.recommender import Recommender


results = Path('./results')
param_default = OrderedDict({
    'dataset': './data/ml-20m',
})
param_test = OrderedDict({
    'factor': [100, 50, 150, 25, 200],
    'epoch': [20, 100, 150, 200, 10],
    'mean': [.0],
    'dev': [.1, .2, .05],
    'lr': [.005, .001, .01, .0001, .1],
    'reg': [.02, .01, .1, .05],
})
train, test = None, None


def main(args: argparse.Namespace):
    global train, test
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
    result = test.copy()
    result[:, 2] = predictions
    data.to_csv(args.result, result,
                header=dataset.rating_headers)


def wrapper(*args):
    param = dict(zip(param_test.keys(), *args))
    param['result'] = str(results.joinpath(f"{'-'.join(map(str, param.values()))}.csv"))
    param.update(param_default)

    args = argparse.Namespace(**param)

    print(f'Testing param as {args}')
    main(args)


if __name__ == '__main__':
    results.mkdir(exist_ok=True, parents=True)
    dataset = data.Dataset(param_default['dataset'])
    train, test = dataset.split_train_test()

    with Pool() as pool:
        pool.map(wrapper, product(*param_test.values()))
