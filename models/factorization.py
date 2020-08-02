import numpy as np
from tqdm import tqdm

from lib.models import Estimator


class MatrixFactorization(Estimator):
    def __init__(self, factors: int = 100, epochs: int = 20,
                 mean: float = .0, derivation: float = .1,
                 lr: float = .005, reg: float = .02,
                 random_state=None):
        """Perform matrix factorization
        """
        super(self.__class__, self).__init__()

        self.state = random_state or np.random.mtrand._rand
        self.factors = factors
        self.epochs = epochs
        self.init_mean = mean
        self.init_dev = derivation
        self.lr = lr
        self.reg = reg

        self.mean = 0
        self.unique_user = None
        self.unique_item = None
        self.bias_user = None
        self.bias_item = None
        self.param_user = None
        self.param_item = None
        self.user_indexer = None
        self.item_indexer = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Matrix factorization

        :param X: [user, item] information
        :param y: ratings to predict
        """

        mean = np.mean(y)

        unique_user = np.unique(X[:, 0])
        unique_item = np.unique(X[:, 1])

        # Initialize biases
        bias_user = np.zeros(unique_user.size, dtype=np.double)
        bias_item = np.zeros(unique_item.size, dtype=np.double)

        # Initialize user and item with random state normal distribution
        param_user = self.state.normal(self.init_mean, self.init_dev,
                                       size=(unique_user.size, self.factors))
        param_item = self.state.normal(self.init_mean, self.init_dev,
                                       size=(unique_item.size, self.factors))

        self.user_indexer = dict(zip(unique_user, np.arange(unique_user.size)))
        self.item_indexer = dict(zip(unique_item, np.arange(unique_item.size)))

        for _ in tqdm(range(self.epochs)):
            for (u, i), r in zip(X, y):
                u = self.user_indexer[u]
                i = self.item_indexer[i]

                # calculate current error
                dot = sum(param_item[i, f] * param_user[u, f] for f in range(self.factors))
                err = r - (mean + bias_user[u] + bias_item[i] + dot)

                # update biases
                bias_user[u] += self.lr * (err - self.reg * bias_user[u])
                bias_item[i] += self.lr * (err - self.reg * bias_item[i])

                # update parameters
                param_user[u] += self.lr * (err * param_item[i] - self.reg * param_user[u])
                param_item[i] += self.lr * (err * param_user[u] - self.reg * param_item[i])

        self.mean = mean
        self.unique_user = unique_user
        self.unique_item = unique_item
        self.bias_user = bias_user
        self.bias_item = bias_item
        self.param_user = param_user
        self.param_item = param_item

    def predict(self, X: np.ndarray) \
            -> np.ndarray:
        """Make recommendation using bias, param matrix.

        If not known user and known item then return mean value

        :param X: [user, item] information
        :return: Recommendataion matrix
        """

        estimate = np.full(np.size(X, 0), self.mean)
        for e, (u, i) in enumerate(X):
            known_user = u in self.unique_user
            known_item = i in self.unique_item

            if known_user:
                u = self.user_indexer[u]
                estimate[e] += self.bias_user[u]

            if known_item:
                i = self.item_indexer[i]
                estimate[e] += self.bias_item[i]

            if known_user and known_item:
                estimate[e] += np.dot(self.param_item[i], self.param_user[u])

        return estimate
