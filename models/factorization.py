import numpy as np

from lib.models import Estimator


class SVD(Estimator):
    def __init__(self, factors:int = 100, epochs=20,
                 mean: float = .0, derivation: float = .1,
                 lr: float = .005, reg: float = .02,
                 random_state=None):
        """Perform SVD matrix factorization
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
        self.biase_user = None
        self.biase_item = None
        self.param_user = None
        self.param_item = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        mean = np.mean(y)

        unique_user = np.unique(X[:, 0])
        unique_item = np.unique(X[:, 1])

        # Initialize biases
        biase_user = np.zeros(unique_user.size, dtype=np.double)
        biase_item = np.zeros(unique_item.size, dtype=np.double)

        # Initialize user and item with random state normal distribution
        param_user = self.state.normal(self.init_mean, self.init_dev,
                                       size=(unique_user.size, self.factors))
        param_item = self.state.normal(self.init_mean, self.init_dev,
                                       size=(unique_item.size, self.factors))

        for _ in range(self.epochs):
            for (u, i), r in zip(X, y):
                u = np.where(unique_user == u)[0][0]
                i = np.where(unique_item == i)[0][0]

                # calculate current error
                dot = sum(param_item[i, f] * param_user[u, f] for f in range(self.factors))
                err = r - (mean + biase_user[u] + biase_item[i] + dot)

                # update biases
                biase_user[u] += self.lr * (err - self.reg * biase_user[u])
                biase_item[i] += self.lr * (err - self.reg * biase_item[i])

                param_user[u] += self.lr * (err * param_item[i] - self.reg * param_user[u])
                param_item[i] += self.lr * (err * param_user[u] - self.reg * param_item[u])

                # # update params
                # for f in range(self.factors):
                #     param_user[u, f] += self.lr * (err * param_item[i, f] - self.reg * param_user[u, f])
                #     param_item[i, f] += self.lr * (err * param_user[u, f] - self.reg * param_item[i, f])

        self.mean = mean
        self.unique_user = unique_user
        self.unique_item = unique_item
        self.biase_user = biase_user
        self.biase_item = biase_item
        self.param_user = param_user
        self.param_item = param_item

    def predict(self, X: np.ndarray) \
            -> np.ndarray:

        estimate = np.full(np.size(X, 0), self.mean)
        for e, (u, i) in enumerate(X):
            known_user = u in self.unique_user
            known_item = i in self.unique_item

            if known_user:
                u = np.where(self.unique_user == u)[0][0]
                estimate[e] += self.biase_user[u]

            if known_item:
                i = np.where(self.unique_item == i)[0][0]
                estimate[e] += self.biase_item[i]

            if known_user and known_item:
                estimate[e] += np.dot(self.param_item[i], self.param_user[u])

        return estimate
