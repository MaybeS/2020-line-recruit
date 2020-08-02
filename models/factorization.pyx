# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
import numpy as np
cimport numpy as np
from tqdm import tqdm

from lib.models import Estimator


class MatrixFactorization(Estimator):
    def __init__(self, factors=100, epochs=20,
                 mean=.0, derivation=.1, lr=.005,
                 reg=.02, random_state=None):
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

        # typedef for cython optimization
        cdef list uniques
        cdef list biases
        cdef list params

        cdef np.ndarray[np.long_t] unique_user, unique_item
        cdef np.double_t[::1] bias_user, bias_item
        cdef np.double_t[:, ::1] param_user, param_item

        cdef int u, i, f
        cdef double r, err, dot, param_userf, param_itemf
        cdef double mean = np.mean(y)

        cdef double lr = self.lr
        cdef double reg = self.reg

        unique_user = np.unique(X[:, 0])
        unique_item = np.unique(X[:, 1])
        bias_user = np.zeros(unique_user.size, np.double)
        bias_item = np.zeros(unique_item.size, np.double)
        param_user = self.state.normal(self.init_mean, self.init_dev,
                                      (unique_user.size, self.factors))
        param_item = self.state.normal(self.init_mean, self.init_dev,
                                      (unique_item.size, self.factors))

        self.user_indexer = dict(zip(unique_user, np.arange(unique_user.size)))
        self.item_indexer = dict(zip(unique_item, np.arange(unique_item.size)))

        # The code below may look a bit different for
        # cython's limitations and computational efficiency,
        # but it does the same thing.
        for _ in tqdm(range(self.epochs)):
            for (u, i), r in zip(X, y):
                u = self.user_indexer[u]
                i = self.item_indexer[i]

                # calculate current error
                dot = 0
                for f in range(self.factors):
                    dot += param_item[i, f] * param_user[u, f]
                err = r - (mean + bias_user[u] + bias_item[i] + dot)

                # update biases
                bias_user[u] = bias_user[u] + lr * (err - reg * bias_user[u])
                bias_item[i] = bias_item[i] + lr * (err - reg * bias_item[i])

                # update parameters
                for f in range(self.factors):
                    param_user[u, f] += lr * (err * param_item[i, f] - reg * param_user[u, f])
                    param_item[i, f] += lr * (err * param_user[u, f] - reg * param_item[i, f])

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

        # typedef for cython optimization
        cdef np.ndarray[np.double_t] estimate
        cdef double mean = self.mean
        cdef bint known_user, known_item

        cdef int e, u, i
        cdef np.ndarray[np.long_t] unique_user = self.unique_user
        cdef np.ndarray[np.long_t] unique_item = self.unique_item
        cdef np.double_t[::1] bias_user = self.bias_user
        cdef np.double_t[::1] bias_item = self.bias_item
        cdef np.double_t[:, ::1] param_user = self.param_user
        cdef np.double_t[:, ::1] param_item = self.param_item

        estimate = np.full(np.size(X, 0), mean)
        for e, (u, i) in enumerate(X):
            known_user = u in unique_user
            known_item = i in unique_item

            if known_user:
                u = self.user_indexer[u]
                estimate[e] += bias_user[u]

            if known_item:
                i = self.item_indexer[i]
                estimate[e] += bias_item[i]

            if known_user and known_item:
                estimate[e] += np.dot(param_item[i], param_user[u])

        return estimate
