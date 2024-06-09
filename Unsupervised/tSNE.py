import numpy as np
from sklearn.metrics import pairwise_distances

class tSNE:
    def __init__(self, n_components=2, perplexity=30, n_iter=1000, learning_rate=200):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def _hbeta(self, D, beta):
        P = np.exp(-D * beta)
        sumP = np.sum(P)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P

    def _x2p(self, X, tol=1e-5):
        (n, d) = X.shape
        D = pairwise_distances(X, squared=True)
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(self.perplexity)

        for i in range(n):
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            (H, thisP) = self._hbeta(Di, beta[i])

            Hdiff = H - logU
            tries = 0

            while np.abs(Hdiff) > tol and tries < 50:
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2
                    else:
                        beta[i] = (beta[i] + betamax) / 2
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2
                    else:
                        beta[i] = (beta[i] + betamin) / 2

                (H, thisP) = self._hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

        return P

    def fit_transform(self, X):
        (n, d) = X.shape
        P = self._x2p(X)
        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)

        Y = np.random.randn(n, self.n_components)
        dY = np.zeros((n, self.n_components))
        iY = np.zeros((n, self.n_components))
        gains = np.ones((n, self.n_components))

        for iter in range(self.n_iter):
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            PQ = P - Q

            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (self.n_components, 1)).T * (Y[i, :] - Y), 0)

            if iter < 20:
                momentum = 0.5
            else:
                momentum = 0.8

            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            gains[gains < 0.01] = 0.01

            iY = momentum * iY - self.learning_rate * (gains * dY)
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        return Y
