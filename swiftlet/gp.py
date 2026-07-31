"""
Pure Python Gaussian Process — zero dependencies outside the standard library.

v2 fixes:
  - log_marginal_likelihood() enables hyperparameter optimization (fix #2)
  - optimize_kernel() does grid search over (length_scale, variance) pairs
  - Numerical stability: jitter on the diagonal, clamped variance
"""

import math


# ─── Linear Algebra ──────────────────────────────────────────────────

def cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(i + 1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
            if i == k:
                val = A[i][i] - tmp_sum + 1e-8
                L[i][k] = math.sqrt(max(val, 1e-12))
            else:
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return L


def forward_sub(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        tmp = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - tmp) / L[i][i]
    return y


def backward_sub(U, y):
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        tmp = sum(U[j][i] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - tmp) / U[i][i]
    return x


def solve_cholesky(L, b):
    y = forward_sub(L, b)
    return backward_sub(L, y)


# ─── Kernel ──────────────────────────────────────────────────────────

class RBFKernel:
    def __init__(self, length_scale=1.0, variance=1.0):
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, x1, x2):
        sqdist = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return self.variance * math.exp(
            -0.5 * sqdist / (self.length_scale ** 2)
        )


# ─── GP Regressor ────────────────────────────────────────────────────

class GaussianProcessRegressor:
    def __init__(self, kernel, noise=1e-3):
        self.kernel = kernel
        self.noise = noise
        self.X_train = []
        self.y_train = []
        self.L = None
        self.alpha = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        n = len(X)
        if n == 0:
            return

        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                K[i][j] = self.kernel(X[i], X[j])
                if i == j:
                    K[i][j] += self.noise

        self.L = cholesky(K)
        self.alpha = solve_cholesky(self.L, y)

    def predict(self, X_test):
        if not self.X_train:
            return (
                [0.0] * len(X_test),
                [self.kernel.variance] * len(X_test),
            )

        n_test = len(X_test)
        means = [0.0] * n_test
        variances = [0.0] * n_test

        for i, x in enumerate(X_test):
            k_star = [self.kernel(x, xt) for xt in self.X_train]
            means[i] = sum(k * a for k, a in zip(k_star, self.alpha))
            v = forward_sub(self.L, k_star)
            v_dot_v = sum(v_j ** 2 for v_j in v)
            variances[i] = self.kernel(x, x) - v_dot_v

        return means, variances

    # FIX #2: log marginal likelihood for hyperparameter optimization.
    #
    # log p(y|X,θ) = -½ y^T (K+σ²I)^{-1} y - ½ log|K+σ²I| - n/2 log(2π)
    #
    # Using the Cholesky factor L:
    #   y^T (K+σ²I)^{-1} y = ||L^{-1} y||²   (already computed as alpha)
    #   log|K+σ²I|          = 2 Σ log L_ii
    #
    # This is the objective we maximize when choosing kernel hyperparameters.
    # A higher marginal likelihood means the kernel is better calibrated to
    # the data — it captures the right level of smoothness.

    def log_marginal_likelihood(self) -> float:
        if self.L is None or not self.y_train:
            return -float("inf")

        n = len(self.y_train)
        # Data fit: -½ y^T K^{-1} y = -½ ||L^{-1} y||²
        # Since alpha = K^{-1} y, this is -½ y^T alpha
        data_fit = -0.5 * sum(
            yi * ai for yi, ai in zip(self.y_train, self.alpha)
        )

        # Complexity: -½ log|K| = -Σ log L_ii
        log_det = -sum(math.log(self.L[i][i]) for i in range(n))

        # Constant: -n/2 log(2π)
        constant = -0.5 * n * math.log(2 * math.pi)

        return data_fit + log_det + constant


# ─── Acquisition ─────────────────────────────────────────────────────

def ucb(mean, variance, beta=2.0):
    """Upper Confidence Bound acquisition function."""
    return mean + beta * math.sqrt(max(variance, 0.0))


# FIX #2: Hyperparameter optimization via grid search.
#
# Instead of fixing length_scale=0.5 and variance=10.0, we evaluate
# multiple candidates and pick the one with the highest marginal
# likelihood.  This is type-II maximum likelihood (empirical Bayes).
#
# The grid is coarse (5×5 = 25 candidates) because each candidate
# requires fitting a GP (O(n³)).  For n=50, that's 25 × 125K ≈ 3M
# operations — fast enough in Python.

def optimize_kernel(X, y, noise=1.0) -> RBFKernel:
    """
    Grid search over kernel hyperparameters to maximize log marginal
    likelihood.  Returns the best kernel found.
    """
    length_scales = [0.1, 0.3, 0.5, 1.0, 2.0]
    variances = [1.0, 5.0, 10.0, 20.0, 50.0]

    best_ll = -float("inf")
    best_kernel = RBFKernel(length_scale=0.5, variance=10.0)

    for ls in length_scales:
        for vs in variances:
            kernel = RBFKernel(length_scale=ls, variance=vs)
            gp = GaussianProcessRegressor(kernel=kernel, noise=noise)
            gp.fit(X, y)
            ll = gp.log_marginal_likelihood()
            if ll > best_ll:
                best_ll = ll
                best_kernel = kernel

    return best_kernel
