import numpy as np


class LinearRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            tol=0.001,
            random_state=None,
            eta0=0.01,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            shuffle=True,
            batch_size=32
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.coef_ = None
        self.intercept_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def get_penalty_grad(self):
        if self.penalty == "l1":
            return self.alpha * np.sign(self.coef_)
        elif self.penalty == "l2":
            return 2 * self.alpha * self.coef_
        else:
            raise ValueError("penalty must be 'l1' or 'l2'")

    def fit(self, x, y):
        N, M = x.shape
        self.coef_ = np.zeros(M)
        self.intercept_ = 0.0

        if self.early_stopping:
            indices = np.arange(N)
            if self.shuffle:
                np.random.shuffle(indices)

            split_idx = int(N * self.validation_fraction)
            train_idx, val_idx = indices[split_idx:], indices[:split_idx]
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            best_loss = np.inf
            no_improvement_count = 0

        else:
            x_train, y_train = x, y

        for iteration in range(self.max_iter):
            if self.shuffle and not self.early_stopping:
                permutation = np.random.permutation(x_train.shape[0])
                x_train = x_train[permutation]
                y_train = y_train[permutation]

            for batch_start in range(0, x_train.shape[0], self.batch_size):
                x_batch = x_train[batch_start:batch_start + self.batch_size]
                y_batch = y_train[batch_start:batch_start + self.batch_size]

                predictions = self.predict(x_batch)
                error = predictions - y_batch

                coef_grad = ((2 / x_batch.shape[0]) * x_batch.T @ error +
                             self.get_penalty_grad())
                intercept_grad = (2 / x_batch.shape[0]) * np.sum(error)

                self.coef_ -= self.eta0 * coef_grad
                self.intercept_ -= self.eta0 * intercept_grad

            if self.early_stopping:
                val_predictions = self.predict(x_val)
                val_loss = np.mean((val_predictions - y_val) ** 2)

                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.n_iter_no_change:
                    break

            if (np.linalg.norm(coef_grad) < self.tol and
                    np.abs(intercept_grad) < self.tol):
                break

        return self

    def predict(self, x):
        return self.intercept_ + x @ self.coef_

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @coef_.setter
    def coef_(self, value):
        self._coef = value

    @intercept_.setter
    def intercept_(self, value):
        self._intercept = value
