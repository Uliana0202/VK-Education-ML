import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBCustomRegressor:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self._estimators = []
        self.initial_prediction = None

    def fit(self, x, y):
        self.initial_prediction = np.mean(y)
        current_prediction = np.full_like(
            y, self.initial_prediction, dtype=np.float64
        )

        for _ in range(self.n_estimators):
            residuals = y - current_prediction

            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            tree.fit(x, residuals)

            current_prediction += self.learning_rate * tree.predict(x)

            self._estimators.append(tree)

    def predict(self, x):
        prediction = np.full(
            x.shape[0], self.initial_prediction, dtype=np.float64
        )
        for tree in self._estimators:
            prediction += self.learning_rate * tree.predict(x)
        return prediction

    @property
    def estimators_(self):
        return self._estimators


class GBCustomClassifier:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self._estimators = []
        self.initial_prediction = None

    def fit(self, x, y):
        y = np.where(y == 0, -1, 1)

        pos = np.sum(y == 1)
        neg = np.sum(y == -1)
        self.initial_prediction = np.log(pos / neg) / 2
        current_prediction = np.full_like(
            y, self.initial_prediction, dtype=np.float64
        )

        for _ in range(self.n_estimators):
            p = 1 / (1 + np.exp(-current_prediction))
            residuals = (y + 1) / 2 - p

            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            tree.fit(x, residuals)

            current_prediction += self.learning_rate * tree.predict(x)

            self._estimators.append(tree)

    def predict_proba(self, x):
        log_odds = np.full(
            x.shape[0], self.initial_prediction, dtype=np.float64
        )
        for tree in self._estimators:
            log_odds += self.learning_rate * tree.predict(x)

        proba = 1 / (1 + np.exp(-log_odds))
        return np.vstack([1 - proba, proba]).T

    def predict(self, x):
        proba = self.predict_proba(x)
        return (proba[:, 1] > 0.5).astype(int)

    @property
    def estimators_(self):
        return self._estimators
