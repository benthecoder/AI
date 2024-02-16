import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

plt.style.use("bmh")


"""
f(x) = xW + b

MSELoss = (actual - predicted)^2 / n_samples

wrt weights
((y - f(x))^2)' = 2(y - f(x))( y - f(x))'
                = 2(y - f(x))(y - xW - b)
                = -2x(y - f(x))

wrt bias
((y - f(x))^2)' = 2(y - f(x))( y - f(x))'
                = 2(y - f(x))(y - xW - b)
                = -2(y - f(x))
"""


@dataclass
class LinearRegression:
    features: np.ndarray
    labels: np.ndarray
    learning_rate: float
    epochs: int
    logging: bool

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fits LR model"""

        n_samples, n_features = features.shape
        self.weights, self.bias = np.zeros(n_features), 0

        for epoch in range(self.epochs):
            residuals = labels - self.predict(features)

            d_weights = -2 / n_samples * features.T.dot(residuals)

            d_bias = -2 / n_samples * residuals.sum()

            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias

            mse_loss = np.mean(np.square(residuals))
            if self.logging:
                print(f"MSE loss [{epoch}] : {mse_loss:.15f}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Perform inference using given features"""

        return features.dot(self.weights) + self.bias


if __name__ == "__main__":
    # training data
    X_train = np.arange(0, 250).reshape(-1, 1)
    y_train = np.arange(0, 500, 2)

    # testing data
    X_test = np.arange(300, 400, 8).reshape(-1, 1)
    y_test = np.arange(600, 800, 16)

    # Train model
    LR = LinearRegression(X_train, y_train, learning_rate=1e-5, epochs=75, logging=True)

    LR.fit(X_train, y_train)

    preds = LR.predict(X_test).round()

    # Plot the data
    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("f(x) = 2x")
    fig.tight_layout()
    fig.set_size_inches(18, 8)

    axs[0].set_title("Visualization for f(x) = 2x")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].plot(X_train, y_train)

    axs[1].set_title("Scatterplot for f(x) = 2x Data")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].scatter(X_test, y_test, color="blue")

    axs[2].set_title("Visualization for Approximated f(x) = 2x")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].scatter(X_test, y_test, color="blue")
    axs[2].plot(X_test, preds)

    plt.show()

    accuracy = accuracy_score(preds, y_test)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test, preds, average="macro"
    )

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {recall:.3f}")
    print(f"Recall:    {precision:.3f}")
    print(f"F-score:   {fscore:.3f}")
