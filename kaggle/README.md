# Kaggle Notes

Sources

- [Kaggle Solutions](https://farid.one/kaggle-solutions/)
- [The Kaggle Book](https://learning.oreilly.com/library/view/the-kaggle-book/9781801817479/)
- [Winning Toolkit for Competitive ML](https://mlcontests.com/winning-toolkit/)
- [[1811.12808] Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://arxiv.org/abs/1811.12808)

## Metrics

- in real world, your model will be evaluated against multiple metrics, and some of the metrics won't even be related to how your predictions perform against the ground truth you are using for testing
- ex: domain of knowledge you're working in, scope of project, number of features considered for model, overall memory usage, requirements for special hardware, latency of prediction process, complexity of prediction model, and many other aspects may count more that just predictive performance.
- it is dominated by business and tech infrastructure concerns

### evaluation metrics and objective functions

- objective functions: serves model during training, involved in process of error minimization (or score maximization)
- evaluation metric: serves your model after it has been trained by providing a score
  - it cannot influence how model fits data, but it helps you select the best configurations within a model, and the best model among competing ones
  - analysis of evaluation metric should be your first act in competitions

### objective function, cost function and loss function

- loss function: single data point (penalty = |pred - ground truth|)
- cost function: on whole dataset (or a batch), sum or average over loss penalties. Can comprise further constraints, i.e. L1 or L2 penalties
- objective function: related to scope of optimization during ML training, comprise cost function, but not limited to them. Can also take into account goals not related to target, ex: requiring sparse coefficients of estimated model or minimization of coefficients' values, i.e. L1 and L2 regularization.

Loss & cost imply optimization based on minimization, objective function is neutral, it can be a maximization or minimization activity.

Scoring function (higher score = better prediction, maximization process), error functions (smaller error = better prediction, minimization process)

### basic tasks

- regression
  - a model that can predict a real number (often positive, sometimes negative)
  - evaluations: diff = dist(pred, true), square(diff) it to punish large errors / log(diff) to penalize predictions of the wrong scale
- classification
  - binary: 0 or 1 / probabilities of class (in medical fields)
    - churn/not churn, cancer/not cancer (probability is important here)
    - !watch out for imbalance!, use eval metrics that take imbalance into account
  - multi-class: >2 classes
    - ex: leaf classification
    - ensure performance across class is comparable (model can underperform with respect to certain classes)
  - multi-label: predictions are not exclusive and you can predict multiple class ownership
    - ex: classify news articles with relevant topics
    - require further evaluations to control whether model is predicting correct classes, as well as the correct number and mix of classes
- Ordinal
  - halfway between regression and classification
  - ex: magnitude of earthquake, customer preferences
  - as multiclass
    - get prediction as integer value, but not take into account the order of class
    - problem: probabilities distributed across entire range of possible values, depicting multi-model and often asymmetric distribution (you should expect Gaussian around max probability class)
  - as regression
    - output as a float number, and results include full range of values between integers of ordinal distribution, and possible outside of it
    - one solution is to crop the output values, cast into int by unit rounding, but may lead to inaccuracies, requiring more sophisticated post-processing

### common metrics

Top Kaggle metrics

- **AUC**: measures if your model's predicted probabilities tend to predict positive cases with high probabilities
- **log loss**: how far your predicted probabilities are from the ground truth (as you optimize for log loss, you optimize for AUC metric)
- **MAC@{k}**: common in recsys and search engines, used for information retrieval evaluations
  - ex: whale identification and having 5 possible guesses
  - ex2: quickdraw doodle recognition (guess the content of a drawn sketch in 3 attempts, score not just if you can guess correctly, but if your correct guess is among a certain number, the "K" in the name of the function, of other incorrect predictions)
- RMSLE (root mean squared logarithmic error):
- quadratic weigthed kappa: for ordinal scale problems (problems that involve guessing a progressive integer number)

Metrics for regression

- MSE
  - mean of sum of squared errors (SSE)
  - cautions
    - sensitive to outliers
    - imbalanced errors
    - not robust to non-gaussian errors
    - lack of sensitivity to small errors
- R squared (coefficient of determination)
