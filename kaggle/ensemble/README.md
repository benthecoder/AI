# Ensemble Learning

## What

A technique that blends predictions from a diverse set of models.

See [Wisdom of Crowds](https://arxiv.org/abs/1605.04074)

## Why they work

- performance: ensemble reduce variance component of prediction error by adding bias
- robustness: ensemble reduces reliance on any single model's prediction, making it better at handling noisy data.

## Diversity

ensemble learning is based on concept of combining multiple weak learners, weak because individual models don't need to be very accurate, as long as they're better than a random model, combining them is beneficial.

Diversity is a concept referring to the idea that individual models have to be **as different from each other as possible**. This is because different models are likely to make different types of errors. By combining predictions of a diverse set, we can reduce overall error of the ensemble.

In order for accuracy of ensemble to be better than individual models, there needs to be diversity.

### how

- train each model on different subset of data
  - bagging (w replacement)
  - pasting (w/out replacement)
  - ex:
    - random forest: achieves diversity using random number of features at each split
    - extremely randomized trees: a random split to lower correlation between trees
- train each model with a different set of features
- train each model using a different type of algorithm
  - voting and stacking meta-models

### good and bad

good diversity: ensemble is already correct, low disagreement between classifier, several votes wasted

bad diversity: ensemble is incorrect, any disagreement represent a wasted vote, as individual classifier did not contribute to correct decision.

increase good diversity (where disagreements among classifiers contribute to correct decisions) and reduce bad diversity (where disagreements does not contribute to correct decisions).

### metrics

- let f_1, f_2, f_3 be predictions of diff models in ensemble

two types of measures

- pairwise: computed for every f_i, f_j pair, represented by NxN matrix
- global: computed on whole matrix of predictions, represented by a single value

Measures

- pearson correlation coefficient
- disagreement
- Yule's Q
- entropy

References

- [Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy | Machine Learning](https://link.springer.com/article/10.1023/A:1022859003006)
- [Understanding the Importance of Diversity in Ensemble Learning](https://towardsdatascience.com/understanding-the-importance-of-diversity-in-ensemble-learning-34fb58fd2ed0#:~:text=Ensemble%20learning%20is%20a%20powerful,of%20the%20ensemble%20also%20increased.)

## Methods

1. blending : averaging, weighted averaging, and rank averaging
   - average the outputs
   - weights given to model can be assigned explicitly or implicitly by [rank averaging](https://towardsdatascience.com/ensemble-averaging-improve-machine-learning-performance-by-voting-246106c753ee), which ranks models by performance and gives more accurate models greater weights
   - involves using optuna or hyperopt to find optimal blend by taking into account cross validation metrics
2. Voting : for classification
   - ex: majority voting: class that most models predict is chosen
3. classical trio: bagging, boosting and stacking
   - bagging: train multiple models on different subsets of training data and average prediction
   - boosting: sequentially training models, each new model focuses on errors made by predecessors.
   - Stacking: feed predictions of various models as input to higher-level model.

## Reality

Building robust and highly accurate models are only half the solution. an equally challenging part is explainability and fairness.

see: [On Transparency of Machine Learning Models: A Position Paper](https://crcs.seas.harvard.edu/sites/projects.iq.harvard.edu/files/crcs/files/ai4sg_2020_paper_62.pdf) and [[2105.06791] Agree to Disagree: When Deep Learning Models With Identical Architectures Produce Distinct Explanations](https://arxiv.org/abs/2105.06791)

## References

- [Unreasonably Effective Ensemble Learning](https://www.kaggle.com/code/yeemeitsang/unreasonably-effective-ensemble-learning/notebook#Conclusion)

## implementations

- [EnsemblesTutorial/ensemble_functions.py at main · PadraigC/EnsemblesTutorial](https://github.com/PadraigC/EnsemblesTutorial/blob/main/ensemble_functions.py)
