# from sklearn import ensemble, tree

# models = {
#     "dt_gini": tree.DecisionTreeClassifier(criterion="gini", random_state=42),
#     "dt_entropy": tree.DecisionTreeClassifier(criterion="entropy", random_state=42),
#     "rf": ensemble.RandomForestClassifier(random_state=42),
# }

from sklearn import dummy, ensemble, linear_model, preprocessing, tree
from sklearn.pipeline import make_pipeline

models = {
    # Floor: always predicts the base rate (AUC = 0.5). Every model must beat this.
    "dummy": dummy.DummyClassifier(strategy="prior"),
    # Simplest real model; needs scaling
    "logreg": make_pipeline(
        preprocessing.StandardScaler(), linear_model.LogisticRegression(max_iter=1000)
    ),
    "dt_gini": tree.DecisionTreeClassifier(criterion="gini", random_state=42),
    "rf": ensemble.RandomForestClassifier(random_state=42),
    # GBDT with default settings (no tuning yet)
    "hgb": ensemble.HistGradientBoostingClassifier(random_state=42),
}
