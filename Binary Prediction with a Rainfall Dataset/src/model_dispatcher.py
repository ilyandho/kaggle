import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn import dummy, ensemble, linear_model, preprocessing, svm, tree
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
    "lgbm": lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    ),
    "xgb": xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
    ),
    "cat": CatBoostClassifier(
        iterations=500, learning_rate=0.03, depth=4, random_seed=42, verbose=0
    ),
    "svc": make_pipeline(
        preprocessing.StandardScaler(),
        svm.SVC(C=1.0, probability=True, random_state=42),
    ),
}
