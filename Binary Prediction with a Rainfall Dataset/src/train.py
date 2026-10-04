import argparse
import os

import joblib
import pandas as pd
from sklearn import metrics

import config
import model_dispatcher


def run(fold, model):
    df = pd.read_csv(config.TRAINING_FILE)
    # df_train = df[df.kfold != fold].reset_index(drop=True)
    # df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Time-based split: use all data before the fold as training, and the fold as validation
    df_train = df[df.kfold < fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(columns=["kfold", "rainfall"])
    y_train = df_train.rainfall

    x_valid = df_valid.drop(columns=["kfold", "rainfall"])
    y_valid = df_valid.rainfall

    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, preds)
    f1_score = metrics.f1_score(y_valid, preds)
    predict_proba = clf.predict_proba(x_valid)[:, 1]
    roc_auc = metrics.roc_auc_score(y_valid, predict_proba)

    row = pd.DataFrame(
        [
            {
                "model": model,
                "fold": fold,
                "auc": roc_auc,
                "accuracy": accuracy,
                "f1": f1_score,
            }
        ]
    )
    path = "../models/results.csv"
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    print(
        f"Fold: {fold}, Accuracy: {accuracy}, F1 Score: {f1_score}, ROC AUC: {roc_auc}"
    )

    joblib.dump(clf, f"{config.MODEL_OUTPUT}{model}_{fold}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number to run")
    parser.add_argument("--model", type=str, required=True, help="Model to run")
    args = parser.parse_args()
    run(args.fold, args.model)
