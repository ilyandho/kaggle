import itertools

import pandas as pd
from sklearn import metrics

import config

MODELS = ["logreg", "rf", "hgb", "lgbm", "xgb", "cat", "svc"]
FOLDS = [1, 2, 3, 4, 5]


def load(model):
    return pd.concat(
        pd.read_csv(f"{config.MODEL_OUTPUT}oof_{model}_{f}.csv") for f in FOLDS
    )


oof = {m: load(m).set_index("row") for m in MODELS}
y = oof[MODELS[0]].y


def score(preds):
    # Mean of per-fold AUC, to match how results.csv scores single models
    df = pd.DataFrame({"y": y, "p": preds, "fold": oof[MODELS[0]].fold})
    return df.groupby("fold").apply(lambda g: metrics.roc_auc_score(g.y, g.p)).mean()


for m in MODELS:
    print(f"{m:20s} {score(oof[m].pred):.5f}")

# Every pair and triple, simple average of ranks
for k in (2, 3):
    for combo in itertools.combinations(MODELS, k):
        avg = sum(oof[m].pred.rank() for m in combo) / k
        print(f"{'+'.join(combo):20s} {score(avg):.5f}")

print("weighted average of ranks")
for w in [0.5, 0.6, 0.7, 0.8, 0.9]:
    blend = w * oof["logreg"].pred.rank() + (1 - w) * oof["xgb"].pred.rank()
    print(f"logreg {w:.1f} / xgb {1 - w:.1f}  {score(blend):.5f}")


print("weighted average of ranks, logreg + xgb + lgbm")
blend = 0.8 * oof["logreg"].pred.rank() + 0.2 * oof["xgb"].pred.rank()
df = pd.DataFrame(
    {"y": y, "fold": oof["logreg"].fold, "logreg": oof["logreg"].pred, "blend": blend}
)
per_fold = df.groupby("fold").apply(
    lambda g: pd.Series(
        {
            "logreg": metrics.roc_auc_score(g.y, g.logreg),
            "blend": metrics.roc_auc_score(g.y, g.blend),
        }
    ),
    # include_groups=False,
)
per_fold["diff"] = per_fold.blend - per_fold.logreg
print(per_fold)
