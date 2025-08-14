# XGBoost
# california
# diabetes

# LGBM
# cancer
# digits

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

RANDOM_STATE = 333
TEST_SIZE = 0.2
CV_FOLDS = 5

def load_dataset(name: str):
    name = name.lower()
    if name == "cancer":
        return load_breast_cancer(return_X_y=True)
    elif name == "digits":
        return load_digits(return_X_y=True)
    else:
        raise ValueError("Choose one of: 'cancer', 'digits'")

DATASETS = ["cancer", "digits"]

pipe = Pipeline([
    ("pca", "passthrough"),     # 그리드에서 on/off
    ("lgbm", LGBMClassifier(n_jobs=-1, random_state=RANDOM_STATE))
])

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

summary = []

for name in DATASETS:
    print(f"\n========== DATASET: {name} ==========")
    X, y = load_dataset(name)
    n_classes = np.unique(y).size

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 공통 그리드(파라미터 크면 느려지니 적당히)
    base_grid = {
        "pca": ["passthrough", PCA(n_components=0.95, random_state=RANDOM_STATE)],
        "lgbm__n_estimators": [300, 600],
        "lgbm__learning_rate": [0.05, 0.1],
        "lgbm__num_leaves": [31, 63],
        "lgbm__max_depth": [-1, 6, 10],
        "lgbm__subsample": [0.8, 1.0],
        "lgbm__colsample_bytree": [0.8, 1.0],
        "lgbm__min_child_samples": [20, 40],
    }

    # 데이터셋별 objective 설정
    if n_classes == 2:
        base_grid["lgbm__objective"] = ["binary"]
    else:
        base_grid["lgbm__objective"] = ["multiclass"]
        base_grid["lgbm__num_class"] = [n_classes]

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=base_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid.fit(x_train, y_train)

    print(f"[{name}] Best Params : {grid.best_params_}")
    print(f"[{name}] Best CV Acc: {grid.best_score_:.5f}")

    y_pred = grid.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    print(f"[{name}] Test Acc   : {acc:.5f}")
    print(f"[{name}] Test F1(m) : {f1m:.5f}")

    summary.append({
        "dataset": name,
        "best_params": grid.best_params_,
        "cv_accuracy": round(grid.best_score_, 5),
        "test_accuracy": round(acc, 5),
        "test_f1_macro": round(f1m, 5),
    })

print("\n========= SUMMARY =========")
print(pd.DataFrame(summary).to_string(index=False))

'''
[digits] Test Acc   : 0.96111
[digits] Test F1(m) : 0.96104


========= SUMMARY =========
dataset
                                   best_params  cv_accuracy  test_accuracy  test_f1_macro
 cancer                             {'lgbm__colsample_bytree': 1.0, 'lgbm__learning_rate': 0.1, 'lgbm__max_depth': 6, 'lgbm__min_child_samples': 40, 'lgbm__n_estimators': 600, 'lgbm__num_leaves': 31, 'lgbm__objective': 'binary', 'lgbm__subsample': 0.8, 'pca': 'passthrough'}      0.96923        0.98246        0.98115
 digits {'lgbm__colsample_bytree': 0.8, 'lgbm__learning_rate': 0.1, 'lgbm__max_depth': 10, 'lgbm__min_child_samples': 40, 'lgbm__n_estimators': 300, 'lgbm__num_class': 10, 'lgbm__num_leaves': 31, 'lgbm__objective': 'multiclass', 'lgbm__subsample': 0.8, 'pca': 'passthrough'}      0.97635        0.96111        0.96104
'''