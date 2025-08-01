import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing, load_digits
from sklearn.model_selection import KFold, cross_val_score # fit 과 score가 섞여있음
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor


# 1. 데이터
digits = load_digits()
x, y = digits.data, digits.target

n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=190) # 분류형태일 때 성능이 더 좋을 때도 있다.
# shuffle=False로 하면 고정이 되지만 데이터가 앞에 50개가 0, 그 다음 1 이런식이라면 라벨 인코딩 이슈가 생길수 있음
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=190) # 분류형태에 사용


# 2. 모델
model = MLPClassifier()
# model = RandomForestClassifier()

# 3. 훈련
print("gannasherry")
scores = cross_val_score(model, x, y, cv=kfold) # fit까지 포함 된거쥬~~?
print('acc : ', scores, '\n평균 acc : ', round(np.mean(scores), 4))

# MLPClassifier
# acc :  [0.975      0.97222222 0.96935933 0.98885794 0.97493036] 
# 평균 acc :  0.9761

# RandomForestClassifier
# acc :  [0.97222222 0.96666667 0.96935933 0.97214485 0.97493036] 
# 평균 acc :  0.9711

