from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np
from sklearn.model_selection import train_test_split

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y
)


# 2. 모델구성
model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)

models = [model1, model2, model3, model4]
for model in models:
    model.fit(x_train, y_train)
    print("================", model.__class__.__name__, "================")
    print('acc : ', model.score(x_test, y_test))
    print(model.feature_importances_)

import matplotlib.pyplot as plt

def plot_feature_importance_datasets(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    # 수평 막대 그래프, 4개의 열의 feature importance 그래프, 값 위치 센터
    plt.yticks(np.arange(n_features), model.feature_importances_)
    # 눈금, 숫자 레이블 표시
    plt.xlabel("feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)        # 축 범위 설정
    plt.title(model.__class__.__name__)

plot_feature_importance_datasets(model)
plt.show()