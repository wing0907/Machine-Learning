from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
# print(xgb.__version__)
from sklearn.metrics import accuracy_score, r2_score
import sklearn as sk
# print(sk.__version__) # 1.6.1
import warnings
warnings.filterwarnings('ignore')
import ssl

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_digits()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)  


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y
)



es = xgb.callback.EarlyStopping(
    rounds = 50,
    metric_name = 'mlogloss',
    data_name = 'validation_0',
    # save_best = True,
    
)

# 2. 모델구성
model = XGBClassifier(
    n_estimators = 500,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,                 # 회귀 : rmse, mae, rmsle
    eval_metric = 'mlogloss',     # 다중분류 : mlogloss, merror // 이진분류 : logloss, error
                                # 2.1.1버전 이후로 fit에서 모델로 위치이동.
    callbacks = [es],
    random_state=seed
    
    
    )


model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose=0,
          )


print('acc : ', model.score(x_test, y_test))
print(model.feature_importances_)
# [0.05029093 0.07584832 0.52197117 0.35188958]             얘네들을 순서대로 제거했을 때, 성능 나오겠죠?
                                                            # 두개를 제거했을 때, 세개를 제거했을 때, 성능 나오겠죠?
                                                            # 한번에 보게 할거야. 앞에서는 25%를 퉁 쳤죠? 그러면 중간에
                                                            # 향상 되는 걸 못 보잖아~~ 그죠옹?? 아 그렇잖아~~
                                                            # 그래서 그 구간을 찾아내는게 우리의 목적이다잉?!?!
thresholds = np.sort(model.feature_importances_)  # 오름차순              # 얘를 정렬할거야.
print(thresholds)
# [0.05029093 0.07584832 0.35188958 0.52197117]         # 순서가 좀 바꼈죠잉? 그죠잉?

from sklearn.feature_selection import SelectFromModel

for i in thresholds:            # 첫번째 i 에는 뭐가 들어갈까? 0.05029093 이게 들어가겠쮜이~~~??
    selection = SelectFromModel(model, threshold=i, prefit=False) # SelectFromModel 이라는 클래스를 인스턴스화 하고 모델은 xgboost를 사용했어.
    # threshold 가 i 값 이상인 것을 모두 훈련시킨다.
    # prefit = False : 모델이 아직 학습되지 않았을 때, fit 호출해서 훈련한다. (기본값)
    # prefit = True : 이미 학습된 모델을 전달할 때
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape)
    # (120, 4)
    # (120, 3)
    # (120, 2)
    # (120, 1)
    
    select_model = XGBClassifier(
        n_estimators = 500,
        max_depth = 6,
        gamma = 0,
        min_child_weight = 0,
        subsample = 0.4,
        reg_alpha = 0,
        reg_lambda = 1,                 # 회귀 : rmse, mae, rmsle
        eval_metric = 'mlogloss',     # 다중분류 : mlogloss, merror // 이진분류 : logloss, error
                                        # 2.1.1버전 이후로 fit에서 모델로 위치이동.
        callbacks = [es],
        random_state=seed,
    )   
    
    select_model.fit(select_x_train, y_train,
          eval_set = [(select_x_test, y_test)],
          verbose = 0,)
    
    
    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, ACC: %.4f%%' %(i, select_x_train.shape[1], score*100))

# Trech=0.000, n=64, ACC: 98.0556%
# Trech=0.000, n=64, ACC: 83.6111%
# Trech=0.000, n=64, ACC: 83.6111%
# Trech=0.000, n=61, ACC: 83.6111%
# Trech=0.000, n=60, ACC: 83.6111%
# Trech=0.000, n=59, ACC: 83.6111%
# Trech=0.000, n=58, ACC: 83.6111%
# Trech=0.000, n=57, ACC: 83.6111%
# Trech=0.000, n=56, ACC: 83.6111%
# Trech=0.000, n=55, ACC: 83.6111%
# Trech=0.000, n=54, ACC: 83.6111%
# Trech=0.000, n=53, ACC: 83.6111%
# Trech=0.001, n=52, ACC: 83.6111%
# Trech=0.002, n=51, ACC: 83.6111%
# Trech=0.002, n=50, ACC: 84.1667%
# Trech=0.003, n=49, ACC: 84.1667%
# Trech=0.004, n=48, ACC: 84.1667%
# Trech=0.004, n=47, ACC: 84.1667%
# Trech=0.005, n=46, ACC: 84.1667%
# Trech=0.005, n=45, ACC: 84.1667%
# Trech=0.005, n=44, ACC: 83.6111%
# Trech=0.005, n=43, ACC: 83.6111%
# Trech=0.006, n=42, ACC: 83.6111%
# Trech=0.006, n=41, ACC: 83.6111%
# Trech=0.007, n=40, ACC: 83.6111%
# Trech=0.008, n=39, ACC: 83.8889%
# Trech=0.008, n=38, ACC: 84.4444%
# Trech=0.009, n=37, ACC: 85.0000%
# Trech=0.009, n=36, ACC: 84.4444%
# Trech=0.009, n=35, ACC: 86.1111%
# Trech=0.009, n=34, ACC: 85.8333%
# Trech=0.010, n=33, ACC: 86.3889%
# Trech=0.010, n=32, ACC: 86.3889%
# Trech=0.011, n=31, ACC: 86.3889%
# Trech=0.012, n=30, ACC: 86.3889%
# Trech=0.013, n=29, ACC: 86.6667%
# Trech=0.014, n=28, ACC: 86.1111%
# Trech=0.015, n=27, ACC: 86.1111%
# Trech=0.015, n=26, ACC: 84.4444%
# Trech=0.015, n=25, ACC: 83.6111%
# Trech=0.016, n=24, ACC: 83.8889%
# Trech=0.016, n=23, ACC: 83.6111%
# Trech=0.016, n=22, ACC: 83.3333%
# Trech=0.017, n=21, ACC: 85.0000%
# Trech=0.018, n=20, ACC: 86.1111%
# Trech=0.020, n=19, ACC: 86.3889%
# Trech=0.021, n=18, ACC: 86.6667%
# Trech=0.022, n=17, ACC: 85.2778%
# Trech=0.024, n=16, ACC: 84.1667%
# Trech=0.024, n=15, ACC: 84.7222%
# Trech=0.025, n=14, ACC: 84.4444%
# Trech=0.025, n=13, ACC: 84.4444%
# Trech=0.027, n=12, ACC: 83.6111%
# Trech=0.030, n=11, ACC: 83.3333%
# Trech=0.032, n=10, ACC: 83.6111%
# Trech=0.035, n=9, ACC: 81.9444%
# Trech=0.038, n=8, ACC: 79.4444%
# Trech=0.039, n=7, ACC: 72.7778%
# Trech=0.047, n=6, ACC: 72.7778%
# Trech=0.050, n=5, ACC: 65.2778%
# Trech=0.057, n=4, ACC: 59.1667%
# Trech=0.058, n=3, ACC: 47.5000%
# Trech=0.058, n=2, ACC: 28.0556%
# Trech=0.061, n=1, ACC: 21.1111%





exit()
print("25%지점 : ", np.percentile(model.feature_importances_, 25)) # 25% 지점을 출력하게쒈~!
# 0.02461671084165573

percentile =  np.percentile(model.feature_importances_, 25)
print(type(percentile))     # <class 'numpy.float64'>

col_name = []
# 삭제할 컬럼(25% 이하인 놈)을 찾아내자!!!
for i, fi in enumerate(model.feature_importances_): # index 와 값(fi)
    # print(i, fi)
    if fi <= percentile:        # 값이 낮은 놈을 찾아
        col_name.append(datasets.feature_names[i])  # col_name에 집어넣어
    else:
        continue
print(col_name)         # ['sepal length (cm)']

x = pd.DataFrame(x, columns=datasets.feature_names) # 얘는 위에서 만들었어도 된다
x = x.drop(columns=col_name)                        # 고놈을 삭제해.

print(x)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y
)

model.fit(x_train, y_train)
print('acc :', model.score(x_test, y_test))         # acc : 0.9333333333333333


# tqdm 간지나는 progress bar

# import matplotlib.pyplot as plt

# def plot_feature_importance_datasets(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     # 수평 막대 그래프, 4개의 열의 feature importance 그래프, 값 위치 센터
#     plt.yticks(np.arange(n_features), model.feature_importances_)
#     # 눈금, 숫자 레이블 표시
#     plt.xlabel("feature Importance")
#     plt.ylabel("Feature")
#     plt.ylim(-1, n_features)        # 축 범위 설정
#     plt.title(model.__class__.__name__)

# plot_feature_importance_datasets(model)
# plt.show()



