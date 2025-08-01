from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
# import xgboost as xgb
# print(xgb.__version__)


seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
path = 'C:\Study25\_data\kaggle\santander\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
print(train_csv)        # [200000 rows x 201 columns]
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape)
print(x.shape, y.shape)  

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y
)


# 2. 모델구성
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
print("================", model.__class__.__name__, "================")
print('acc : ', model.score(x_test, y_test))
print(model.feature_importances_)
# acc :  0.83707103301617

# [0.47826383 0.07366086 0.0509511  0.02446287 0.02366972 0.14824368
#  0.0921493  0.10859864]
print("25%지점 : ", np.percentile(model.feature_importances_, 25)) # 25% 지점을 출력하게쒈~!
# 25%지점 :  0.044329043477773666


percentile =  np.percentile(model.feature_importances_, 25)
# print(type(percentile))     # <class 'numpy.float64'>

col_name = []
# 삭제할 컬럼(25% 이하인 놈)을 찾아내자!!!
for i, fi in enumerate(model.feature_importances_): # index 와 값(fi)
    # print(i, fi)
    if fi <= percentile:        # 값이 낮은 놈을 찾아
        col_name.append(x_train.columns[i])  # col_name에 집어넣어
    else:
        continue
print(col_name)         # ['var_3', 'var_7', 'var_10', 'var_14', 'var_16', 'var_17', 'var_19', 'var_25', 'var_27', 'var_29', 'var_30', 'var_38', 'var_39', 'var_42', 'var_46', 'var_47', 'var_60', 'var_61', 'var_62', 'var_64', 'var_65', 'var_69', 'var_72', 'var_73', 'var_79', 'var_84', 'var_96', 'var_97', 'var_98', 'var_100', 'var_101', 'var_103', 'var_113', 'var_117', 'var_120', 'var_124', 'var_126', 'var_129', 'var_136', 'var_142', 'var_143', 'var_153', 'var_158', 'var_159', 'var_160', 'var_161', 'var_176', 'var_181', 'var_185', 'var_189']   얘네를 PCA로 압축해서 붙여버리겠드아!

# exit()

x_f = pd.DataFrame(x, columns=x_train.columns) # 얘는 위에서 만들었어도 된다
x1 = x_f.drop(columns=col_name)                        # 고놈을 삭제해.
x2 = x_f[['var_3', 'var_7', 'var_10', 'var_14', 'var_16', 'var_17', 'var_19', 'var_25', 'var_27', 'var_29', 'var_30', 'var_38', 'var_39', 'var_42', 'var_46', 'var_47', 'var_60', 'var_61', 'var_62', 'var_64', 'var_65', 'var_69', 'var_72', 'var_73', 'var_79', 'var_84', 'var_96', 'var_97', 'var_98', 'var_100', 'var_101', 'var_103', 'var_113', 'var_117', 'var_120', 'var_124', 'var_126', 'var_129', 'var_136', 'var_142', 'var_143', 'var_153', 'var_158', 'var_159', 'var_160', 'var_161', 'var_176', 'var_181', 'var_185', 'var_189']]
print(x2)

x1_train, x1_test, x2_train, x2_test,  = train_test_split(
    x1, x2, train_size=0.8, random_state=seed,
    stratify=y
)
print(x1_train.shape, x1_test.shape)    # (16512, 6) (4128, 6)
print(x2_train.shape, x2_test.shape)    # (16512, 2) (4128, 2)
print(y_train.shape, y_test.shape)      # (16512,) (4128,)

pca = PCA(n_components=1)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)
print(x2_train.shape, x2_test.shape)    # (16512, 1) (4128, 1)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)
print(x_train.shape, x_test.shape)      # (16512, 7) (4128, 7)


model.fit(x_train, y_train)                                   # acc :  0.91405
print('FI_Drop + PCA :', model.score(x_test, y_test))         # FI_Drop : 0.89885
                                                              # FI_Drop + PCA : 0.91365
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



