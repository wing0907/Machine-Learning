import random
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import sklearn as sk
import imblearn
from sklearn.preprocessing import LabelEncoder

# 1. Seed 고정
seed = 123
random.seed(seed)
np.random.seed(seed)

# 2. 데이터 로딩
path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)



x = train_csv.drop(['target'], axis=1)
y = train_csv['target']


# 2. 라벨 인코딩
le = LabelEncoder()
y_enc = le.fit_transform(y)   # now 0~8
print(np.unique(y, return_counts=True))

# 3. 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y_enc, train_size=0.75, random_state=seed,
    stratify=y_enc
)

print(np.unique(y_train, return_counts=True))

from imblearn.over_sampling import SMOTE, RandomOverSampler
import sklearn as sk
print('sklearn version : ', sk.__version__) # sklearn version :  1.6.1
import imblearn
print('imblearn version :', imblearn.__version__) # imblearn version : 0.12.4

ros = RandomOverSampler(random_state=seed,
              # k_neighbors=5, # default
              sampling_strategy='auto', # default. 머리쓰기 싫을땐 디폴트가 아름답다.
            #   sampling_strategy=0.75, # 최대값의 75% 지점
              # sampling_strategy={0:500, 1:500}, # 이렇게 하면 1.0 나옴. 아름다운 데이터일 경우.
              # n_jobs=-1, # 0.13에서는 삭제됨. 기냥 포함됨          
              )

x_train, y_train = ros.fit_resample(x_train, y_train) 


# 5. 모델 구성 및 학습
model = RandomForestClassifier(random_state=seed, n_jobs=-1)
model.fit(x_train, y_train)

# 6. 예측 및 평가
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print("accuracy_score :", acc)
print("f1_score :", f1)


# accuracy_score : 0.8050420168067227
# f1_score : 0.7709787748025572


# =====================================
# ROS 적용
# accuracy_score : 0.811376858435682
# f1_score : 0.7728013990947062