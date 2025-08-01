import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC       # svm support vector machine
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

# 2. 모델
# model = Perceptron()        # 0.5
# model = LinearSVC()           # 0.5
# model = LogisticRegression()  # 0.75 나옴 나머지는 1.0나옴
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])

model.fit(x_data, y_data, epochs=100)


# 4. 평가, 예측
y_predict = model.predict(x_data)

# results = model.score(x_data, y_data)
results = model.evaluate(x_data, y_data)
print("model.score :" , results)

acc = accuracy_score(y_data, np.round(y_predict))
print("acc :" , acc)



# acc = accuracy_score(y_data, y_predict)
# print("acc :" , acc)

# model.score : 0.5
# acc : 0.5


# from sklearn.neural_network import MLPClassifier

# model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
# model.fit(x_data, y_data)

# results = model.score(x_data, y_data)
# print("model.score :" , results)

# y_predict = model.predict(x_data)
# print("acc :", accuracy_score(y_data, y_predict))

# # model.score : 1.0
# # acc : 1.0