import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn

# (1)데이터 가져오기
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

# (2)모델에 입력할 데이터 x
df_x = diabetes.data
# (3)모델에 예측할 데이터 y
df_y = diabetes.target /346

# (4)train 데이터와 test 데이터로 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)
# (5)모델 준비비
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 무작위로 가중치와 편향 초기화
W = np.random.rand(10)  
b = np.random.rand()

# 예측 모델 정의
def model1(X, W, b):
    predictions = 0
    for i in range(10):
        predictions += X[:, i] * W[i]
    predictions += b
    return predictions

# (6)손실 함수 MSE 정의
def MSE(a, b):
    mse = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    return mse

# (6)손실 함수 정의
def loss(X, W, b, y):
    predictions = model1(X, W, b)
    L = MSE(predictions, y)
    return L

# (7)그라디언트 계산 함수 정의
def gradients(X, y, W, b):
    predictions = model1(X, W, b)
    dW = np.dot(X.T, (predictions - y)) / len(y)
    db = np.mean(predictions - y)
    return dW, db

# (8)학습률 설정
LEARNING_RATE = 0.0001

# 손실 기록 리스트 초기화
losses = []

# (9)학습 반복
for i in range(1, 100):
    dW, db = gradients(X_train, y_train, W, b)
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    L = loss(X_train, W, b, y_train)
    losses.append(L)
    if i % 10 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))

# (10)test데이터 성능확인
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Loss Over Iterations')
plt.show()

prediction = model1(X_test, W, b)
mse = loss(X_test, W, b, y_test)
print(mse)

#(11) 정답 데이터와 예측한 데이터 시각화하기
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], y_test, label='Actual', alpha=0.5)
plt.scatter(X_test[:, 0], prediction, label='Prediction', alpha=0.5)
plt.xlabel('첫번쨰칼럼')
plt.ylabel('Target')
plt.legend()
plt.show()