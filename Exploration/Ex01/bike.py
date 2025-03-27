import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
#(1) 데이터 가져오기기
df = pd.read_csv('train.csv')

#(2) 컬럼을 자료형으로 변환
df['datetime'] = pd.to_datetime(df['datetime'])

df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['second'] = df['datetime'].dt.second
#결측치 확인
print(df.isnull().sum())

#(3)데이터 개수 시각화
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
sns.countplot(data=df, x='year', ax=axs[0, 0]).set_xlabel('')
axs[0, 0].set_title('year')
sns.countplot(data=df, x='month', ax=axs[0, 1]).set_xlabel('')
axs[0, 1].set_title('month')
sns.countplot(data=df, x='day', ax=axs[0, 2]).set_xlabel('')
axs[0, 2].set_title('day')
sns.countplot(data=df, x='hour', ax=axs[1, 0]).set_xlabel('')
axs[1, 0].set_title('hour')
sns.countplot(data=df, x='minute', ax=axs[1, 1]).set_xlabel('')
axs[1, 1].set_title('minute')
sns.countplot(data=df, x='second', ax=axs[1, 2]).set_xlabel('')
axs[1, 2].set_title('second')
plt.tight_layout()
plt.show()

#(4)x,y 컬럼 선택
# 연속형 변수의 상관관계 계산(상위5개 파악)
correlation = ['year','month','day','hour','minute','second','season','holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered','count' ]
co_matrix = df[correlation].corr()
print("상관관계:")
print(co_matrix['count'].sort_values(ascending=False))

top = co_matrix['count'].sort_values(ascending=False)[1:6].index.tolist()
print("상위5개")
print(top)

features = ['hour','temp','atemp','humidity']
x = df[features]
y = df['count']

# LinearRegression 모델 학습
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print('mse',mse)
rmse = np.sqrt(mse)
print('rmse',rmse)

# 예측결과 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

ax1.scatter(X_test['temp'], y_test, color='blue', label='Actual', alpha=0.5)
ax1.scatter(X_test['temp'], y_pred, color='red', label='Prediction', alpha=0.5)
ax1.set_xlabel('temp')
ax1.set_ylabel('count')
ax1.set_title('actual-prediction(temp)')
ax1.legend()

ax2.scatter(X_test['humidity'], y_test, color='blue', label='Actual', alpha=0.5)
ax2.scatter(X_test['humidity'], y_pred, color='red', label='Prediction', alpha=0.5)
ax2.set_xlabel('humidity')
ax2.set_ylabel('count')
ax2.set_title('actual-prediction(humidity)')
ax2.legend()

plt.show()