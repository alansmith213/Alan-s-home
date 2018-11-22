import pandas as pd
import sklearn
import keras
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential

from keras.layers import Dense

from sklearn.datasets import make_regression

from sklearn.preprocessing import MinMaxScaler

def z_score(x, axis):
    x = np.array(x).astype(float)
    xr = np.rollaxis(x, axis=axis)
    xr -= np.mean(x, axis=axis)
    xr /= np.std(x, axis=axis)
    # print(x)
    return x

train = pd.read_csv('../5001/train.csv')
test = pd.read_csv('../5001/test.csv')

train['penalty_code'] = LabelEncoder().fit_transform(train['penalty'])
test['penalty_code'] = LabelEncoder().fit_transform(test['penalty'])

train['penalty_codee'] = [i + 100 for i in train['penalty_code']]
test['penalty_codee'] = [i + 100 for i in test['penalty_code']]

train['n_job'] = [16 if i < 0 else i for i in train['n_jobs']]
test['n_job'] = [16 if i < 0 else i for i in test['n_jobs']]


# print(train[['penalty_code', 'penalty']])
# 挑选特征值
# selected_features = ['id', 'penalty_code', 'l1_ratio', 'alpha', 'max_iter','n_jobs','n_samples',
#                      'n_features', 'n_classes', 'n_clusters_per_class', 'n_informative', 'flip_y','scale']
selected_features = ['penalty_codee', 'max_iter', 'random_state', 'n_job','n_samples',
                     'n_features', 'n_classes', 'n_clusters_per_class']

X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['time']


# from sklearn.feature_selection import VarianceThreshold
# v = VarianceThreshold(1)
# X_train = v.fit_transform(X_train)
# X_test = v.fit_transform(X_test)
# y_test = test['time']
a = (X_train.shape[1])
print(a)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# print(X_train.values)
# 生成回归数据集

# X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)

# scalerX = preprocessing.StandardScaler().fit(X_train)
# X = scalerX.transform(X_train)

scalarX = MinMaxScaler()
scalarX.fit(X_train)
X = scalarX.transform(X_train)

# # Zscore method
# X_train = z_score(X_train, axis=0)
# X_test = z_score(X_test, axis=0)

# scalerY = preprocessing.StandardScaler().fit(y_train.values.reshape(400,1))
# y = scalerY.transform(y_train.values.reshape(400,1))

# 定义并拟合模型

model = Sequential()

model.add(Dense(320, input_dim=a, activation='relu'))

model.add(Dense(320, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')

model.fit(X, y_train.values, epochs=1000, verbose=0)



Xnew = scalarX.transform(X_test)

# 作出预测

ynew = model.predict(Xnew)

# ynew2 = scalerX.transform(ynew)


tmpData = []
for i in ynew:
    for j in i:
        tmpData.append(j)
ynew = tmpData




test1 = []
for item in ynew:
    if item < 0:
        item = 0
    test1.append(item)

# print(mean_squared_error(y_test, test1))
#
# 显示输入输出

rfr_submission = pd.DataFrame({'Id': test['id'], 'time': test1})
rfr_submission.to_csv('../5001/competit2.csv', index=False)