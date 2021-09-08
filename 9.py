import pandas as pd
from statistics import mean
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix


df = pd.read_csv('College.csv')
df.Private.replace(('Yes', 'No'), (1, 0), inplace=True)

df['Apps01'] = np.where(df['Apps'] >= df['Apps'].median(), 1, 0)

scaled_features = df.copy()
col_names = ['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad',
        'P.Undergrad', 'Outstate', 'Room.Board', 'Books', 'Personal', 'PhD',
        'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'Grad.Rate']
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[col_names] = features
scaled_features['Private'] = df['Private']
col_names += ['Private']
output_samples = 100
dataset_split_fraction = 0.8
accuracy = [0] * output_samples
msk = np.random.rand(len(df)) < dataset_split_fraction
train_X = scaled_features[msk]
test_X = scaled_features[~msk]
train_y = df['Apps01'][msk]
test_y = df['Apps01'][~msk]

# a) Perform logistic regression in order to predict Apps01 using
# all the features on the test data set.
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
X = train_X[col_names].values
y = train_y.values
model.fit(X, y)

# b) Compute the training error rate. Produce a confusion matrix.
# Compute the test error rate
y_prediction_training = model.predict(X)
training_error_rate = zero_one_loss(y, y_prediction_training)
print('Training error rate: ' + str(training_error_rate))
print('----------------')

y_prediction_test = model.predict(test_X[col_names].values)
test_error_rate = zero_one_loss(test_y.values, y_prediction_test)
print('Test error rate: ' + str(test_error_rate))
print('----------------')

conf_mat = confusion_matrix(test_y.values, y_prediction_test)
print('Confusion matrix:')
print(conf_mat)

# c) parameter interpretation

