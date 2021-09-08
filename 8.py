import pandas as pd
import trivialClassifier as tc
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

df = pd.read_csv('College.csv')
df.Private.replace(('Yes', 'No'), (1, 0), inplace=True)

# a) Create a binary attribute 'Apps01'

df['Apps01'] = np.where(df['Apps'] >= df['Apps'].median(), 1, 0)

# a) Compute the entropy of Apps01
d = df['Apps01'].value_counts()
entropy = entropy(d.array)

# b) Scale the data
scaled_features = df.copy()
col_names = ['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad',
        'P.Undergrad', 'Outstate', 'Room.Board', 'Books', 'Personal', 'PhD',
        'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'Grad.Rate']
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[col_names] = features

#print(scaled)

# c) Split the dataframe into test and train 20/80
output_samples = 100
dataset_split_fraction = 0.05
accuracy = [0] * output_samples
for i in range(output_samples):
    msk = np.random.rand(len(df)) < dataset_split_fraction
    train = scaled_features[msk]
    test = scaled_features[~msk]
    # print(len(test), len(train))

    # d) Train a trivial classifier
    TC = tc.trivialClassifier()
    TC.fit(train['Apps'])
    TC.predict(test['Apps'])
    accuracy[i] = TC.score(test['Apps'], test['Apps01'])

print('Mean accuracy over 100 tests: ' + str(mean(accuracy)))

plt.plot(accuracy)
plt.show()
