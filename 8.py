import pandas as pd
import numpy as np
from scipy.stats import entropy
import trivialClassifier as tc
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('College.csv')


df.Private.replace(('Yes', 'No'), (1, 0), inplace=True)

# a) Create a binary attribute 'Apps01'
df['Apps01'] = np.where(df['Apps'] >= df['Apps'].median(), 1, 0)

# a) Compute the entropy of Apps01
d = df['Apps01'].value_counts()
entropy = entropy(d.array)

# b) Scale the data


# c) Split the dataframe into test and train 20/80
accuracy = [0]*100
for i in range(100):

        msk = np.random.rand(len(df)) < 0.05
        train = df[msk]
        test = df[~msk]
        print(len(test), len(train))

# d) Make a trivial classifier.
        TC = tc.trivialClassifier()
        TC.fit(train['Apps'])
        TC.predict(test['Apps'])
        accuracy[i] = TC.score(test['Apps'], test['Apps01'])

print('Mean accuracy over 100 tests: '
      + str(mean(accuracy)))

plt.plot(accuracy)
plt.show()

