import pandas as pd
from sklearn import linear_model

df = pd.read_csv('College.csv')
df.Private.replace(('Yes', 'No'), (1, 0), inplace=True)

X = df[['Private', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad',
        'P.Undergrad', 'Outstate', 'Room.Board', 'Books', 'Personal', 'PhD',
        'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'Grad.Rate']]
y = df['Apps']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)