import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_df = pd.DataFrame({'X': [0.0, 1.0, 3.0, 5.0, 6.0],
                        'Y': [5.0, 4.0, 3.0, 2.0, 1.0]})
print(data_df.iloc[:,1].values)
x = data_df.iloc[:,0].values.reshape(-1,1)
y = data_df.iloc[:,1].values.reshape(-1,1)
reg = LinearRegression().fit(x, y)

print(reg.score(x, y))

pred = reg.predict(x)
plt.scatter(x, y)
plt.plot(x, pred, color='red')
plt.show()