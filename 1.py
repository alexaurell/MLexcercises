import pandas as pd
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt


def predictor(cutoff, score):
    output = 1
    if score <= cutoff:
        output = -1
    return output


def precision(conf_mat):
    return conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[1, 0])


def accuracy(conf_mat):
    return (conf_mat[0, 0] + conf_mat[1, 1]) / np.sum(np.sum(conf_mat))


def recall(conf_mat):
    return conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])


def fallout(conf_mat):
    return conf_mat[1, 0] / (conf_mat[1, 0] + conf_mat[1, 1])


data_df = pd.DataFrame({'true_class': [1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0],
                        'score': [7.0, 4.0, 2.0, 1.0, -1.0, -4.0, -5.0, -6.0]})
result_df = pd.DataFrame({'cutoff': [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -3.0, -4.0, -5.0, -6.0, -7.0]})

p = [0] * len(result_df)
p2 = [0] * len(result_df)
a = [0] * len(result_df)
r = [0] * len(result_df)
f = [0] * len(result_df)

for i in range(len(result_df)):

    classification = [0] * len(data_df)
    cutoff = result_df['cutoff'].loc[i]
    for j in range(len(data_df)):
        classification[j] = predictor(cutoff, data_df['score'].loc[j])

    conf_mat = skm.confusion_matrix(data_df['true_class'], classification)
    print(conf_mat)
    #### CONTINUE FROM HERE!

    p[i] = precision(conf_mat)
    a[i] = accuracy(conf_mat)
    r[i] = recall(conf_mat)
    f[i] = fallout(conf_mat)

    column_name = 'classification' + str(cutoff)
    data_df[column_name] = classification

print(data_df)
result_df['precision'] = p
result_df['accuracy'] = a
result_df['recall'] = r
result_df['fallout'] = f

print(result_df)

plt.scatter(result_df['fallout'], result_df['recall'])


#fpr, tpr, thresholds = skm.roc_curve(data_df['true_class'], data_df['score'], pos_label=1)
#print(fpr)
#print(tpr)
#print(thresholds)
#plt.plot(fpr, tpr)
plt.show()
