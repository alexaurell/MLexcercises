import numpy as np
from sklearn.metrics import confusion_matrix as cm

class trivialClassifier():
    def __init__(self):
        self.parameter = 0 # Median of applications

    def fit(self, X):
        self.parameter = X.median()
        return self.parameter

    def predict(self, y):
        if self.parameter == 0:
            print('The model has not been trained')
            pass
        return np.where(y >= self.parameter, 1, 0)

    def score(self, y, test, print=False):
        '''
        :param y: testing data on feature
        :param test: testing data on classes
        :return: accuracy
        '''
        pred = self.predict(y)      # Array of predicted class
        conf_mat = cm(test, pred)   # Confusion matrix
        accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / np.sum(np.sum(conf_mat))
        if print == True:
            print('Acuracy of prediction: ' + str(accuracy))
        return accuracy


