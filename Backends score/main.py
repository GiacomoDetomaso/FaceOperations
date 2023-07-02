import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
import os

#actual = numpy.random.binomial(1,.9,size = 1000)
#predicted = numpy.random.binomial(1,.9,size = 1000)


a = [0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 2, 2, 1, 0]
p = [0, 1, 0, 2, 1, 0, 1, 2, 0, 1, 1, 2, 0, 1]

confusion_matrix = metrics.multilabel_confusion_matrix(a, p, labels=[0, 1, 2])
print(confusion_matrix[0].ravel())

# accuracy = metrics.accuracy_score(a, p)
# precision = metrics.precision_score(a, p)
# recall = metrics.recall_score(a, p, labels=[0, 1, 2], average='macro')
# print(precision) 


dir = "dataset/full/"

print("Numero elementi: ", len(os.listdir(dir)))

actual = []

