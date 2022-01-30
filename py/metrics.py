"""
常用指标计算
"""

from sklearn.metrics import confusion_matrix

y_pred = [0, 1, 1, 0, 1]
y_true = [0, 1, 1, 1, 1]

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print(tn, fp, fn, tp)
acc = (tp + tn) / len(y_true)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

f1 = 2 / (1 / precision + 1 / recall)

print(acc, precision, recall, f1)
