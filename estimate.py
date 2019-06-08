import cv2
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix

i=cv2.imread("C:/tmp/epoch_61.jpg",0);
i1=i[0:6911,216:432]
i2=i[0:6911,432:648]
v1=i1.reshape(-1,1)
v2=i2.reshape(-1,1)
y_test = np.where(v1>128, 1, 0)
preds= v2/255.
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
