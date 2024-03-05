from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, classification_report

svc = SVC(C=10, kernel='sigmoid')

with open('train_vectorized.pickle', 'rb') as file:
    X_train_text = pickle.load(file)

with open('test_vectorized.pickle', 'rb') as file:
    X_test_text = pickle.load(file)

with open('train.pickle', 'rb') as file:
    X_train = pickle.load(file)

with open('test.pickle', 'rb') as file:
    X_test = pickle.load(file)

y_train = X_train.iloc[:,1]
y_test = X_test.iloc[:, 1]

print("\nSVC : ")

svc.fit(X_train_text, y_train)
y_pred = svc.predict(X_test_text)
print('classification_report : ', classification_report(y_test, y_pred))
print('confusion matrix : ', confusion_matrix(y_true=y_test, y_pred=y_pred))
print('accuracy score :' , accuracy_score(y_test, y_pred))
recall_average = recall_score(y_test, y_pred, average="binary", pos_label='real')
print("recall_average : ", recall_average)
precision_average = precision_score(y_test, y_pred, average="binary", pos_label='real')
print("precision_average : ", precision_average)
f1_average = f1_score(y_test, y_pred, average="binary", pos_label='real')
print("f1_score : ", f1_average)