
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
print(digits.images.shape)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict([[ 0,  0,  1, 11, 14, 15,  3,  0,  0,  1, 13, 16, 12, 16,  8,  0,  0,  8, 16,  4,  6, 16,  5,  0,  0,  5, 15, 11, 13, 14,  0,  0,  0,  0,  2, 12, 16, 13,  0,  0,  0,  0,  0, 13, 16, 16,  6,  0,  0,  0,  0, 16, 16, 16, 7,  0,  0,  0,  0, 11, 13, 12,  1,  0]])
print(predicted)
#print(X_test[0])

import joblib
joblib.dump(clf, 'digit_recognizer_model.pkl')










