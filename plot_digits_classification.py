
from poplib import CR
from sklearn import datasets, svm, metrics, tree
from itertools import product as pdt
from joblib import dump, load
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score as metric


def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label


# random split generator
def random_split_generator(num_sets):
    train = []
    dev = []
    test = []
    for i in range(0,num_sets):
        train_dev_test = np.array(np.random.random(3))
        train_dev_test /= train_dev_test.sum()
        train.append(train_dev_test[0])
        dev.append(train_dev_test[1])
        test.append(train_dev_test[2])
    return train, dev, test




def train_dev_test_split(data, label, train_frac, dev_frac):

    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def h_param_tuning_svm(h_param_comb, clf, x_train, y_train, x_dev, y_dev):
    best_accuracy = -1.0
    best_model = None
    best_h_params = None
    # 2. For every combination-of-hyper-parameter values
    for Gamma,c in h_param_comb:

        # PART: setting up hyperparameter
        h_params = {'gamma':Gamma, 'C':c}
        hyper_params = h_params
        clf.set_params(**hyper_params)

        # PART: Train model
        # 2.a train the model
        # Learn the digits on the train subset
        clf.fit(x_train, y_train)

        # PART: get dev set predictions
        dev_prediction = clf.predict(x_dev)

        # 2.b compute the accuracy on the validation set
        model_accuracy = metric(y_dev, dev_prediction)

        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
        if model_accuracy > best_accuracy:
            best_accuracy = model_accuracy
            best_model = clf
            best_h_params = h_params
            print("Found new best metric for SVM with :" + str(h_params))
            print("New best val metric for SVM:" + str(model_accuracy))
    return best_model, best_accuracy, best_h_params




def h_param_tuning_dect(h_param_comb, clf, x_train, y_train, x_dev, y_dev):
    best_accuracy = -1.0
    best_model = None
    best_h_params = None
    # 2. For every combination-of-hyper-parameter values
    for Criterion, Splitter in h_param_comb:

        # PART: setting up hyperparameter
        h_params = {'criterion':Criterion, 'splitter':Splitter}
        hyper_params = h_params
        clf.set_params(**hyper_params)

        # PART: Train model
        # 2.a train the model
        # Learn the digits on the train subset
        clf.fit(x_train, y_train)

        # print(cur_h_params)
        # PART: get dev set predictions
        dev_prediction = clf.predict(x_dev)

        # 2.b compute the accuracy on the validation set
        model_accuracy = metric(y_dev, dev_prediction)

        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
        if model_accuracy > best_accuracy:
            best_accuracy = model_accuracy
            best_model = clf
            best_h_params = h_params
            print("Found new best metric for Decision Tree Classifier with :" + str(h_params))
            print("New best val metric for Decision Tree Classifier:" + str(model_accuracy))
    return best_model, best_accuracy, best_h_params

def get_accuracy(y_test, predicted):
    accuracy = metric(y_test, predicted)
    return accuracy


def get_mean(arr):
    _mean = np.mean(np.array(arr))
    return _mean
def get_std(arr):
    _std = np.std(np.array(arr))
    return _std


train_fracs, dev_fracs, test_fracs = random_split_generator(5)
#print(train_frac,'\n', dev_frac, '\n', test_frac)

# set the hyper parameters SVM
GAMMA = [0.0001, 0.0004, 0.0005, 0.0008, 0.001]
C = [0.5, 2.0, 3.0, 4.0, 5.0]

# 2. set hyper parameters for decision tree classifier
Criterion = ['gini', 'entropy']
Splitter = ['best', 'random']


# Loading dataset
digits = datasets.load_digits()
#data_viz(digits)
data, label = preprocess_digits(digits)

# Create a svm classifier
clf_svm = svm.SVC()

# create a decision tree classifier
clf_dect = tree.DecisionTreeClassifier()

best_prediction_accuracy_svm =[]
best_prediction_accuracy_dect = []


for i in range(0, 5):
    # Creating hyperparameters combination for SVM
    h_param_comb_svm = pdt(GAMMA,C)
    # Creating hyperparameter Combination for Decision Tree Classifier
    h_param_comb_dect = pdt(Criterion, Splitter)
    

    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_fracs[i], dev_fracs[i]
    )


    # getting best model for SVM
    best_model_svm, best_metric_svm, best_h_params_svm = h_param_tuning_svm(
        h_param_comb_svm, clf_svm, x_train, y_train, x_dev, y_dev)
    

    # save the best_model for SVM 
    best_param_config_svm = "_".join([param + "=" + str(best_h_params_svm[param]) for param in best_h_params_svm])
    dump(best_model_svm, "svm_" + best_param_config_svm + ".joblib")

    # getting best model for Decision Tree Classifier
    best_model_dect, best_metric_dect, best_h_params_dect = h_param_tuning_dect(
        h_param_comb_dect, clf_dect, x_train, y_train, x_dev, y_dev)

    # save the best model for Decision Tree Classifier
    best_param_config_dect = "_".join([param + "=" + str(best_h_params_dect[param]) for param in best_h_params_dect])
    dump(best_model_dect, "dect_" + best_param_config_dect + ".joblib")


    # load the best_model for SVM
    best_model_svm = load("svm_" + best_param_config_svm + ".joblib")

    # load the best_model for Decision Tree Classifier
    best_model_dect = load("dect_" + best_param_config_dect + ".joblib")

    # Predict the value of the digit on the test set for SVM Model
    predicted_svm = best_model_svm.predict(x_test)

    # Predict the value of the digit on the test set for Decision Tree Classifier
    predicted_dect = best_model_dect.predict(x_test)

    #pred_image_viz(x_test, predicted_svm)

    # Compute evaluation metrics for SVM
    print(
        f"Classification report for SVM classifier {clf_svm}:\n"
        f"{metrics.classification_report(y_test, predicted_svm)}\n"
    )

    print("Best hyperparameters for SVM Classifier were:")
    print(best_h_params_svm)
    print('\n\n')

    # Compute evaluation metrics for Decision Tree Classifier
    print(
        f"Classification report for Decision Tree classifier {clf_dect}:\n"
        f"{metrics.classification_report(y_test, predicted_dect)}\n"
    )

    print("Best hyperparameters for Decision Tree Classifier were:")
    print(best_h_params_svm)
    print('\n\n\n\n')

    # prediction accuracy for each train, dev, test set for svm and decision tree
    predict_accuracy_svm = get_accuracy(y_test, predicted_svm)
    predict_accuracy_dect = get_accuracy(y_test, predicted_dect)

    # storing accuracies for future use
    best_prediction_accuracy_svm.append(predict_accuracy_svm)
    best_prediction_accuracy_dect.append(predict_accuracy_dect)
print("accuracy list svm: ", best_prediction_accuracy_svm)
print("accuracy list decision tree: ", best_prediction_accuracy_dect)
print('\n\n')

# calculating mean and standard deviation of accuracy for svm and decision tree classifier
svm_accuracy_mean = get_mean(best_prediction_accuracy_svm)
svm_accuracy_std = get_std(best_prediction_accuracy_svm)
dect_accuracy_mean = get_mean(best_prediction_accuracy_dect)
dect_accuracy_std = get_std(best_prediction_accuracy_dect)

print("Mean accuracy for SVM: ", svm_accuracy_mean, '\nStandard deviation for accuarcy for SVM', svm_accuracy_std)
print("Mean accuracy for Decision Tree Classifier: ", dect_accuracy_mean, '\nStandard deviation for accuarcy for Decision Tree Classifier', dect_accuracy_std)
