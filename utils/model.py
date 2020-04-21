import glob
import os
import random
import string
from operator import itemgetter

import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from utils.plot import plot_cm, plot_pca

WEIGHTS_DIR = 'weights/'


def latest_modified_weight():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def load_model():
    """

    :param path: weight path
    :return: load model based on the path
    """
    path = latest_modified_weight()

    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def generate_model_name(size=5):
    """

    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def split_method(features, labels):
    """
    run train_test_split method
    :param features: 'income'', 'age', 'loan'
    :param labels: 'default'
    :return:
    """
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.3)
    plot_pca(feature_train, label_train, 'train', 'split')

    model = GaussianNB()
    model.fit(feature_train, label_train)

    predictions = model.predict(feature_test)

    cm = confusion_matrix(label_test, predictions)

    plot_cm(cm, meth='split')

    plot_pca(feature_test, predictions, 'test', 'split')

    print(accuracy_score(label_test, predictions))

    ans = input('Do you want to save the model weight? ')
    if ans in ('yes', '1'):
        model_name = WEIGHTS_DIR + 'NaivBay-' + generate_model_name(5) + '.pkl'
        with open(model_name, 'wb') as f:
            joblib.dump(value=model, filename=f, compress=3)
            print(f'Model saved at {model_name}')

    return model


def cv_method(features, labels):
    """
    run Cross-Validation method
    :param features: 'income'', 'age', 'loan'
    :param labels: 'default'
    :return:
    """
    plot_pca(features, labels, 'train+test_data', meth='cv')

    cv_results = cross_validate(GaussianNB(),
                                features, labels,
                                return_estimator=True, return_train_score=True, cv=10)
    estimator_testscore = zip(list(cv_results['estimator']), cv_results['test_score'])
    model = max(estimator_testscore, key=itemgetter(1))[0]

    preds = model.predict(features)
    cm = confusion_matrix(labels, preds)

    plot_cm(cm, meth='cv')

    plot_pca(features, preds, 'test', meth='cv')

    print(accuracy_score(labels, preds))

    ans = input('Do you want to save the model weight? ')
    if ans in ('yes', '1'):
        model_name = WEIGHTS_DIR + 'NaivBay-' + generate_model_name(5) + '.pkl'
        with open(model_name, 'wb') as f:
            joblib.dump(value=model, filename=f, compress=3)
            print(f'Model saved at {model_name}')

    return model


def train_model(features, labels, method):
    """
    run appropriate training method
    :param features: 'income'', 'age', 'loan'
    :param labels: 'default'
    :param method: 'cv' for Cross-Validation method , 'split' for train_test_split method
    :return:
    """
    return training_methods[method](features, labels)


training_methods = {
    'split': split_method,
    'cv': cv_method
}
