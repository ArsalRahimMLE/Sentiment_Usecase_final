from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from numpy import array
def fetch_model():
    """
    load model from pickle file
    :return:
    """
    model = pickle.load(open("staticfiles/model.pkl", "rb"))
    return model


def calculate_accuracy(ypred, y_test):
    """
    Calculate accuracy of model
    :param ypred:
    :param y_test:
    :return:
    """
    accuracy = accuracy_score(ypred, y_test)
    return accuracy


def calculate_precision(ypred, y_test):
    """
    Calculate accuracy of model
    :param ypred:
    :param y_test:
    :return:
    """
    precision = precision_score(ypred, y_test)
    return precision


def calculate_recall(ypred, y_test):
    """
    Calculate recall of model
    :param ypred:
    :param y_test:
    :return:
    """
    recall = recall_score(ypred, y_test)
    return recall


def fetch_model_nb():
    """
    load  Naive bayes model from pickle file
    :return:
    """
    model_nb = pickle.load(open("staticfiles/model_nb.pkl", "rb"))
    return model_nb


def fetch_model_rf():
    """
    load  Random Forest model from pickle file
    :return:
    """
    model_rf = pickle.load(open("staticfiles/model_rf.pkl", "rb"))
    return model_rf


def fetch_model_svm():
    """
    load  Support Vector Machine model from pickle file
    :return:
    """
    model_svm = pickle.load(open("staticfiles/model_svm.pkl", "rb"))
    return model_svm



def display_confusion_matrix(ypred, y_test):
    """
        Calculate confusion matrix of model
        :param ypred:
        :param y_test:
        :return:
        """
    Confusion_Matrix = confusion_matrix(ypred, y_test)
    return Confusion_Matrix

def display_roc_curve (y_test, preds):

    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_values = array([fpr, tpr, threshold])
    return roc_values

