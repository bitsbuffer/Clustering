import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score,confusion_matrix, ConfusionMatrixDisplay


def train_nn(X, y):
    classifier = MLPClassifier(hidden_layer_sizes=[64, 32, 16], activation='relu',
                               solver='adam',
                               alpha=0.01,
                               batch_size='auto',
                               learning_rate='constant',
                               max_iter=200,
                               random_state=1234,
                               early_stopping=True,
                               verbose=1,
                               validation_fraction=0.1,
                               n_iter_no_change=10)
    classifier.fit(X, y)
    return classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('-t', '--train_data_name', type=str, help='Training dataset name', required=True)
    args = parser.parse_args()

    X = pd.read_csv(args.train_data_name, compression="zip")
    y = X.pop('target')

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1234)
    scalar = StandardScaler()
    train_x = scalar.fit_transform(train_x)
    test_x = scalar.transform(test_x)

    classifier = train_nn(train_x, train_y)
    test_pred = classifier.predict(test_x)
    cm = confusion_matrix(test_y, test_pred)
    print("Confusion Matrix")
    print(cm)
    print(f"F1 score is {f1_score(test_y, test_pred)}")
    print(f"Balanced Accuracy is {balanced_accuracy_score(test_y, test_pred)}")
    print(f"ROC AUC is {roc_auc_score(test_y, test_pred)}")
    ConfusionMatrixDisplay(cm)