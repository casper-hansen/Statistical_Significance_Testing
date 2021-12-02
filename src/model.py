import cv2
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

def data_split(df, x_features, y_feature):
    X_train = df.loc[:, x_features]
    y_train = df.loc[:, y_feature]

    return train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def linear_regression_model(df, x_features, y_feature):
    X_train, X_test, y_train, y_test = data_split(df, x_features, y_feature)

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)

    print(f'Linear Regression R^2: {r2}')


def logistic_regression_model(df, x_features, y_feature):
    X_train, X_test, y_train, y_test = data_split(df, x_features, y_feature)

    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    print(f'Logistic Regression Accuracy: {accuracy}')


def compute_thickness(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(img)
    thinned = cv2.ximgproc.thinning(img)

    return (np.sum(inverted == 255) - np.sum(thinned == 255)) / np.sum(thinned == 255)