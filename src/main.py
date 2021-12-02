from math import sin
import cv2
import utils
import model
import pandas as pd
import significance

df = pd.read_csv('data/diabetes.csv')
df["OldOverweight"] = df.apply(lambda x: True if x.Age >= 50 and x.BMI >= 25.0 else False, axis=1)

def distribution_test(df):
    for col in df.columns:
        curr_values = df.loc[:, col]
        significance.gaussian_test(col, curr_values)


def correlation_test(df):
    pearson, spearman = significance.correlation_test(df)

    utils.save_correlation_map(pearson, 'article/images/heatmap_pearson.png', title="Pearson's Correlation")
    utils.save_correlation_map(spearman, 'article/images/heatmap_spearman.png', title="Spearman's Correlation")


def pregnancy_prediction(df):
    x_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome', 'OldOverweight']
    y_feature = 'Pregnancies'
    
    model.linear_regression_model(df, x_features, y_feature)

    x_features = ['Age', 'Outcome', 'OldOverweight']
    model.linear_regression_model(df, x_features, y_feature)


def bold_text():
    img1 = cv2.imread('data/logo.png')
    img2 = cv2.imread('data/heading.png')
    img3 = cv2.imread('data/body.png')

    bold1 = model.compute_thickness(img1)
    bold2 = model.compute_thickness(img2)
    bold3 = model.compute_thickness(img3)

    print(bold1, bold2, bold3)

    significance.boldness_test(bold1, bold2, bold3)
    

if __name__ == '__main__':
    bold_text()