import utils
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

    utils.save_correlation_map(pearson, 'heatmap_pearson.png', title="Pearson's Correlation")
    utils.save_correlation_map(spearman, 'heatmap_spearman.png', title="Spearman's Correlation")

