import numpy as np
import pandas as pd
from scipy import stats

def gaussian_test(col, values):
    stat1, p1 = stats.shapiro(values)
    stat2, p2 = stats.normaltest(values)

    print(f"Gaussian: {col}\n\t{p1:5f} (Shapiro-Wilk)\n\t{p2:5f} (D'Agostino's)")


def correlation_test(df):
    pearson = df.corr(method=lambda x, y: stats.pearsonr(x, y)[1])
    spearman = df.corr(method=lambda x, y: stats.spearmanr(x, y)[1])

    pearson = pearson.round(4)
    spearman = spearman.round(4)

    return pearson, spearman
