import seaborn as sns
import matplotlib.pyplot as plt

def save_correlation_map(stat, p_values, save_name, title):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))
    sns.heatmap(stat, annot=True, ax=axes[0])
    sns.heatmap(p_values, annot=True, ax=axes[1])
    axes[0].set_title(title + ' - Coefficients', fontsize=20)
    axes[1].set_title(title + ' - P-values', fontsize=20)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
