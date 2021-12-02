import seaborn as sns
import matplotlib.pyplot as plt

def save_correlation_map(correlation_map, save_name, title):
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(correlation_map, annot=True)
    plt.title(title, fontsize=18)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
