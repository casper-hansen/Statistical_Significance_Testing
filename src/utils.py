import seaborn as sns
import matplotlib.pyplot as plt

def save_correlation_map(correlation_map, save_name, title):
    plt.figure(figsize=(12, 6))
    sns.heatmap(correlation_map, annot=True)
    plt.title(title, fontsize=20)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')

