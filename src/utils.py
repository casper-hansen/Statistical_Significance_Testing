import seaborn as sns
import matplotlib.pyplot as plt

def save_correlation_map(correlation_map, save_name):
    plt.figure(figsize=(12, 6))
    sns.heatmap(correlation_map, annot=True)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
