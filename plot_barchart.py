import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['patch.force_edgecolor'] = True
plt.rcParams['patch.facecolor'] = 'none'
plt.rcParams.update({'font.size': 20}) # Sets a global font size of 14




# Load CSV file
csv_file = "results_new1/optimized_result copy.csv"
df = pd.read_csv(csv_file)

# Values
options = df['option']
T_vals = df['T']
utility_vals = df['Cost']
loss_vals = df['loss']

# Color palette (will cycle if more options than colors)
colors = plt.cm.tab10(np.linspace(0, 1, len(options)))

# Function to plot a single bar chart
def plot_bar(values, ylabel, title, filename):
    x = np.arange(len(options))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, values, width, color=colors)

    # Labels and title
    ax.set_xlabel('Approach',fontsize=20)
    ax.set_ylabel(ylabel,fontsize=20)
    # ax.set_title(title,fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(options)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'results_new1/{filename}',bbox_inches='tight')
    # plt.show()
    plt.close()

# Plot each metric
plot_bar(T_vals, 'T', 'T Comparison', 'T_comparison.pdf')
plot_bar(utility_vals, 'Cost', 'Cost Comparison', 'cost_comparison.pdf')
plot_bar(loss_vals, 'Loss', 'Loss Comparison', 'Loss_comparison.pdf')
