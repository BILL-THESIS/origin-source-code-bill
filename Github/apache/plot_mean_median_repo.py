import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Example DataFrame for each repo
data = {
    'repo': ['ozone', 'seatunnel', 'pulsar'],
    'mean': [12, 4, 10],
    'median': [2, 2, 2],
    'shape': [73, 91, 893]
}

df = pd.DataFrame(data)

# Set bar positions
x = np.arange(len(df['repo']))
width = 0.35  # Bar width

fig, ax = plt.subplots()

# Create bars for mean and median
bars_mean = ax.bar(x - width/2, df['mean'], width, label='Mean', color='lightblue')
bars_median = ax.bar(x + width/2, df['median'], width, label='Median', color='orange')

# Set x and y axis labels and title
ax.set_xlabel('Repository')
ax.set_ylabel('Time (Days)')
ax.set_title('Mean and Median Time each Repository')
ax.set_xticks(x)
ax.set_xticklabels(df['shape'].astype(str) + ' ' + df['repo'])
ax.legend()
plt.savefig('output/mean_median_time_by_repository.png')  # Save the plot as a PNG file
plt.show()