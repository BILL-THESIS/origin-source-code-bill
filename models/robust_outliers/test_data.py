import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample data
data = {
    'Date': pd.date_range(start="2000-01-01", periods=12, freq='Y'),
    'Patch_Count': np.random.randint(100, 1000, 12),
    'Special_Event': [False, True, False, False, True, False, False, True, False, False, True, False],
    'Minor_Version': [False, False, False, True, False, False, True, False, False, True, False, False]
}

df = pd.DataFrame(data)

# Plotting the main bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Bar plot
ax.bar(df['Date'], df['Patch_Count'], color='gray', edgecolor='black')

# Adding annotations for special events and minor versions
for i, row in df.iterrows():
    if row['Special_Event']:
        ax.annotate(
            'Special Event',
            xy=(row['Date'], row['Patch_Count']),
            xytext=(row['Date'], row['Patch_Count'] + 200),
            arrowprops=dict(facecolor='magenta', arrowstyle="->"),
            ha='center'
        )
    if row['Minor_Version']:
        ax.annotate(
            'Minor Version',
            xy=(row['Date'], row['Patch_Count']),
            xytext=(row['Date'], row['Patch_Count'] + 200),
            arrowprops=dict(facecolor='magenta', arrowstyle="->"),
            ha='center'
        )

# Setting the title and labels
ax.set_title("Empirical Research and LOC")
ax.set_xlabel("Timeline")
ax.set_ylabel("Patch Count Level")

# Adding custom phases under the timeline
phases = ["E", "S", "U", "S", "E", "S", "U", "S", "E", "U", "S", "E"]
for i, phase in enumerate(phases):
    ax.text(df['Date'].iloc[i], -150, phase, ha='center', va='center', color='black')

# Adding labels for the phases
ax.annotate("Exploration Oriented", xy=(df['Date'].iloc[0], -400), xytext=(df['Date'].iloc[0], -400), ha='center')
ax.annotate("Service Oriented", xy=(df['Date'].iloc[1], -400), xytext=(df['Date'].iloc[1], -400), ha='center')
ax.annotate("Utility Oriented", xy=(df['Date'].iloc[2], -400), xytext=(df['Date'].iloc[2], -400), ha='center')

# Customizing ticks and grid
ax.set_xticks(df['Date'])
ax.set_xticklabels(df['Date'].dt.strftime('%Y'), rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
