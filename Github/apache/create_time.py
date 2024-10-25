import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame structure
data = {
    'Year': [2021, 2021, 2021, 2022, 2022, 2022],
    'time_hour_class': [0, 1, 2, 0, 1, 2],
    'sum_diff_x': [10, 15, 7, 8, 12, 5],
    'sum_diff_y': [5, 13, 9, 7, 10, 6]
}

df = pd.DataFrame(data)

# Setting up the plot
plt.figure(figsize=(12, 8))

# Pivot the data for easy plotting
pivot_df = df.pivot(index='Year', columns='time_hour_class', values=['sum_diff_x', 'sum_diff_y'])

# Create a grouped bar plot
for year in pivot_df.index:
    time_classes = pivot_df.columns.levels[1]  # Get the time_hour_class values
    sum_diff_x = pivot_df.loc[year, ('sum_diff_x', time_classes)]
    sum_diff_y = pivot_df.loc[year, ('sum_diff_y', time_classes)]

    # Plotting the bars for each year
    plt.barh([f'{year} - {tc}' for tc in time_classes], sum_diff_x, color='blue', alpha=0.6,
             label='sum_diff_x' if year == pivot_df.index[0] else "")
    plt.barh([f'{year} - {tc}' for tc in time_classes], sum_diff_y, color='orange', alpha=0.6, left=sum_diff_x,
             label='sum_diff_y' if year == pivot_df.index[0] else "")

# Adding labels and title
plt.xlabel('Number of Differences')
plt.ylabel('Year and Time Hour Class')
plt.title('Changes in sum_diff_x and sum_diff_y Across Years and Time Classes')
plt.legend()
plt.show()
