import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def liners_regression(df, project_name):
    # Set up the figure and axes for the plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(f'class time 2 of {project_name}')

    # Plot accuracy over time
    axs[0, 0].plot(df['index'], df['accuracy'], marker='o', color='b')
    axs[0, 0].set_title('Accuracy over Time')
    axs[0, 0].set_xlabel('Index')
    axs[0, 0].set_ylabel('Accuracy')

    # Plot precision_macro over time
    axs[0, 1].plot(df['index'], df['precision_macro'], marker='o', color='g')
    axs[0, 1].set_title('Precision Macro over Time')
    axs[0, 1].set_xlabel('Index')
    axs[0, 1].set_ylabel('Precision Macro')

    # Plot recall_macro over time
    axs[1, 0].plot(df['index'], df['recall_macro'], marker='o', color='r')
    axs[1, 0].set_title('Recall Macro over Time')
    axs[1, 0].set_xlabel('Index')
    axs[1, 0].set_ylabel('Recall Macro')

    # Plot f1_macro over time
    axs[1, 1].plot(df['index'], df['f1_macro'], marker='o', color='m')
    axs[1, 1].set_title('F1 Macro over Time')
    axs[1, 1].set_xlabel('Index')
    axs[1, 1].set_ylabel('F1 Macro')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(f'../matplotlib/output/{project_name}_time_class2.png'))
    return plt.show()


if __name__ == '__main__':
    ozone = pd.read_parquet('../output/ozone_GBC_time_class2.parquet')
    ozone = ozone.reset_index()
    print("ozone describe", ozone.describe())
    ozone.to_csv('ozone_csv.csv')
    pulsar = pd.read_parquet('../output/pulsar_GBC_time_class2.parquet')
    pulsar = pulsar.reset_index()
    print("pulsar describe", pulsar.describe())

    ozone_plot = liners_regression( ozone, 'ozone')
    pulsar_plot = liners_regression(pulsar, 'pulsar')
