import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import logging


def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = joblib.load(file)
            logging.info(f"Successfully loaded model from {file_path}")
            return model
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while loading the model from {file_path}: {e}")


def robust_outlier_detection(df):
    data = df['total_time']

    # median of the absolute deviations from the median (MAD)
    median = data.median()
    print("Median: ", median)

    mad = np.abs(data - median).median()
    print("MAD: ", mad)

    MADN = (mad / 0.6745)
    print("MADN: ", MADN)

    threshold = 2.24
    outlier = (data - median).abs() / MADN > threshold
    print("Sum outliers :", outlier.sum())

    # divided the dataset into two parts: normal and outliers
    df_outliers = df[outlier]
    df_normal = df[~outlier]

    return df_outliers, df_normal


def polt_3d_html(x_label, y_label, z_label):
    # Prepare the data for 3D plotting
    X = x_label
    Y = y_label
    Z = z_label

    fig = go.Figure(data=[
        go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=12, color=Z, colorscale='Viridis', opacity=0.8))])
    fig.update_layout(
        title=f"3D Plot of {z_label.name}",
        scene=dict(xaxis_title=f'{x_label.name}', yaxis_title=f'{y_label.name}', zaxis_title=f'{z_label.name}'))
    fig.select_xaxes('time01 (hours)')
    fig.select_yaxes('time12 (hours)')
    fig.show()
    fig.write_html(f"{z_label.name}_3.html", include_plotlyjs=True, full_html=True)
    return fig.show()


if __name__ == '__main__':
    model = load_model('../output/class_time_3_normal.parquet')
    model_smote = load_model('../output/class_time_3_smote.parquet')
    model_smote_cut = pd.read_pickle('../output/class_time_3_smote_utilize.pkl')
    model_original = pd.read_parquet('../output/seatunnel_all_information.parquet')

    # df_outliers, df_normal = robust_outlier_detection(model)
    df_outliers_smote, df_normal_smote = robust_outlier_detection(model_original)
    # df_outliers_original, df_normal_original = robust_outlier_detection(model_original)

    # x_index = df_normal['index_time01']
    # y_index = df_normal['index_time12']

    # x_index_smote = df_normal_smote['index_time01']
    # y_index_smote = df_normal_smote['index_time12']

    # z_label = df_normal['f1_macro']
    # z_label_smote = df_normal_smote['f1_smote']

    # plot_3d_index = polt_3d_html(x_index, y_index, z_label)
    # plot_3d_index_smote = polt_3d_html(x_index_smote, y_index_smote, z_label_smote)

    df_normal_smote.to_parquet('../output/time_modify_outlier.parquet')
