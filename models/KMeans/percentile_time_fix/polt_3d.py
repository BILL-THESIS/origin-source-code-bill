import os
import joblib
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import plotly.graph_objects as go
from scipy.interpolate import griddata


def polt_3d_subplot(x_label, y_label, z_label):
    # Prepare the data for 3D plotting
    X = x_label
    Y = y_label
    Z = z_label

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    sc = ax.scatter(X, Y, Z, c=Z, cmap='viridis', marker='o')

    fig = go.Figure(data=[
        go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=12, color=Z, colorscale='Viridis', opacity=0.8))])
    fig.select_xaxes('time01 (hours)')
    fig.select_yaxes('time12 (hours)')
    fig.show()
    fig.write_html("test_02.html", include_plotlyjs=True, full_html=True)

    # # Add color bar which maps values to colors
    # plt.colorbar(sc)
    #
    # # Set labels and limits
    # ax.set_xlabel('time01 (hours)')
    # ax.set_ylabel('time12 (hours)')
    # # how to get the variable of z label to save str in the name of table
    # ax.set_zlabel(f'{z_label.name}')
    # plt.title(f'{z_label.name} vs time01 and time12')
    #
    # plt.savefig(os.path.join(f'../../KMeans/output/{z_label.name}_point.png'))
    # plt.show()

    return fig.show()


def polt_3d_grid(x_label, y_label, z_label):
    # Prepare the data for 3D plotting
    X = x_label
    Y = y_label
    Z = z_label

    # Create a grid of points
    xi = np.linspace(X.min(), X.max(), 100)
    yi = np.linspace(Y.min(), Y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate Z values on the grid
    zi = griddata((X, Y), Z, (xi, yi), method='cubic')

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', lw=0.5, rstride=1, cstride=1, alpha=0.7)

    # Add color bar which maps values to colors
    fig.colorbar(surf)

    # Set labels and limits
    ax.set_xlabel('time01 (hours)')
    ax.set_ylabel('time12 (hours)')
    ax.set_zlabel(f'{z_label.name}')
    plt.title(f'{z_label.name} vs time01 and time12')
    plt.savefig(os.path.join(f'../../KMeans/output/{z_label.name}_surface.png'))
    plt.show()

    return plt.show()


if __name__ == '__main__':
    # Load the data
    with open(os.path.join('../../KMeans/output/class_time_2_smote.parquet'), 'rb') as f:
        df_class_2_smote = joblib.load(f)

    with open(os.path.join('../../KMeans/output/class_time_3_smote.parquet'), 'rb') as f:
        df_class_3_smote = joblib.load(f)

    df_utilize = pd.read_pickle('../../KMeans/output/class_time_3_smote_utilize.pkl')

    # Convert 'time01' and 'time12' to total seconds in hours
    df_class_3_smote['time01'] = pd.to_timedelta(df_class_3_smote['time01']).dt.total_seconds() / 3600
    df_class_3_smote['time12'] = pd.to_timedelta(df_class_3_smote['time12']).dt.total_seconds() / 3600

    # Prepare the data for 3D plotting
    X = df_class_3_smote['index_time01'].values
    Y = df_class_3_smote['index_time12'].values
    # Assuming 'f1_macro' & 'f1_smote' is a single value for Z
    z_normal = df_class_3_smote['f1_macro']
    z_smote = df_class_3_smote['f1_smote']

    x_utilize = df_utilize['index_time01'].values
    y_utilize = df_utilize['index_time12'].values
    z_utilize = df_utilize['f1_smote']

    plot_normal = polt_3d_subplot(X, Y, z_normal)
    plot_smote = polt_3d_subplot(X, Y, z_smote)
    plot_normal_grid = polt_3d_grid(X, Y, z_normal)
    plot_smote_grid = polt_3d_grid(X, Y, z_smote)
    p1 = polt_3d_subplot(x_utilize, y_utilize, z_utilize)
    p2 = polt_3d_grid(x_utilize, y_utilize, z_utilize)
