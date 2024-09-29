import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def polt_3d_html(x_label, y_label, z_label, project_name):
    # Prepare the data for 3D plotting
    X = x_label
    Y = y_label
    Z = z_label

    # Plotting
    fig = go.Figure(data=[
        go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=12, color=Z, colorscale='Viridis', opacity=0.8))])
    fig.update_layout(
        title=f"3D Plot of {project_name} {z_label.name}",
        scene=dict(xaxis_title=f'{x_label.name}', yaxis_title=f'{y_label.name}', zaxis_title=f'{z_label.name}'))

    fig.select_xaxes('time01 (hours)')
    fig.select_yaxes('time12 (hours)')
    fig.show()
    fig.write_html(f"../matplotlib/output/{project_name}_{z_label.name}.html", include_plotlyjs=True, full_html=True)
    return fig.show()


def polt_3d_subplot(x_label, y_label, z_label, project_name):
    # Prepare the data for 3D plotting
    X = x_label
    Y = y_label
    Z = z_label

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    sc = ax.scatter(X, Y, Z, c=Z, cmap='viridis', marker='o')

    # Add color bar which maps values to colors
    plt.colorbar(sc)

    # Set labels and limits
    ax.set_xlabel('time01 (hours)')
    ax.set_ylabel('time12 (hours)')
    # how to get the variable of z label to save str in the name of table
    ax.set_zlabel(f'{z_label.name}')
    plt.title(f'{project_name} {z_label.name} vs time01 and time12')

    plt.savefig(os.path.join(f'../matplotlib/output/{project_name}_{z_label.name}_point.png'))

    return plt.show()


if __name__ == '__main__':
    with open(os.path.join('../output/ozone_GBC_class_time3_smote_newversion_9Sep.parquet'), 'rb') as f:
        ozone = joblib.load(f)

    with open(os.path.join('../output/pulsar_GBC_class_time3_smote_new.parquet'), 'rb') as f:
        pulsar = joblib.load(f)

    ozone_plot = polt_3d_html(ozone['index_time01'], ozone['index_time12'], ozone['f1_macro'], 'ozone')
    ozone_plot_smote = polt_3d_html(ozone['index_time01'], ozone['index_time12'], ozone['f1_smote'], 'ozone')
    # pulsar_plot = polt_3d_html(pulsar['index_time01'], pulsar['index_time12'], pulsar['f1_macro'], 'pulsar')
    # pulsar_plot_smote = polt_3d_html(pulsar['index_time01'], pulsar['index_time12'], pulsar['f1_smote'], 'pulsar')
    #
    ozone_plot2 = polt_3d_subplot(ozone['index_time01'], ozone['index_time12'], ozone['f1_macro'], 'ozone')
    ozone_plot2_smote = polt_3d_subplot(ozone['index_time01'], ozone['index_time12'], ozone['f1_smote'], 'ozone')
    # pulsar_plot2 = polt_3d_subplot(pulsar['index_time01'], pulsar['index_time12'], pulsar['f1_macro'], 'pulsar')
    # pulsar_plot2_smote = polt_3d_subplot(pulsar['index_time01'], pulsar['index_time12'], pulsar['f1_smote'], 'pulsar')
