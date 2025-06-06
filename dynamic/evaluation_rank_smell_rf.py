import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_rf():
    seatunnel_rf = joblib.load("rank_f1/output_rank/seatunnal_optuna_result_rank_rf.pkl")
    ozone_rf = joblib.load("rank_f1/output_rank/ozone_optuna_result_rank_rf.pkl")
    pulsar_rf = joblib.load("rank_f1/output_rank/pulsar_optuna_result_rank_rf.pkl")
    return seatunnel_rf, pulsar_rf, ozone_rf,

def f1_to_color(f1):
    if f1 <= 0.6:
        return 'red'
    elif f1 <= 0.7:
        return 'yellow'
    else:
        return 'green'

def plot_f1_scores(data, project_name, model_name):
    data['color'] = data['result'].apply(
        lambda x: f1_to_color(x[0]) if isinstance(x, list) else f1_to_color(x))
    data['feature_group'] = data['feature_group'].apply(
        lambda x: str(tuple(x)) if isinstance(x, list) else str(x))

    plt.figure(figsize=(10, 5))
    # plt.scatter(data['feature_group'], data['result'], c=data['color'], s=100)

    plt.scatter(range(len(data['feature_group'])), data['result'], c=data['color'], s=100)

    plt.xticks(rotation=45, ha='right')
    plt.axhline(0.6, color='red', linestyle='--', linewidth=0.8, label='≤ 0.6')
    plt.axhline(0.7, color='yellow', linestyle='--', linewidth=0.8, label='≤ 0.7')

    plt.title(f'F1 Score by {model_name} \n Feature Group by {project_name}')
    plt.xlabel('Feature Group')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"lightgbm/output_lightgbm/{project_name}_lgbm_f1_score_{model_name}.png")
    plt.show()


if __name__ == "__main__":

    seatunnel_rf, pulsar_rf, ozone_rf  = load_data_rf()

    seatunnel_rf['Project Name'] = 'seatunnel'
    pulsar_rf['Project Name'] = 'pulsar'
    ozone_rf['Project Name'] = 'ozone'

    rf = pd.concat([seatunnel_rf, pulsar_rf, ozone_rf], axis=0, ignore_index=True)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=rf, x='Project Name', y='f1', hue='Project Name', style='Project Name', s=120)
    plt.axhline(0.6, color='red', linestyle='--', linewidth=0.8, label='≤ 0.6')
    plt.axhline(0.7, color='yellow', linestyle='--', linewidth=0.8, label='≤ 0.7')
    plt.title(f'F1 Score by Random Forest for  Feature Group Smells')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"rank_f1/Compare_f1_potuna_3_projectsRF.png")
    plt.show()

