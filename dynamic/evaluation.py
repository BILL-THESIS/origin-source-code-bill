import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_lgbm():
    ozone_lgbm = "lightgbm/output_lightgbm/ozone_optuna_result.pkl"
    pulsar_lgbm = "lightgbm/output_lightgbm/pulsar_optuna_result.pkl"
    seatunnel_lgbm = "lightgbm/output_lightgbm/seatunnel_optuna_result.pkl"

    ozone_lgbm = joblib.load(ozone_lgbm)
    pulsar_lgbm = joblib.load(pulsar_lgbm)
    seatunnel_lgbm = joblib.load(seatunnel_lgbm)

    return ozone_lgbm, pulsar_lgbm, seatunnel_lgbm

def load_data_rf():

    ozone_rf = "randomforest/output_randomforest/ozone_optuna_result_rdf.pkl"
    pulsar_rf = "randomforest/output_randomforest/pulsar_optuna_result_rdf.pkl"
    seatunnel_rf = "randomforest/output_randomforest/seatunnal_optuna_result_rdf.pkl"

    ozone_rf = joblib.load(ozone_rf)
    pulsar_rf = joblib.load(pulsar_rf)
    seatunnel_rf = joblib.load(seatunnel_rf)

    return ozone_rf, pulsar_rf, seatunnel_rf

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
    # main
    ozone_lgbm, _ , seatunnel_lgbm = load_data_lgbm()
    ozone_rf, _ , seatunnel_rf = load_data_rf()
    pulsar_lgbm = joblib.load("lightgbm/output_lightgbm/pulsar_optuna_result_each_smell.pkl")
    pulsar_rf = joblib.load("randomforest/output_randomforest/pulsar_optuna_result_each_smell_rdf.pkl")

    # plot_f1_scores(ozone_lgbm, "ozone", "LightGBM")
    # plot_f1_scores(pulsar_lgbm, "pulsar", "LightGBM")
    # plot_f1_scores(seatunnel_lgbm, "seatunnel", "LightGBM")
    #
    # plot_f1_scores(ozone_rf, project_name="ozone", model_name="Random Forest")
    # plot_f1_scores(pulsar_rf, project_name="pulsar", model_name="Random Forest")
    # plot_f1_scores(seatunnel_rf, project_name="seatunnel", model_name="Random Forest")

    ozone_lgbm['Project Name'] = 'ozone'
    ozone_rf['Project Name'] = 'ozone'
    pulsar_lgbm['Project Name'] = 'pulsar'
    pulsar_rf['Project Name'] = 'pulsar'
    # seatunnel_lgbm['Project Name'] = 'seatunnel'
    # seatunnel_rf['Project Name'] = 'seatunnel'

    lgbm = pd.concat([ozone_lgbm, pulsar_lgbm, seatunnel_lgbm], axis=0, ignore_index=True)
    rf = pd.concat([ozone_rf, pulsar_rf, seatunnel_rf], axis=0, ignore_index=True)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=rf, x='Project Name', y='result', hue='Project Name', style='Project Name', s=120)
    plt.axhline(0.6, color='red', linestyle='--', linewidth=0.8, label='≤ 0.6')
    plt.axhline(0.7, color='yellow', linestyle='--', linewidth=0.8, label='≤ 0.7')
    # plt.title(f'F1 Score by LightGBM for Each Code Smell Type')
    plt.title(f'F1 Score by Random Forest for Each Code Smell Type')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(f"lightgbm/output_lightgbm/Compare_f1_potuna_each_smell_LGBM.png")
    plt.savefig(f"lightgbm/output_lightgbm/Compare_f1_potuna_each_smell_RF.png")
    plt.show()

