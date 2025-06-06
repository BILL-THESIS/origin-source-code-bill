import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_lgbm():
    ozone_lgbm = joblib.load("rank_f1/output_rank/ozone_optuna_result_rank_lgbm.pkl")
    pulsar_lgbm = joblib.load('rank_f1/output_rank/pulsar_optuna_result_rank_lgbm.pkl')
    seatunnel_lgbm = joblib.load("rank_f1/output_rank/seatunnel_optuna_result_rank_lgbm.pkl")

    return seatunnel_lgbm, pulsar_lgbm, ozone_lgbm


def f1_to_color(f1):
    if f1 <= 0.6:
        return 'red'
    elif f1 <= 0.7:
        return 'yellow'
    else:
        return 'green'

def plot_f1_scores(data, project_name, model_name):
    data['color'] = data['f1'].apply(
        lambda x: f1_to_color(x[0]) if isinstance(x, list) else f1_to_color(x))
    data['feature_group'] = data['feature_group'].apply(
        lambda x: str(tuple(x)) if isinstance(x, list) else str(x))

    plt.figure(figsize=(10, 5))
    # plt.scatter(data['feature_group'], data['f1'], c=data['color'], s=100)

    plt.scatter(range(len(data['feature_group'])), data['f1'], c=data['color'], s=100)

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
    plt.savefig(f"lightgbm/output/{project_name}_lgbm_f1_score_{model_name}.png")
    plt.show()


if __name__ == "__main__":
    # # main
    seatunnel_lgbm, pulsar_lgbm, ozone_lgbm = load_data_lgbm()


    # plot_f1_scores(ozone_lgbm, "ozone", "LightGBM")
    # plot_f1_scores(pulsar_lgbm, "pulsar", "LightGBM")
    # plot_f1_scores(seatunnel_lgbm, "seatunnel", "LightGBM")

    seatunnel_lgbm['Project Name'] = 'seatunnel'
    pulsar_lgbm['Project Name'] = 'pulsar'
    ozone_lgbm['Project Name'] = 'ozone'


    lgbm = pd.concat([seatunnel_lgbm, pulsar_lgbm, ozone_lgbm], axis=0, ignore_index=True)

    # plt.figure(figsize=(8, 5))
    # sns.scatterplot(data=lgbm, x='Project Name', y='f1', hue='Project Name', style='Project Name', s=120)
    # plt.axhline(0.6, color='red', linestyle='--', linewidth=0.8, label='≤ 0.6')
    # plt.axhline(0.7, color='yellow', linestyle='--', linewidth=0.8, label='≤ 0.7')
    # plt.title(f'F1 Score by LightGBM for Feature Group Smells')
    # plt.ylabel('F1 Score')
    # plt.ylim(0, 1)
    # plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"rank_f1/Compare_f1_potuna_3_projectsLGBM.png")
    # plt.show()

