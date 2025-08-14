# Plotting utilities for Ising ML criticality project
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

def plot_adversarial_results(df_adversarial, df_ordinary, scatter_info_dict, G_label, save_dir="Generated_plots"):
    """
    Plot adversarial and ordinary results for a given G value.
    Args:
        df_adversarial: DataFrame with adversarial results (columns: 'Ts', 'mu', 'ordinary_data_avg_pred', 'shuffled_data_avg_pred')
        df_ordinary: DataFrame with ordinary results (columns: 'Ts', 'ordinary_data_avg_pred', 'shuffled_data_avg_pred')
        scatter_info_dict: dict with keys 'Ts', 'outputs', 'T0', 'T1'
        G_label: str, e.g. '0', '0_1', '0_5'
        save_dir: directory to save plots
    """
    plt.figure(dpi=150)
    plt.plot(df_adversarial["Ts"], df_adversarial["mu"], "r--", alpha=0.4, label="Magnetization")
    plt.plot(df_adversarial["Ts"], df_adversarial["ordinary_data_avg_pred"], "o-", markersize=4, color='darkgreen', label="adve_ord")
    plt.plot(df_adversarial["Ts"], df_adversarial["shuffled_data_avg_pred"], "*-", color='purple', label="adve_shuff")
    if df_ordinary is not None:
        plt.plot(df_ordinary["Ts"], df_ordinary["shuffled_data_avg_pred"], '.-', label="ord_shuff")
    Ts = scatter_info_dict["Ts"]
    outputs = scatter_info_dict["outputs"]
    plt.scatter(Ts, outputs, alpha=0.002, s=8.0, color='springgreen')
    plt.xlabel("T--->")
    plt.ylabel("Accuracies--->")
    plt.legend()
    plt.grid(True)
    plt.title(f"G={G_label}")
    figure_file_name = f"G_{G_label}_accuracies_vs_T.jpg"
    figure_file_path = os.path.join(save_dir, figure_file_name)
    plt.tight_layout()
    plt.savefig(figure_file_path)
    plt.close()
