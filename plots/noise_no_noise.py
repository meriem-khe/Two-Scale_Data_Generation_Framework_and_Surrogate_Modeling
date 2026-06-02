import matplotlib.pyplot as plt
import pandas as pd


def plot_macro_comparison(macro_path, macro_noisy_path):
    """
    Compare A_sample (clean vs noisy) from macro data over time.
    """
    # Load CSVs
    df_clean = pd.read_csv(macro_path)
    df_noisy = pd.read_csv(macro_noisy_path)

    # Ensure both have the same time length
    assert len(df_clean) == len(df_noisy), "Data length mismatch!"

    t = df_clean["t[s]"]
    A_clean = df_clean["A_sample"]
    A_noisy = df_noisy["A_sample_noisy"]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, A_clean, label="Clean A_sample", color="royalblue", linewidth=1.5)
    plt.plot(t, A_noisy, label="Noisy A_sample", color="darkorange", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude A")
    plt.title("Macro Data: Clean vs Noisy A_sample")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Macro-Data-Clean-vs-Noisy-A_sample.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_micro_comparison(micro_path, micro_noisy_path):
    """
    Compare phi_rms (clean vs noisy) from micro data over time.
    """
    df_clean = pd.read_csv(micro_path)
    df_noisy = pd.read_csv(micro_noisy_path)

    assert len(df_clean) == len(df_noisy), "Data length mismatch!"

    t = df_clean["t[s]"]
    phi_clean = df_clean["phi_rms"]
    phi_noisy = df_noisy["phi_rms_noisy"]

    plt.figure(figsize=(10, 5))
    plt.plot(t, phi_clean, label="Clean φ_rms", color="seagreen", linewidth=1.5)
    plt.plot(t, phi_noisy, label="Noisy φ_rms", color="crimson", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Phase Variance φ_rms")
    plt.title("Micro Data: Clean vs Noisy φ_rms")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Micro-Data-Clean-vs-Noisy-φ_rms.png", dpi=300, bbox_inches="tight")
    plt.show()


plot_macro_comparison("../macro_data.csv", "../noise/macro_noisy.csv")
plot_micro_comparison("../micro_sliding.csv", "../noise/micro_noisy.csv")
