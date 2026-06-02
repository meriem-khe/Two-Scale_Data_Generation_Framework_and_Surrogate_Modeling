import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

# --------- Generating correlated noise ----------


def generate_correlated_noise(N, beta=0.0, std=1.0):
    """
    Generate correlated (colored) Gaussian noise of length N.

    Parameters
    ----------
    N : int
        Number of data points.
    beta : float
        Spectral exponent:
          0 = white, 1 = pink, 2 = red.
    std : float
        Standard deviation (amplitude scaling).

    Returns
    -------
    noise : np.ndarray
        Real-valued correlated noise.
    """
    # 1. White noise
    white = np.random.normal(0, 1, N)

    # 2. Fourier transform
    f = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(N)
    freqs[0] = freqs[1]  # avoid division by zero

    # 3. Shape spectrum
    f *= freqs ** (-beta / 2.0)

    # 4. Back to time domain
    colored = np.fft.irfft(f, n=N)
    colored = colored / np.std(colored) * std  # normalize

    return colored


# ----------- Plot the generated noise -------------


def plot_noise(noise, sample_rate=1.0, title="Noise Signal"):
    """
    Plot time series and power spectrum of the generated noise.

    Parameters
    ----------
    noise : np.ndarray
        The noise sequence (1D array).
    sample_rate : float
        Sampling rate (Hz) – 1.0 by default if you have no time unit.
    title : str
        Plot title.
    """
    N = len(noise)
    t = np.arange(N) / sample_rate

    # Compute Power Spectral Density (Welch’s method)
    freqs, psd = welch(noise, fs=sample_rate, nperseg=N // 4)

    # Plot setup
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(title, fontsize=14)

    # --- Time domain ---
    axs[0].plot(t, noise, color="royalblue", linewidth=1)
    axs[0].set_title("Time Series")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(alpha=0.3)

    # --- Frequency domain (Power Spectrum) ---
    axs[1].loglog(freqs[1:], psd[1:], color="darkorange")
    axs[1].set_title("Power Spectral Density (log–log)")
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("Power")
    axs[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
    plt.show()


# ------- Add noise to macro dataset ---------


def add_macro_noise(macro_df, noise_strength=0.05, beta=1.8):
    """
    Adds red noise to macro-scale data (sea surface height).
    """
    N = len(macro_df)
    # Generate red noise
    noise = generate_correlated_noise(
        N, beta=beta, std=macro_df["A_sample"].std() * noise_strength
    )
    plot_noise(noise, title="Correlated noise - macroscopic energy density A (β=1.8)")

    # Apply to amplitude-like quantities
    macro_df["A_sample_noisy"] = macro_df["A_sample"] + noise
    macro_df["A_mean_noisy"] = (
        macro_df["A_mean"] + noise * 0.5
    )  # smaller effect on mean
    macro_df["A_max_noisy"] = macro_df["A_max"] + noise * 0.8
    macro_df["A_min_noisy"] = macro_df["A_min"] + noise * 0.8

    return macro_df


# -------- Adding noise to micro data --------------


def add_micro_noise(micro_df, noise_strength=0.02, beta=1.0):
    """
    Adds pink noise to micro-scale data (wave oscillations).
    """
    N = len(micro_df)
    noise_amp = generate_correlated_noise(
        N, beta=beta, std=micro_df["amp"].std() * noise_strength
    )
    plot_noise(noise_amp, title="Pink noise on micro amplitude (β=1)")
    noise_phi = generate_correlated_noise(
        N, beta=beta, std=micro_df["phi_var"].std() * noise_strength
    )
    plot_noise(noise_phi, title="Pink noise on micro phi_var (β=1)")

    noise_phi_rms = generate_correlated_noise(
        N, beta=beta, std=micro_df["phi_rms"].std() * noise_strength
    )
    plot_noise(noise_phi_rms, title="Correlated noise - microscopic phi_rms (β=1)")

    micro_df["amp_noisy"] = micro_df["amp"] + noise_amp
    micro_df["phi_var_noisy"] = micro_df["phi_var"] + noise_phi
    micro_df["phi_rms_noisy"] = micro_df["phi_rms"] + noise_phi_rms
    micro_df["E_micro_noisy"] = (
        micro_df["E_micro"] + noise_amp * 0.1
    )  # slight energy perturbation

    return micro_df


# ------- Application ----------

# Load data
macro = pd.read_csv("../macro_data.csv")
micro = pd.read_csv("../micro_sliding.csv")

# Add realistic sensor noise
macro_noisy = add_macro_noise(macro)
micro_noisy = add_micro_noise(micro)

# Save results
macro_noisy.to_csv("macro_noisy.csv", index=False)
micro_noisy.to_csv("micro_noisy.csv", index=False)
