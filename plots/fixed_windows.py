import matplotlib.pyplot as plt
import pandas as pd

# df = pd.read_csv("../micro_fixed_1.csv")

# plt.figure(figsize=(7, 4))
# plt.plot(df["t[s]"], df["phi_rms"], label="φ RMS")
# plt.xlabel("Time [s]")
# plt.ylabel("φ RMS")
# plt.title("Fixed window 1: micro response")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("fixed1_phi_rms_vs_time.png", dpi=300)
# plt.show()

# ---------- Comparaison ---------------

# plt.figure(figsize=(7, 4))

# for i in [1, 2, 3]:
#     df = pd.read_csv(f"../micro_fixed_{i}.csv")
#     plt.plot(df["t[s]"], df["phi_rms"], label=f"Fixed window {i}")

# plt.xlabel("Time [s]")
# plt.ylabel("φ RMS")
# plt.title("Comparison of fixed micro windows")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("fixed_windows_comparison.png", dpi=300)
# plt.show()

# ---------- Correlation --------------

macro = pd.read_csv("../macro_data.csv")
fixed = pd.read_csv("../micro_fixed_2.csv")

plt.figure(figsize=(5, 5))
plt.scatter(fixed["amp"], fixed["phi_rms"], s=10, alpha=0.6)
plt.xlabel("Macro-derived amplitude")
plt.ylabel("Micro φ RMS")
plt.title("Macro–micro coupling (fixed window 2)")
plt.grid(True)
plt.tight_layout()
plt.savefig("macro_micro_correlation_fixed2.png", dpi=300)
plt.show()
