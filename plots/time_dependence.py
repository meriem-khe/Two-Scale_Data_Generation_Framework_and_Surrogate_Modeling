import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------
# Load CSV data
# -------------------------
macro = pd.read_csv("../macro_data.csv")
micro = pd.read_csv("../micro_sliding.csv")

# t_macro = macro["t[s]"]
# A_macro = macro["A_sample"]
x_center = micro["center_x"].values
y_center = micro["center_y"].values
t_micro = micro["t[s]"]
phi_micro = micro["phi_rms"]

# Times to highlight
t_marks = [2.0, 3002.0, 8762.0]
# # -------------------------
# # Plot
# # -------------------------
fig, ax1 = plt.subplots(figsize=(12, 7))

# ax1.plot(t_macro, A_macro, label="Macro A(t)", linewidth=2, color="C2")
# ax1.set_xlabel("Time [s]")
# ax1.set_ylabel("Macro amplitude A")
# ax1.grid(True)

ax1.plot(t_micro, phi_micro, label="Micro φ(t) (RMS)", linewidth=2, color="C1")
ax1.set_ylabel("Micro amplitude φ (RMS)")
ax1.set_xlabel("Time [s]")
plt.title("Time evolution of micro amplitudes")


# -------------------------
# Mark points + annotation boxes
# -------------------------
for t0 in t_marks:
    idx = np.argmin(np.abs(t_micro - t0))

    # Point on curve
    ax1.plot(t_micro[idx], phi_micro[idx], "o", markersize=4, color="black", label="")

    # Vertical line
    ax1.axvline(t_micro[idx], linestyle=":", alpha=0.5, label="")

    # Text box content
    label = (
        f"t = {t_micro[idx]:.0f} s\n"
        f"x = {x_center[idx] / 1000:.1f} km\n"
        f"y = {y_center[idx] / 1000:.1f} km"
    )

    # Annotation box
    ax1.annotate(
        label,
        xy=(t_micro[idx], phi_micro[idx]),
        xytext=(10, -25),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
        fontsize=8,
    )

# Legend
# lines = ax1.get_lines()  # + ax2.get_lines()
# labels = [line.get_label() for line in lines]
ax1.legend([ax1.get_lines()[0]], ["Micro φ(t) (RMS)"], loc="upper right")

plt.tight_layout()

# -------------------------
# SAVE FIGURE
# -------------------------
plt.savefig("micro_time_dependence.png", dpi=300, bbox_inches="tight")

plt.show()

#------------ Macroscopic part ----------

# t_macro = macro["t[s]"].values
# A_macro = macro["A_sample"].values

# t_micro = micro["t[s]"].values
# phi_micro = micro["phi_rms"].values

# # Times to highlight
# t_marks = [2.0, 3002.0, 8762.0]

# # -------------------------
# # Plot
# # -------------------------
# fig, ax1 = plt.subplots(figsize=(8, 5))

# # Macro curve
# ax1.plot(t_macro, A_macro, label="Macro A(t)", linewidth=2)
# ax1.set_xlabel("Temps [s]")
# ax1.set_ylabel("Amplitude macro A")
# ax1.grid(True)

# # Micro curve (second axis)
# ax2 = ax1.twinx()
# ax2.plot(t_micro, phi_micro, "--", label="Micro φ(t) (RMS)", linewidth=2)
# ax2.set_ylabel("Amplitude micro φ (RMS)")

# # -------------------------
# # Mark specific times
# # -------------------------
# for t0 in t_marks:
#     # find nearest macro index
#     idx = np.argmin(np.abs(t_macro - t0))
#     ax1.plot(
#         t_macro[idx], A_macro[idx], "o", markersize=8, label=f"A(t={t_macro[idx]:.0f}s)"
#     )
#     ax1.axvline(t_macro[idx], linestyle=":", alpha=0.5)

# # -------------------------
# # Legend and title
# # -------------------------
# lines = ax1.get_lines() + ax2.get_lines()
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc="upper right")

# plt.title("Évolution temporelle macro–micro\n(points marqués à t = 2, 3002, 8762 s)")
# plt.tight_layout()

# # -------------------------
# # Save figure
# # -------------------------
# plt.savefig("macro_micro_with_marked_times.png", dpi=300, bbox_inches="tight")
# plt.show()
