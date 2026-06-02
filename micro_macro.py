# -*- coding: utf-8 -*-

# ==========================================================
# Two-scale simulation:
#   Macro: energy advection-dissipation
#   Micro (sliding): wave field centered on a moving point
#   Micro (fixed x3): wave fields at 3 fixed windows
# ==========================================================

# --- Libraries ---
import csv
import os
import time

import imageio.v2 as imageio  # make GIF animations
import matplotlib.pyplot as plt  # plotting
import numpy as np  # math
from fenics import *  # PDE/FEM engine (old FEniCS interface)

# ==========================================================
#  Save the simulation start time
# ==========================================================
t_start = time.time()
# ==========================================================
# 1. Domain parameters
# ==========================================================
Lx = 108_000.0  # macro domain size in x-direction (meters) = 108 km
Ly = 108_000.0  # macro domain size in y-direction = 108 km

Lz_half = 500.0  # micro domain extends [-500, 500] m in both directions
Lz1_min, Lz1_max = -Lz_half, Lz_half
Lz2_min, Lz2_max = -Lz_half, Lz_half

# ==========================================================
# 2. Discretization
# ==========================================================
Nx, Ny = 108, 108  # macro mesh divisions → ~1 km resolution
Nz1, Nz2 = 200, 200  # micro mesh divisions → 5 m resolution

# ==========================================================
# 3. Time parameters
# ==========================================================
T = 8800.0  # total simulation time [s] = 147 minutes = 2 heures 27 minutes
dt_macro = 2.0  # macro timestep [s]
Nt = int(T / dt_macro)  # number of macro steps

# ==========================================================
# 4. Physical parameters
# ==========================================================
Ux, Uy = 10.0, 10.0  # macro advection velocity (m/s)
beta = 1e-4  # damping coefficient for macro energy
c = 10.0  # micro wave speed (m/s)

# Micro wave parameters
lambda_micro = 200.0  # wavelength [m]
kx = 2 * np.pi / lambda_micro
ky = 2 * np.pi / lambda_micro
k_mag = np.sqrt(kx**2 + ky**2)
omega = c * k_mag  # angular frequency of wave

# Sampling point in macro domain to feed micro model (sliding center starts here)
x0, y0 = 20_000.0, 20_000.0

# ==========================================================
# 5. Output folders & utility
# ==========================================================
os.makedirs("frames_macro", exist_ok=True)
os.makedirs("frames_micro_sliding", exist_ok=True)

# Fixed windows folders
for i in range(1, 4):
    os.makedirs(f"frames_micro_fixed_{i}", exist_ok=True)


def make_gif(frame_folder, gif_name, fps=10):
    """Convert a folder of PNG frames into an animated GIF."""
    images = sorted(os.listdir(frame_folder))
    frames = [
        imageio.imread(os.path.join(frame_folder, img))
        for img in images
        if img.endswith(".png")
    ]
    if frames:
        imageio.mimsave(gif_name, frames, fps=fps)


# ==========================================================
# 6. Macro model setup
# ==========================================================
mesh_macro = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny)
V_macro = FunctionSpace(mesh_macro, "P", 1)  # P1 FE space

# Unknowns
A_old = Function(V_macro)  # solution at previous timestep
A_new = Function(V_macro)  # solution at current timestep

# Initial condition: Gaussian bump centered at (20 km, 20 km)
x_c0, y_c0 = 20_000.0, 20_000.0
sigma = 5_000.0
expr_A0 = Expression(
    "exp(-((x[0]-xc)*(x[0]-xc) + (x[1]-yc)*(x[1]-yc)) / (2*sigma*sigma))",
    xc=x_c0,
    yc=y_c0,
    sigma=sigma,
    degree=2,
)
A_old.interpolate(expr_A0)

# Variational form for implicit Euler step
v_macro = TestFunction(V_macro)
A_trial = TrialFunction(V_macro)
U_vec = Constant((Ux, Uy))  # velocity vector

# Left-hand side (implicit update)
a_macro = (
    (A_trial * v_macro) * dx
    + dt_macro * dot(U_vec, grad(A_trial)) * v_macro * dx
    + dt_macro * beta * A_trial * v_macro * dx
)
# Right-hand side (explicit old solution)
L_macro = (A_old * v_macro) * dx


# ==========================================================
# 7. Micro model: helpers
# ==========================================================
def create_micro_mesh(center_x, center_y):
    """Create a micro mesh centered at (center_x, center_y)."""
    return RectangleMesh(
        Point(center_x - Lz_half, center_y - Lz_half),
        Point(center_x + Lz_half, center_y + Lz_half),
        Nz1,
        Nz2,
    )


# CFL condition (based on fixed 5m resolution)
h = (2 * Lz_half) / Nz1
print("h=", h)
safety = 0.5
dt_cfl = safety * h / (c * np.sqrt(2.0))  # 2D CFL condition
print("dt_cfl=", dt_cfl)
substeps_micro = int(np.ceil(dt_macro / dt_cfl))
print("substeps_micro=", substeps_micro)
dt_micro = dt_macro / substeps_micro
print("dt_micro=", dt_micro)
print(
    f"[MICRO-CFL] h={h:.3f} m, c={c:.2f} m/s, "
    f"dt_micro={dt_micro:.5f} s, substeps_micro={substeps_micro}, "
    f"CFL={c * dt_micro / h:.3f}"
)

# ==========================================================
# 8. Micro (sliding) initial setup
# ==========================================================
mesh_micro_sliding = create_micro_mesh(x0, y0)
V_micro_sliding = FunctionSpace(mesh_micro_sliding, "P", 1)

phi_s = Function(V_micro_sliding)  # current field
phi_s_old = Function(V_micro_sliding)  # previous field
phi_s_new = Function(V_micro_sliding)  # next field

phi_s_trial = TrialFunction(V_micro_sliding)
v_micro_s = TestFunction(V_micro_sliding)
a_wave_s = phi_s_trial * v_micro_s * dx

# ==========================================================
# 9. Micro (fixed x3) setup
#    These are "before", "during", "after" relative to when the macro packet reaches them in time.
# ==========================================================
fixed_centers = [
    (20_000.0, 20_000.0),  # fixed window 1
    (60_000.0, 60_000.0),  # fixed window 2
    (100_000.0, 100_000.0),  # fixed window 3
]

fixed_centers = [((cx % Lx), (cy % Ly)) for (cx, cy) in fixed_centers]

# Allocate per-window structures
fixed_windows = []
for i, (cx, cy) in enumerate(fixed_centers, start=1):
    mesh = create_micro_mesh(cx, cy)
    V = FunctionSpace(mesh, "P", 1)
    phi = Function(V)
    phi_old = Function(V)
    phi_new = Function(V)
    v_micro = TestFunction(V)
    phi_trial = TrialFunction(V)
    a_wave = phi_trial * v_micro * dx

    # Writers and frame folders
    xdmf = XDMFFile(f"micro_fixed_{i}_fields.xdmf")
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True

    fixed_windows.append(
        {
            "i": i,
            "center": (cx, cy),
            "mesh": mesh,
            "V": V,
            "phi": phi,
            "phi_old": phi_old,
            "phi_new": phi_new,
            "v_micro": v_micro,
            "a_wave": a_wave,
            "xdmf": xdmf,
            "frames_dir": f"frames_micro_fixed_{i}",
            "csv_path": f"micro_fixed_{i}.csv",
        }
    )

# ==========================================================
# 10. Files for macro & sliding micro
# ==========================================================
macro_xdmf = XDMFFile("macro_fields.xdmf")
micro_sliding_xdmf = XDMFFile("micro_sliding_fields.xdmf")
macro_xdmf.parameters["flush_output"] = True
micro_sliding_xdmf.parameters["flush_output"] = True
macro_xdmf.parameters["functions_share_mesh"] = True
micro_sliding_xdmf.parameters["functions_share_mesh"] = True

# ==========================================================
# 11. Time loop (macro + all micro systems)
# ==========================================================
with (
    open("macro_data.csv", "w", newline="") as f_macro,
    open("micro_sliding.csv", "w", newline="") as f_micro_s,
    open(fixed_windows[0]["csv_path"], "w", newline="") as f_mf1,
    open(fixed_windows[1]["csv_path"], "w", newline="") as f_mf2,
    open(fixed_windows[2]["csv_path"], "w", newline="") as f_mf3,
    open("timing_log.csv", "w", newline="") as f_timing,
):
    writer_macro = csv.writer(f_macro)
    writer_micro_s = csv.writer(f_micro_s)
    writer_mf = [csv.writer(f_mf1), csv.writer(f_mf2), csv.writer(f_mf3)]
    writer_timing = csv.writer(f_timing)
    # CSV headers
    writer_macro.writerow(
        [
            "t[s]",
            "x_t",
            "y_t",
            "A_sample",
            "A_mean",
            "A_var",
            "A_max",
            "A_min",
            "macro_step_time[s]",
        ]
    )
    writer_micro_s.writerow(
        [
            "t[s]",
            "center_x",
            "center_y",
            "A_sample",
            "dA_dx",
            "dA_dy",
            "amp",
            "phi_var",
            "phi_max",
            "phi_min",
            "phi_rms",
            "E_micro",
            "sigma_phi",
            "micro_step_time[s]",
        ]
    )

    for i in range(3):
        writer_mf[i].writerow(
            [
                "t[s]",
                "center_x",
                "center_y",
                "amp",
                "phi_var",
                "phi_max",
                "phi_min",
                "phi_rms",
            ]
        )
    writer_timing.writerow(
        [
            "macro_step[s]",
            "micro_total_supsteps_time[s]",
            "fixed_micro_total_substeps_1[s]",
            "fixed_micro_total_substeps_2[s]",
            "fixed_micro_total_substeps_3[s]",
        ]
    )

    for n in range(Nt):
        # --- MACRO step ---
        t0_macro = time.time()
        t_macro = (n + 1) * dt_macro
        solve(a_macro == L_macro, A_new)
        t1_macro = time.time()
        macro_step_time = t1_macro - t0_macro
        print(f"Macro step {n}/{Nt}: {macro_step_time:.3f} s")

        # Macro stats
        A_array = A_new.vector().get_local()
        A_mean = A_array.mean()
        A_var = A_array.var()
        A_max = A_array.max()
        A_min = A_array.min()

        # Find sliding center as location of maximal nodal A (ensures window follows Gaussian peak)
        coords = mesh_macro.coordinates()
        values = A_new.compute_vertex_values(mesh_macro)
        imax = np.argmax(values)
        x_t = coords[imax, 0]
        y_t = coords[imax, 1]

        # Compute the sampled scalar explicitly at that point (interpolated)
        A_sample_sliding = float(A_new(Point(x_t, y_t)))
        # Macro gradient at sliding center (for coupling)
        Vg = VectorFunctionSpace(mesh_macro, "P", 1)
        gradA = project(grad(A_new), Vg)

        dA_dx, dA_dy = gradA(Point(x_t, y_t))

        # Save macro plot every 1 minutes
        if n % 30 == 0:
            # Write to macro CSV
            writer_macro.writerow(
                [
                    t_macro,
                    x_t,
                    y_t,
                    A_sample_sliding,
                    A_mean,
                    A_var,
                    A_max,
                    A_min,
                    macro_step_time,
                ]
            )
            # Save macro field
            macro_xdmf.write(A_new, t_macro)

            # Save macro plot
            coords = mesh_macro.coordinates()
            values = A_new.compute_vertex_values(mesh_macro)
            idx = np.lexsort((coords[:, 1], coords[:, 0]))
            x = coords[:, 0][idx]
            y = coords[:, 1][idx]
            z = values[idx]
            x_unique = np.unique(x)
            y_unique = np.unique(y)
            X, Y = np.meshgrid(x_unique, y_unique)
            Z = z.reshape((len(y_unique), len(x_unique)))

            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z, cmap="plasma", edgecolor="k", linewidth=0.1)
            ax.set_zlim(0.0, 1.0)
            ax.set_zticks(np.arange(0.0, 1.02, 0.2))
            ax.set_title(f"Macro A at t={t_macro:.1f} s")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_zlabel("A")
            plt.tight_layout()
            plt.savefig(f"frames_macro/macro_{n:04d}.png")
            plt.close()

            # ==================================================
            # --- MICRO (sliding window) ---
            # Rebuild sliding mesh/function space centered at (x_t, y_t)
            mesh_micro_sliding = create_micro_mesh(x_t, y_t)
            V_micro_sliding = FunctionSpace(mesh_micro_sliding, "P", 1)

            phi_s = Function(V_micro_sliding)
            phi_s_old = Function(V_micro_sliding)
            phi_s_new = Function(V_micro_sliding)
            phi_s_trial = TrialFunction(V_micro_sliding)
            v_micro_s = TestFunction(V_micro_sliding)
            a_wave_s = phi_s_trial * v_micro_s * dx

            # Driving amplitude from macro at sliding center
            amp_s = np.sqrt(abs(A_sample_sliding))

            # Local harmonic wave initial condition
            expr_phi0_s = Expression(
                "amp * cos(omega*t - (kx*(x[0]-cx) + ky*(x[1]-cy)))",
                amp=amp_s,
                t=t_macro,
                omega=omega,
                kx=kx,
                ky=ky,
                cx=x_t,
                cy=y_t,
                degree=2,
            )
            expr_phi_t0_s = Expression(
                "-amp * omega * sin(omega*t - (kx*(x[0]-cx) + ky*(x[1]-cy)))",
                amp=amp_s,
                t=t_macro,
                omega=omega,
                kx=kx,
                ky=ky,
                cx=x_t,
                cy=y_t,
                degree=2,
            )

            phi_s_old.interpolate(expr_phi0_s)
            phi_t0_s = interpolate(expr_phi_t0_s, V_micro_sliding)
            phi_s.vector()[:] = phi_s_old.vector() + dt_micro * phi_t0_s.vector()

            # Subcycling for sliding
            total0_micro_substeps = time.time()
            for k in range(substeps_micro):
                t0_micro = time.time()
                t_micro = n * dt_macro + (k + 1) * dt_micro
                print("k=", k, "t_micro", t_micro)

                L_wave_s = (
                    2 * phi_s - phi_s_old
                ) * v_micro_s * dx - dt_micro**2 * c**2 * dot(
                    grad(phi_s), grad(v_micro_s)
                ) * dx
                solve(a_wave_s == L_wave_s, phi_s_new)

                # save micro time
                t1_micro = time.time()
                dt_micro_step = t1_micro - t0_micro

                print("k=", k, "n=", n, "t-micro=", t_micro, "time=", dt_micro_step)

                # Micro stats
                phi_array = phi_s_new.vector().get_local()

                # Basic stats
                phi_var = phi_array.var()
                phi_max = phi_array.max()
                phi_min = phi_array.min()
                phi_rms = np.sqrt(np.mean(phi_array**2))

                # --- MICRO ENERGY ---
                # Gradient energy
                Vg_micro = VectorFunctionSpace(mesh_micro_sliding, "P", 1)
                grad_phi = project(grad(phi_s_new), Vg_micro)
                grad_phi_array = grad_phi.vector().get_local()

                E_micro = np.mean(phi_array**2) + c**2 * np.mean(grad_phi_array**2)

                # --- MICRO ENVELOPE WIDTH ---
                coords_m = mesh_micro_sliding.coordinates()
                r2 = (coords_m[:, 0] - x_t) ** 2 + (coords_m[:, 1] - y_t) ** 2

                # Avoid division by zero
                if np.sum(phi_array**2) > 1e-14:
                    sigma_phi = np.sum(r2 * phi_array**2) / np.sum(phi_array**2)
                else:
                    sigma_phi = 0.0

                # Write to micro sliding CSV
                writer_micro_s.writerow(
                    [
                        t_micro,
                        x_t,
                        y_t,
                        A_sample_sliding,
                        dA_dx,
                        dA_dy,
                        amp_s,
                        phi_var,
                        phi_max,
                        phi_min,
                        phi_rms,
                        E_micro,
                        sigma_phi,
                        dt_micro_step,
                    ]
                )
                # Save sliding field
                micro_sliding_xdmf.write(phi_s_new, t_micro)

                # Save micro plots
                coords_m = mesh_micro_sliding.coordinates()
                values_m = phi_s_new.compute_vertex_values(mesh_micro_sliding)

                idxm = np.lexsort((coords_m[:, 1], coords_m[:, 0]))
                xm = coords_m[:, 0][idxm]
                ym = coords_m[:, 1][idxm]
                zm = values_m[idxm]
                xu = np.unique(xm)
                yu = np.unique(ym)
                Xm, Ym = np.meshgrid(xu, yu)
                Zm = zm.reshape((len(yu), len(xu)))

                # Crop inner region (optional)
                margin = 0.0
                mask = (
                    (Xm > x_t - Lz_half + margin)
                    & (Xm < x_t + Lz_half - margin)
                    & (Ym > y_t - Lz_half + margin)
                    & (Ym < y_t + Lz_half - margin)
                )
                Xm_crop, Ym_crop, Zm_crop = (
                    Xm[mask].reshape(-1),
                    Ym[mask].reshape(-1),
                    Zm[mask].reshape(-1),
                )
                x_uc = np.unique(Xm_crop)
                y_uc = np.unique(Ym_crop)
                Xc, Yc = np.meshgrid(x_uc, y_uc)
                Zc = Zm_crop.reshape((len(y_uc), len(x_uc)))

                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_surface(
                    Xc, Yc, Zc, cmap="viridis", edgecolor="k", linewidth=0.1
                )
                ax.set_zlim(-1.0, 1.0)
                ax.set_zticks(np.arange(-1.0, 1.02, 0.2))
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")
                ax.set_zlabel("φ(x,y)")
                ax.set_title(
                    f"Micro (sliding) φ at t={t_micro:.1f} s (center=({x_t:.0f},{y_t:.0f}))"
                )
                plt.tight_layout()
                plt.savefig(f"frames_micro_sliding/micro_{n:04d}_{k:04d}.png")
                plt.close()

                # Shift time levels (sliding)
                phi_s_old.assign(phi_s)
                phi_s.assign(phi_s_new)

            # save total micro substeps time
            total1_micro_substeps = time.time()
            total_micro_step = total1_micro_substeps - total0_micro_substeps

            # ==================================================
            # --- MICRO (fixed x3) ---
            for j, win in enumerate(fixed_windows):
                cx, cy = win["center"]

                # Driving amplitude from macro field at the fixed center
                A_sample_fixed = float(A_new(Point(cx, cy)))

                amp = np.sqrt(abs(A_sample_fixed))

                # Initialize (re-initialize each macro step, analogous to sliding init)
                expr_phi0 = Expression(
                    "amp * cos(omega*t - (kx*(x[0]-cx) + ky*(x[1]-cy)))",
                    amp=amp,
                    t=t_macro,
                    omega=omega,
                    kx=kx,
                    ky=ky,
                    cx=cx,
                    cy=cy,
                    degree=2,
                )
                expr_phi_t0 = Expression(
                    "-amp * omega * sin(omega*t - (kx*(x[0]-cx) + ky*(x[1]-cy)))",
                    amp=amp,
                    t=t_macro,
                    omega=omega,
                    kx=kx,
                    ky=ky,
                    cx=cx,
                    cy=cy,
                    degree=2,
                )

                win["phi_old"].interpolate(expr_phi0)
                phi_t0 = interpolate(expr_phi_t0, win["V"])
                win["phi"].vector()[:] = (
                    win["phi_old"].vector() + dt_micro * phi_t0.vector()
                )

                # Saved fixed window time
                total_fixed0 = time.time()

                # Subcycling for fixed window j
                for k in range(substeps_micro):
                    t_micro = n * dt_macro + (k + 1) * dt_micro

                    L_wave = (2 * win["phi"] - win["phi_old"]) * win[
                        "v_micro"
                    ] * dx - dt_micro**2 * c**2 * dot(
                        grad(win["phi"]), grad(win["v_micro"])
                    ) * dx
                    solve(win["a_wave"] == L_wave, win["phi_new"])

                    # Stats
                    phi_array = win["phi_new"].vector().get_local()
                    phi_var = phi_array.var()
                    phi_max = phi_array.max()
                    phi_min = phi_array.min()
                    phi_rms_f = np.sqrt(np.mean(phi_array**2))

                    # Write CSV
                    writer_mf[j].writerow(
                        [
                            t_micro,
                            cx,
                            cy,
                            amp,
                            phi_var,
                            phi_max,
                            phi_min,
                            phi_rms_f,
                        ]
                    )

                    # Save field
                    win["xdmf"].write(win["phi_new"], t_micro)

                    # Save fixed micro plots
                    # coords_m = win["mesh"].coordinates()
                    # values_m = win["phi_new"].compute_vertex_values(win["mesh"])

                    # idxm = np.lexsort((coords_m[:, 1], coords_m[:, 0]))
                    # xm = coords_m[:, 0][idxm]
                    # ym = coords_m[:, 1][idxm]
                    # zm = values_m[idxm]
                    # xu = np.unique(xm)
                    # yu = np.unique(ym)
                    # Xm, Ym = np.meshgrid(xu, yu)
                    # Zm = zm.reshape((len(yu), len(xu)))

                    # # Crop inner region (optional)
                    # margin = 0.0
                    # mask = (
                    #     (Xm > cx - Lz_half + margin)
                    #     & (Xm < cx + Lz_half - margin)
                    #     & (Ym > cy - Lz_half + margin)
                    #     & (Ym < cy + Lz_half - margin)
                    # )
                    # Xm_crop, Ym_crop, Zm_crop = (
                    #     Xm[mask].reshape(-1),
                    #     Ym[mask].reshape(-1),
                    #     Zm[mask].reshape(-1),
                    # )
                    # x_uc = np.unique(Xm_crop)
                    # y_uc = np.unique(Ym_crop)
                    # Xc, Yc = np.meshgrid(x_uc, y_uc)
                    # Zc = Zm_crop.reshape((len(y_uc), len(x_uc)))

                    # fig = plt.figure(figsize=(7, 6))
                    # ax = fig.add_subplot(111, projection="3d")
                    # ax.plot_surface(
                    #     Xc, Yc, Zc, cmap="viridis", edgecolor="k", linewidth=0.1
                    # )
                    # ax.set_xlabel("x [m]")
                    # ax.set_ylabel("y [m]")
                    # ax.set_zlabel("φ(x,y)")
                    # ax.set_zlim(-1.0, 1.0)
                    # ax.set_zticks(np.arange(-1.0, 1.02, 0.2))
                    # ax.set_title(
                    #     f"Micro (fixed {win['i']}) φ at t={t_micro:.1f} s (center=({cx:.0f},{cy:.0f}))"
                    # )
                    # plt.tight_layout()
                    # plt.savefig(f"{win['frames_dir']}/micro_{n:04d}.png")
                    # plt.close()

                    # Shift time levels (fixed window)
                    win["phi_old"].assign(win["phi"])
                    win["phi"].assign(win["phi_new"])

                # save fixed window total time
                total_fixed1 = time.time()
                if j == 0:
                    total_fixed_substeps_1 = total_fixed1 - total_fixed0
                if j == 1:
                    total_fixed_substeps_2 = total_fixed1 - total_fixed0
                if j == 2:
                    total_fixed_substeps_3 = total_fixed1 - total_fixed0

        writer_timing.writerow(
            [
                macro_step_time,
                total_micro_step,
                total_fixed_substeps_1,
                total_fixed_substeps_2,
                total_fixed_substeps_3,
            ]
        )
        # Prepare macro for next step
        A_old.assign(A_new)

# ==========================================================
# 12. Build GIF animations
# ==========================================================
make_gif("frames_macro", "macro.gif")
make_gif("frames_micro_sliding", "micro_sliding.gif")
for i in range(1, 4):
    make_gif(f"frames_micro_fixed_{i}", f"micro_fixed_{i}.gif")
t_end = time.time()
print(f"Total wall-clock time = {t_end - t_start:.2f} seconds")
