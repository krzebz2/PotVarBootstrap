import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys

def V_cornell(r, V0, alpha, sigma):
    return V0 + alpha / r + sigma * r

# Switches for alpha and linewidth mode
USE_ALPHA_WITH_MINIMUM = True
USE_WEIGHTED_LINEWIDTH = False

# used if USE_WEIGHTED_ALPHA_LINEAR = True
MIN_ALPHA = 0.1
MAX_ALPHA = 0.3

# used if USE_WEIGHTED_ALPHA_LINEAR = False
GAMMA = 1.5

# used if USE_WEIGHTED_LINEWIDTH = False
LINEWIDTH = 0.05

global_vmin = float("inf")
global_vmax = float("-inf")
# fraction of total weight to include in y-axis scaling
WEIGHT_COVERAGE = 0.7
# track min/max per curve and its weight for weighted y-limit
curve_stats = []  # list of (weight, vmin_curve, vmax_curve)


# === Ensemble wählen ===
if len(sys.argv) == 4:
    ENSEMBLE = sys.argv[1]
    smearing = int(sys.argv[2])
    rmax = int(sys.argv[3])
else:
    with open("input_ensemble.txt") as f:
        ENSEMBLE = f.readline().strip()
        smearing = int(f.readline())
        rmax = int(f.readline())

ensemble = f"{ENSEMBLE}_sm{smearing}_rmax{rmax:02d}"
base_path = os.path.expanduser(f"~/data/analysis/{ensemble}/results_{ensemble}")
mean_dir = os.path.join(base_path, "mean_potential")
bootstrap_dir = os.path.join(base_path, "bootstrap_parameters")

plot_dir = f"{base_path}/plots/cornell_fit_all"
os.makedirs(plot_dir, exist_ok=True)

# === Messpunkte laden ===
x_data, y_data, y_errs = [], [], []
for f in sorted(glob.glob(os.path.join(mean_dir, "mean_r*.dat"))):
    r = int(os.path.basename(f).split("_r")[1].split(".")[0])
    with open(f) as file:
        lines = file.readlines()
    if len(lines) < 2:
        continue
    vals = lines[1].strip().split()
    V_mean = float(vals[0])
    V_err  = float(vals[1])
    x_data.append(r)
    y_data.append(V_mean)
    y_errs.append(V_err)

r_vals = np.linspace(min(x_data), max(x_data), 300)

# === Alle bootstrap_ira_irb.dat Dateien durchgehen ===
for param_file in sorted(glob.glob(os.path.join(bootstrap_dir, "bootstrap_*.dat"))):
    print(f"Contetent of file {param_file}:")
    with open(param_file) as f:
        for i, line in enumerate(f):
            print(f"    {i:02d}: {line.strip()}")
            if i > 10:
                print("    ...")
                break

    with open(param_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 19:
                print(f"Skipping line in {param_file}: only {len(parts)} columns → {parts}")
                continue
            
            ira = int(parts[0])
            irb = int(parts[1])
            V0_val = float(parts[2]);   V0_err_val = float(parts[3])
            alpha_val = float(parts[4]); alpha_err_val = float(parts[5])
            sigma_val = float(parts[6]); sigma_err_val = float(parts[7])
            weight = float(parts[18])

            print(f"  Parsed fit: ira={ira}, irb={irb}, V0={V0_val}, alpha={alpha_val}, sigma={sigma_val}, weight={weight}")

            if USE_ALPHA_WITH_MINIMUM:
                alpha_plot = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * weight ** GAMMA
            else:
                alpha_plot = MAX_ALPHA * weight ** GAMMA


            # === Transparenz aus Gewicht (z.B. linear, max=0.3) ===
            # alpha_plot = min(0.3, weight)

            # === Bootstrapsamples laden ===
            bootstrap_sample_file = os.path.join(
                bootstrap_dir, f"bootstrap_{ira}_{irb}",
                f"bootstrapsamples_{ira}_{irb}.dat"
            )
            print(f"Looking for bootstrap sample file: {bootstrap_sample_file}")

            if not os.path.exists(bootstrap_sample_file):
                print(f"File not found: {bootstrap_sample_file}")
                continue

            valid_sample_lines = 0
            with open(bootstrap_sample_file) as fs:
                next(fs)  # Header
                for s_line in fs:
                    p = s_line.strip().split()
                    if len(p) >= 4:
                        valid_sample_lines += 1
                        V0_b, alpha_b, sigma_b = map(float, p[1:4])
                        V_b = V_cornell(r_vals, V0_b, alpha_b, sigma_b)
                        # update global min/max as before
                        global_vmin = min(global_vmin, np.min(V_b))
                        global_vmax = max(global_vmax, np.max(V_b))

                        # store per-curve stats for weighted limit
                        curve_stats.append((weight, np.min(V_b), np.max(V_b)))
                        if USE_WEIGHTED_LINEWIDTH:
                            linewidth = 0.5 + 1.0 * weight
                        else:
                            linewidth = LINEWIDTH

                        plt.plot(r_vals, V_b, color='blue', alpha=alpha_plot, linewidth=linewidth)

            print(f"  {valid_sample_lines} gültige Sample-Zeilen geplottet.")

# === Plot Messpunkte ===
plt.errorbar(
    x_data, y_data, yerr=y_errs,
    fmt='o', color='black', markerfacecolor='none',
    markersize=6, elinewidth=1.2, capsize=0,
    label="mean $V(r)$", zorder=3
)
plt.xlabel(r"$r$")
plt.ylabel(r"$V(r)$")
plt.grid(True)
plt.legend()
plt.tight_layout()

output_path = os.path.join(plot_dir, "cornell_fit_all_ranges.png")

curve_stats_sorted = sorted(curve_stats, key=lambda x: x[0], reverse=True)
total_weight = sum(w for w, vmin_c, vmax_c in curve_stats_sorted)

cum_weight = 0.0
vmins_selected = []
vmaxs_selected = []

for w, vmin_c, vmax_c in curve_stats_sorted:
    cum_weight += w
    vmins_selected.append(vmin_c)
    vmaxs_selected.append(vmax_c)
    if cum_weight >= WEIGHT_COVERAGE * total_weight:  # stop at e.g. 95% of total weight
        break

# compute y-axis range from selected (weighted) curves
min_y_weighted = min(vmins_selected)
max_y_weighted = max(vmaxs_selected)
# ensure y-axis includes full error bars
ymin_error = min(y - e for y, e in zip(y_data, y_errs))
ymax_error = max(y + e for y, e in zip(y_data, y_errs))
error_margin = 0.05 * (ymax_error - ymin_error)
ymin_error -= error_margin
ymax_error += error_margin

# apply y-axis limit
min_y = min(min_y_weighted, ymin_error)
max_y = max(max_y_weighted, ymax_error)
plt.ylim(min_y, max_y)


plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print("\n--- used parameters ---")
print(f"USE_WEIGHTED_ALPHA_LINEAR = {USE_WEIGHTED_ALPHA_LINEAR}")
if USE_WEIGHTED_ALPHA_LINEAR:
    print(f"  MIN_ALPHA = {MIN_ALPHA}")
    print(f"  MAX_ALPHA = {MAX_ALPHA}")
else:
    print(f"  GAMMA = {GAMMA}")

print(f"USE_WEIGHTED_LINEWIDTH = {USE_WEIGHTED_LINEWIDTH}")
if not USE_WEIGHTED_LINEWIDTH:
    print(f"  LINEWIDTH = {LINEWIDTH}")

print("\n--- y-axis range information ---")
print(f"Global y-axis range (all curves):       [{global_vmin:.2f}, {global_vmax:.2f}]")
print(f"Weighted {int(WEIGHT_COVERAGE*100)}% range (curves only): [{min_y_weighted:.2f}, {max_y_weighted:.2f}]")
print(f"Final y-axis range (including error bars): [{min_y:.2f}, {max_y:.2f}]")






