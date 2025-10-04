import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys

# Switches for alpha and linewidth mode
USE_ALPHA_WITH_MINIMUM = True
USE_WEIGHTED_LINEWIDTH = False

GAMMA = 1.5

# fraction of total weight to include in y-axis scaling
WEIGHT_COVERAGE = 0.7

# === Einstellungen für optionalen gefilterten Plot ===
USE_FILTER_PLOT = False

FILTER_ITA_MODE = "eq"   # "eq", "le", "ge" oder None
FILTER_ITA_VAL  = 6

FILTER_ITB_MODE = "eq"   # "eq", "le", "ge" oder None
FILTER_ITB_VAL  = 9

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

# === Cornell-Fit definieren ===
def V_cornell(r, V0, alpha, sigma):
    return V0 + alpha / r + sigma * r


def plot_cornell_fits(
    r_vals, x_data, y_data, y_errs,
    bootstrap_dir,
    output_path,
    filter_ita_mode=None, filter_ita_val=None,
    filter_itb_mode=None, filter_itb_val=None,
    color="blue", label=None
):  
    label_plotted = False
    # track min/max per curve and its weight for weighted y-limit
    # list of (weight, vmin_curve, vmax_curve)
    curve_stats = []
    global_vmin = float("inf")
    global_vmax = float("-inf")

    weights = []
    bootstrap_files = sorted(glob.glob(os.path.join(bootstrap_dir, "bootstrap_*.dat")))

    for f in bootstrap_files:

        with open(f) as file:
            for line in file:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) < 19: continue

                weight = float(parts[18])
                V0_val = float(parts[2])
                alpha_val = float(parts[4])
                sigma_val = float(parts[6])

                # Nur 1 Sample pro Datei für y-Achsenbereich reicht
                V_b = V_cornell(r_vals, V0_val, alpha_val, sigma_val)
                curve_stats.append((weight, np.min(V_b), np.max(V_b)))
                global_vmin = min(global_vmin, np.min(V_b))
                global_vmax = max(global_vmax, np.max(V_b))

    # === 1. Filterdateien nach ita/itb ===
    selected_files = []
    for f in bootstrap_files:
        ira = int(f.split("_")[-2])
        irb = int(f.split("_")[-1].split(".")[0])

        if filter_ita_mode == "eq" and ira != filter_ita_val:
            continue
        if filter_ita_mode == "le" and ira > filter_ita_val:
            continue
        if filter_ita_mode == "ge" and ira < filter_ita_val:
            continue
        if filter_itb_mode == "eq" and irb != filter_itb_val:
            continue
        if filter_itb_mode == "le" and irb > filter_itb_val:
            continue
        if filter_itb_mode == "ge" and irb < filter_itb_val:
            continue

        selected_files.append(f)

    n_ranges = len(selected_files)

    if n_ranges == 1:
        MIN_LINEWIDTH = 0.06
        MAX_LINEWIDTH = 0.06
        LINEWIDTH = 0.06
        MIN_ALPHA = 0.3
        MAX_ALPHA = 0.5
    elif n_ranges < 5:
        MIN_LINEWIDTH = 0.02
        MAX_LINEWIDTH = 0.06
        LINEWIDTH = 0.0
        MIN_ALPHA = 0.2
        MAX_ALPHA = 0.4
    else:
        MIN_LINEWIDTH = 0.01
        MAX_LINEWIDTH = 0.03
        LINEWIDTH = 0.03
        MIN_ALPHA = 0.2
        MAX_ALPHA = 0.4


    print(f"\nSelected {len(selected_files)} bootstrap files for plotting.")
    if filter_ita_mode:
        print(f"  ita filter: {filter_ita_mode} {filter_ita_val}")
    if filter_itb_mode:
        print(f"  itb filter: {filter_itb_mode} {filter_itb_val}")


    # === 2. Gewichte einlesen ===
    for f in selected_files:
        with open(f) as file:
            for line in file:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) < 19: continue
                weights.append(float(parts[18]))
    
    transformed_weights = [w**GAMMA for w in weights]
    w_min = min(transformed_weights) if transformed_weights else 0.0
    w_max = max(transformed_weights) if transformed_weights else 1.0

    print(f"Loaded {len(weights)} weights for alpha scaling.")
    if len(weights) == 0:
        print("Warning: No weights found. Defaulting w_min=0, w_max=1.")
    else:
        print(f"Weight range after transformation: min={w_min:.4f}, max={w_max:.4f}")

    idx = 0
    plotted_label = False

    for f in selected_files:
        ira = int(f.split("_")[-2])
        irb = int(f.split("_")[-1].split(".")[0])
        with open(f) as file:
            for line in file:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) < 19: continue

                V0_val = float(parts[2])
                alpha_val = float(parts[4])
                sigma_val = float(parts[6])
                weight = float(parts[18])
                weight_trans = weight ** GAMMA

                if w_max - w_min < 1e-10:
                    print(f"Warning: w_max ≈ w_min for ira={ira}, irb={irb}. Using default alpha.")
                    normed = 1.0
                else:
                    normed = (weight_trans - w_min) / (w_max - w_min)
                if USE_ALPHA_WITH_MINIMUM:
                    alpha_plot = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * normed
                else:
                    alpha_plot = MAX_ALPHA * weight_trans
                print(f"ira={ira}, irb={irb}, weight={weight:.4f}, w_trans={weight_trans:.4g}, normed={normed:.4f}, alpha_plot={alpha_plot:.4f}")

                # === Bootstrap-Samples laden ===
                bootstrap_sample_file = os.path.join(
                    bootstrap_dir, f"bootstrap_{ira}_{irb}",
                    f"bootstrapsamples_{ira}_{irb}.dat"
                )

                if not os.path.exists(bootstrap_sample_file):
                    print(f"Missing bootstrap sample file: {bootstrap_sample_file}")
                    continue

                with open(bootstrap_sample_file) as fs:
                    next(fs)  # Header
                    for s_line in fs:
                        p = s_line.strip().split()
                        if len(p) < 4:
                            print(f"Invalid bootstrap sample line (ignored): {p}")
                            continue
                        V0_b, alpha_b, sigma_b = map(float, p[1:4])
                        V_b = V_cornell(r_vals, V0_b, alpha_b, sigma_b)

                        if USE_WEIGHTED_LINEWIDTH:
                            linewidth = MIN_LINEWIDTH + (MAX_LINEWIDTH - MIN_LINEWIDTH) * normed
                        else:
                            linewidth = LINEWIDTH



                        plt.plot(r_vals, V_b, color=color, alpha=alpha_plot, linewidth=linewidth, label=label if not plotted_label else None)
                        plotted_label = True
        idx += 1

    # === Messpunkte plotten ===
    plt.errorbar(
        x_data, y_data, yerr=y_errs,
        fmt='o', color='black', markerfacecolor='none',
        markersize=6, elinewidth=1.2, capsize=0,
        label="mean $V(r)$", zorder=3
    )

    # === y-Achsenbereich setzen ===
    curve_stats_sorted = sorted(curve_stats, key=lambda x: x[0], reverse=True)
    total_weight = sum(w for w, _, _ in curve_stats_sorted)
    cum_weight = 0.0
    vmins_selected, vmaxs_selected = [], []

    for w, vmin_c, vmax_c in curve_stats_sorted:
        cum_weight += w
        vmins_selected.append(vmin_c)
        vmaxs_selected.append(vmax_c)
        if cum_weight >= WEIGHT_COVERAGE * total_weight:
            break

    min_y_weighted = min(vmins_selected) if vmins_selected else global_vmin
    max_y_weighted = max(vmaxs_selected) if vmaxs_selected else global_vmax

    ymin_error = min(y - e for y, e in zip(y_data, y_errs))
    ymax_error = max(y + e for y, e in zip(y_data, y_errs))
    error_margin = 0.05 * (ymax_error - ymin_error)
    ymin_error -= error_margin
    ymax_error += error_margin

    min_y = min(min_y_weighted, ymin_error)
    max_y = max(max_y_weighted, ymax_error)
    plt.ylim(min_y, max_y)

    plt.xlabel(r"$r$")
    plt.ylabel(r"$V(r)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n--- y-axis range info for", os.path.basename(output_path), "---")
    print(f"Global y-axis range:           [{global_vmin:.2f}, {global_vmax:.2f}]")
    print(f"Weighted {int(WEIGHT_COVERAGE*100)}% range:   [{min_y_weighted:.2f}, {max_y_weighted:.2f}]")
    print(f"Final y-axis range (incl. errors): [{min_y:.2f}, {max_y:.2f}]")

    print("\n--- Plot-Konfiguration ---")
    print(f"USE_FILTER_PLOT = {USE_FILTER_PLOT}")
    if USE_FILTER_PLOT:
        print(f"  FILTER_ITA_MODE = {FILTER_ITA_MODE}, FILTER_ITA_VAL = {FILTER_ITA_VAL}")
        print(f"  FILTER_ITB_MODE = {FILTER_ITB_MODE}, FILTER_ITB_VAL = {FILTER_ITB_VAL}")
    print(f"USE_ALPHA_WITH_MINIMUM = {USE_ALPHA_WITH_MINIMUM}")
    if USE_ALPHA_WITH_MINIMUM:
        print(f"  MIN_ALPHA = {MIN_ALPHA}")
        print(f"  MAX_ALPHA = {MAX_ALPHA}")
    else:
        print(f"  GAMMA = {GAMMA}")

    print(f"USE_WEIGHTED_LINEWIDTH = {USE_WEIGHTED_LINEWIDTH}")
    if USE_ALPHA_WITH_MINIMUM:
        print(f"  MIN_LINEWIDTH = {MIN_LINEWIDTH}")
        print(f"  MAX_LINEWIDTH = {MAX_LINEWIDTH}")
    else:
        print(f"  LINEWIDTH = {LINEWIDTH}")




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

if not USE_FILTER_PLOT:
    plot_cornell_fits(
        r_vals, x_data, y_data, y_errs,
        bootstrap_dir=bootstrap_dir,
        output_path=os.path.join(plot_dir, "cornell_fit_all_ranges.png"),
        filter_ita_mode=None,
        filter_ita_val=None,
        filter_itb_mode=None,
        filter_itb_val=None,
        color="blue",
        label="alle Ranges"
    )

if USE_FILTER_PLOT:
    plot_cornell_fits(
        r_vals, x_data, y_data, y_errs,
        bootstrap_dir=bootstrap_dir,
        output_path=os.path.join(
            plot_dir,
            f"cornell_fit_filtered_ita_{FILTER_ITA_MODE}_{FILTER_ITA_VAL}_"
            f"itb_{FILTER_ITB_MODE}_{FILTER_ITB_VAL}.png"
        ),
        filter_ita_mode=FILTER_ITA_MODE,
        filter_ita_val=FILTER_ITA_VAL,
        filter_itb_mode=FILTER_ITB_MODE,
        filter_itb_val=FILTER_ITB_VAL,
        color="red",
        label=f"ita {FILTER_ITA_MODE} {FILTER_ITA_VAL}, itb {FILTER_ITB_MODE} {FILTER_ITB_VAL}"
    )








