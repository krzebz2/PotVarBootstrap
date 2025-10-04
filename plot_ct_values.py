import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

def get_best_t0_t1(base_dir, radius):
    best_file = os.path.join(base_dir, "best_model_potential", f"best_model_{radius}.dat")
    if not os.path.exists(best_file):
        return None, None
    with open(best_file) as f:
        lines = [l for l in f if not l.startswith("#")]
    if not lines:
        return None, None
    parts = lines[0].split()
    t0, t1 = parts[0].strip(), parts[1].strip()
    return t0, t1


# Ensemble definieren

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

base_dir = os.path.expanduser(f"~/data/analysis/{ensemble}/results_{ensemble}")
input_dir = f"{base_dir}/corb"
plot_dir = f"{base_dir}/plots/ct_values"

os.makedirs(plot_dir, exist_ok=True)


# Flag: "all" = alle t0/t1, "best" = nur best_model
plot_mode = "best"

if plot_mode == "all":
    print("Plot mode = all → using all correlator files")
    files = glob.glob(f"{input_dir}/corb_*.dat")

elif plot_mode == "best":
    print("Plot mode = best → using only best_model t0/t1")
    files = []
    for radius in range(1, rmax+1):
        t0, t1 = get_best_t0_t1(base_dir, f"r{radius:02d}")
        if t0 and t1:
            print(f"Using best model for radius {radius}: t0={t0}, t1={t1}")
            pattern = f"{input_dir}/corb_in_wl_{ensemble}_r{radius:02d}_{t0}_{t1}.dat"
            files.extend(glob.glob(pattern))
        else:
            print(f"No best_model found for radius {radius}")

else:
    print("Error: plot_mode must be 'all' or 'best'")
    sys.exit(1)



for filepath in files:
    # Datei einlesen
    data = np.loadtxt(filepath, skiprows=2)
    t = data[:, 0]
    ct = data[:, 1]
    err = data[:, 2]

    # Output-Dateiname (aus .dat → .png)
    filename = os.path.splitext(os.path.basename(filepath))[0].replace("corb_", "ct_values_") + ".png"
    outpath_ln = os.path.join(plot_dir, "ln_" + filename)
    outpath_ct = os.path.join(plot_dir, filename)


    # Plot 1: ln(c(t)) mit Fehlerbalken (nur positive c(t))
    mask = ct > 0
    t_pos = t[mask]
    ct_pos = ct[mask]
    err_pos = err[mask]
    log_ct = np.log(ct_pos)
    err_log = err_pos / ct_pos

    plt.errorbar(t_pos, log_ct, yerr=err_log, fmt='o', linestyle='-', linewidth=0.5,
                 color='blue', markersize=4, capsize=4, elinewidth=1.2, label='ln(c(t))')
    plt.xlabel('t/a')
    plt.ylabel('ln(c(t))')
    plt.title('Logarithmus des Korrelators')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath_ln)
    plt.close()

    # Plot 2: c(t) mit Fehlerbalken
    plt.errorbar(t, ct, yerr=err, fmt='o', linestyle='-', linewidth=0.5,
                 color='green', markersize=4, capsize=4, elinewidth=1.2, label='c(t)')
    plt.xlabel('t/a')
    plt.ylabel('c(t)')
    plt.title('Korrelator c(t)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath_ct)
    plt.close()

