import os
import glob
import math
import re
import sys


if len(sys.argv) == 4:
    ENSEMBLE = sys.argv[1]
    smearing = int(sys.argv[2])
    rmax = int(sys.argv[3])
else:
    with open("input_ensemble.txt") as f:
        ENSEMBLE = f.readline().strip()
        smearing = int(f.readline())
        rmax = int(f.readline())

match = re.search(r"t(\d+)", ENSEMBLE)
n_tot = int(match.group(1))

ensemble = f"{ENSEMBLE}_sm{smearing}_rmax{rmax:02d}"
base_dir = os.path.expanduser(f"~/data/analysis/{ensemble}/results_{ensemble}")

input_dir = f"{base_dir}/final_energies_clean"
weights_dir = f"{base_dir}/aic_weights_potential_t"
mean_dir = f"{base_dir}/mean"
best_model_dir = f"{base_dir}/best_model"

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(mean_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)

# Alte Dateien löschen
for f in glob.glob(f"{weights_dir}/weights_r*_*_*.dat"):
    os.remove(f)
for f in glob.glob(f"{mean_dir}/mean_r*.dat"):
    os.remove(f)
for f in glob.glob(f"{best_model_dir}/best_model_r*.dat"):
    os.remove(f)


for filepath in glob.glob(f"{input_dir}/*.dat"):
    radius = re.search(r"r\d+", filepath).group()
    parts = os.path.splitext(os.path.basename(filepath))[0].split('_')
    t0, t1 = parts[-4], parts[-3]
    ita, itb = parts[-2], parts[-1]

    output_path = f"{weights_dir}/weights_{radius}_{t0}_{t1}_{ita}_{itb}.dat"

    with open(filepath) as f:
        lines = f.readlines()
        n = int(lines[1])
        k = int(lines[2])
        V = float(lines[3])
        err = float(lines[4])  
        bias = float(lines[5]) 
        chi2 = float(lines[6])

    aic = chi2 + 2 * k + 2 * (n_tot - n)
    weight = math.exp(-aic / 2)

    header = f"{'#':<4} {'ita':>4} {'itb':>5} {'weight':>12} {'V':>10} {'error':>10} {'bias':>10} {'chi2':>10} {'n':>4} {'k':>4} {'ntot':>6}\n"
    line = f"{'':<4} {ita:>4} {itb:>5} {weight:12.10f} {V:10.6f} {err:10.6f} {bias:10.6f} {chi2:10.6f} {n:4} {k:4} {n_tot:6}\n"

    if not os.path.isfile(output_path):
        with open(output_path, "w") as f:
            f.write(header)

    with open(output_path, "a") as f:
        f.write(line)

# Globale Mittelwerte und Bestmodelle (über alle Ranges und t0/t1 pro Radius)
for radius in set(re.search(r"(r\d+)", f).group(1) for f in glob.glob(f"{weights_dir}/weights_r*_*_*_*_*.dat")):
    all_lines = []
    for file in glob.glob(f"{weights_dir}/weights_{radius}_*.dat"):
        with open(file) as f:
            lines = [l for l in f if not l.startswith("#")]
            all_lines.extend(lines)

    v_values, s_values, aic_weights, stat_weights = [], [], [], []

    for line in all_lines:
        parts = line.strip().split()
        w_aic, v, s = float(parts[2]), float(parts[3]), float(parts[4])
        if s == 0 or w_aic == 0:
            continue
        v_values.append(v)
        s_values.append(s)
        aic_weights.append(w_aic)
        stat_weights.append(1 / s**2)

    if not v_values:
        mean, err_prop, err_std = float("nan"), float("nan"), float("nan")
    else:
        # Normierung der Gewichte
        total_w_aic = sum(aic_weights)
        aic_weights = [w / total_w_aic for w in aic_weights]
        # Kontrolle Normierung
        total_check = sum(aic_weights)
        # Mittelwert mit normierten AIC-Gewichten
        mean = sum(w * v for w, v in zip(aic_weights, v_values))      
        # statistischer Fehler
        err_stat = math.sqrt(sum(w * s**2 for w, s in zip(aic_weights, s_values)))
        # systematischer Fehler
        err_syst = math.sqrt(sum(w * (v - mean)**2 for w, v in zip(aic_weights, v_values)))
        # totaler Fehler
        err_total = math.sqrt(err_stat**2 + err_syst**2)

    mean_path = f"{mean_dir}/mean_{radius}.dat"
    with open(mean_path, "w") as f:
        f.write(f"{'#':<6}{'mean':>14}{'err_total':>14}{'err_stat':>14}{'err_syst':>14}{'sum_weights':>14}\n")
        f.write(f"{'':<6}{mean:14.10f}{err_total:14.10f}{err_stat:14.10f}{err_syst:14.10f}{total_check:14.10f}\n")


    # Bestes Modell global
    min_aic, best_line = None, ""
    for line in all_lines:
        parts = line.split()
        chi2, n, k = float(parts[6]), int(parts[7]), int(parts[8])
        aic = chi2 + 2 * k + 2 * (n_tot - n)
        if min_aic is None or aic < min_aic:
            min_aic, best_line = aic, line

    out_path = f"{best_model_dir}/best_model_{radius}.dat"
    with open(out_path, "w") as f:
        f.write(header)
        f.write(best_line)
