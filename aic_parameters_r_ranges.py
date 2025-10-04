import os
import glob
import math
import shutil 
import sys

def err_prop(values, errors, weights):
    total_weight = sum(weights)
    if total_weight == 0:
        return float("nan"), float("nan")
    mean = sum(v * w for v, w in zip(values, weights)) / total_weight
    err = math.sqrt(sum(w * s**2 for w, s in zip(weights, errors))) / total_weight
    return mean, err

def err_std(values, errors, weights):
    total_weight = sum(weights)
    if total_weight == 0:
        return float("nan"), float("nan")
    mean = sum(v * w for v, w in zip(values, weights)) / total_weight
    stat_weights = [1 / s**2 if s > 0 else 0 for s in errors]
    total_stat = sum(stat_weights)
    var = sum(w * (v - mean)**2 for w, v in zip(stat_weights, values)) / (total_stat - 1)
    err = math.sqrt(var)
    return mean, err


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
input_dir = f"{base_dir}/bootstrap_parameters"
mean_file = os.path.join(input_dir, "mean_parameters.dat")
best_model_file = os.path.join(input_dir, "best_model_parameters.dat")


# Daten sammeln
# Daten sammeln
V0, V0_err = [], []
alpha, alpha_err = [], []
sigma, sigma_err = [], []
r0, r0_err = [], []
r0sig, r0sig_err = [], []
inv_r0sq, inv_r0sq_err = [], []
sqrt8t0_sig, sqrt8t0_sig_err = [], []
inv_sqrt8t0_sq, inv_sqrt8t0_sq_err = [], []
sqrt8w0_sig, sqrt8w0_sig_err = [], []
inv_sqrt8w0_sq, inv_sqrt8w0_sq_err = [], []
aic_weights = []
best_line = ""
min_aic = None
ira_longest = None
irb_longest = None
max_range_len = -1


# Einlesen
for file in sorted(glob.glob(f"{input_dir}/bootstrap*.dat")):
    with open(file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 19:
                print(f"Zeile übersprungen ({len(parts)} Werte): {parts}")
                continue

            try:
                ira = int(parts[0])
                irb = int(parts[1])
                V0.append(float(parts[2]))
                V0_err.append(float(parts[3]))
                alpha.append(float(parts[4]))
                alpha_err.append(float(parts[5]))
                sigma.append(float(parts[6]))
                sigma_err.append(float(parts[7]))
                r0.append(float(parts[8]))
                r0_err.append(float(parts[9]))
                r0sig.append(float(parts[10]))
                r0sig_err.append(float(parts[11]))
                inv_r0sq.append(float(parts[12]))
                inv_r0sq_err.append(float(parts[13]))
                sqrt8t0_sig.append(float(parts[14]))
                sqrt8t0_sig_err.append(float(parts[15]))
                inv_sqrt8t0_sq.append(float(parts[16]))
                inv_sqrt8t0_sq_err.append(float(parts[17]))
                sqrt8w0_sig.append(float(parts[18]))
                sqrt8w0_sig_err.append(float(parts[19]))
                inv_sqrt8w0_sq.append(float(parts[20]))
                inv_sqrt8w0_sq_err.append(float(parts[21]))
                chi2 = float(parts[22])
                n, k, ntot = int(parts[23]), int(parts[24]), int(parts[25])
                weight = float(parts[26])           
            except ValueError as e:
                print(f"Value error in file {file}: {e}")
                print(f"Problematic line: {line.strip()}")
                continue

            aic_weights.append(weight)

            aic = chi2 + 2 * k + 2 * (ntot - n)
            if min_aic is None or aic < min_aic:
                min_aic = aic
                best_line = line

            range_len = irb - ira
            if range_len > max_range_len:
                max_range_len = range_len
                ira_longest = ira
                irb_longest = irb


print(f"→ Anzahl eingelesener Datenzeilen: {len(aic_weights)}")
print(f"→ Anzahl Gewichte > 0: {sum(1 for w in aic_weights if w > 0)}")
print(f"→ Summe aller Gewichte vor Normierung: {sum(aic_weights)}")


# Mittelwerte + Fehler
with open(mean_file, "w") as f:
    f.write(f"{'#':<10} {'mean':>20} {'err_total':>20} {'err_stat':>20} {'err_syst':>20} {'sum_weights':>10}\n")
    # Gewichte normieren
    total_w_aic = sum(aic_weights)
    aic_weights = [w / total_w_aic for w in aic_weights]
    total_check = sum(aic_weights)  # Kontrolle

    for label, vals, errs in [
        ("V0", V0, V0_err),
        ("alpha", alpha, alpha_err),
        ("sigma", sigma, sigma_err),
        ("r0", r0, r0_err),
        ("r0sig", r0sig, r0sig_err),
        ("inv_r0sq", inv_r0sq, inv_r0sq_err),
        ("sqrt8t0_sig", sqrt8t0_sig, sqrt8t0_sig_err),
        ("inv_sqrt8t0_sq", inv_sqrt8t0_sq, inv_sqrt8t0_sq_err),
        ("sqrt8w0_sig", sqrt8w0_sig, sqrt8w0_sig_err),
        ("inv_sqrt8w0_sq", inv_sqrt8w0_sq, inv_sqrt8w0_sq_err),
    ]:
        
        # Mittelwert
        mean = sum(v * w for v, w in zip(vals, aic_weights))
        # statistischer Fehler
        err_stat = math.sqrt(sum(w * s**2 for w, s in zip(aic_weights, errs)))
        # systematischer Fehler
        err_syst = math.sqrt(sum(w * (v - mean)**2 for w, v in zip(aic_weights, vals)))
        # totaler Fehler
        err_total = math.sqrt(err_stat**2 + err_syst**2)
        f.write(f"{label:<10} {mean:20.6f} {err_total:20.6f} {err_stat:20.6f} {err_syst:20.6f} {total_check:10.6f}\n")


idx = 0
for file in sorted(glob.glob(f"{input_dir}/bootstrap*.dat")):
    lines_out = []
    with open(file) as f:
        for line in f:
            if line.startswith("#"):
                lines_out.append(line)
                continue
            parts = line.strip().split()
            if len(parts) < 26:
                print(f"Skipped line in file {file} (only {len(parts)} values): {line.strip()}")
                lines_out.append(line)
                continue
            # Nur bei gültigen Zeilen (die wir gezählt haben) ersetzen
            line_fmt = (
                f"{'':<4} {int(parts[0]):6} {int(parts[1]):6} "
                f"{float(parts[2]):14.10f} {float(parts[3]):14.10f} "
                f"{float(parts[4]):14.10f} {float(parts[5]):14.10f} "
                f"{float(parts[6]):14.10f} {float(parts[7]):14.10f} "
                f"{float(parts[8]):14.10f} {float(parts[9]):14.10f} "
                f"{float(parts[10]):14.10f} {float(parts[11]):14.10f} "
                f"{float(parts[12]):18.10f} {float(parts[13]):18.10f} "
                f"{float(parts[14]):14.10f} {float(parts[15]):14.10f} "
                f"{float(parts[16]):18.10f} {float(parts[17]):18.10f} "
                f"{float(parts[18]):14.10f} {float(parts[19]):14.10f} "
                f"{float(parts[20]):18.10f} {float(parts[21]):18.10f} "
                f"{float(parts[22]):12.6f} {int(parts[23]):6} {int(parts[24]):6} {int(parts[25]):8} "
                f"{aic_weights[idx]:14.10f}\n"
            )

            idx += 1
            lines_out.append(line_fmt)
    # Datei überschreiben
    with open(file, "w") as f:
        f.writelines(lines_out)

# Bestes Modell speichern
header = (
    f"{'#':<4} {'ira':>6} {'irb':>6} "
    f"{'V0':>14} {'V0_err':>14} "
    f"{'alpha':>14} {'alpha_err':>14} "
    f"{'sigma':>14} {'sigma_err':>14} "
    f"{'r0':>14} {'r0_boot_err':>14} "
    f"{'r0sig':>14} {'r0sig_boot_err':>14} "
    f"{'inv_r0sq':>18} {'inv_r0sq_boot_err':>18} "
    f"{'sqrt8t0_sig':>14} {'sqrt8t0_sig_err':>14} "
    f"{'inv_sqrt8t0_sq':>18} {'inv_sqrt8t0_sq_err':>18} "
    f"{'sqrt8w0_sig':>14} {'sqrt8w0_sig_err':>14} "
    f"{'inv_sqrt8w0_sq':>18} {'inv_sqrt8w0_sq_err':>18} "
    f"{'chi2':>12} {'n':>6} {'k':>6} {'ntot':>8} "
    f"{'aic_weight':>14}\n"
)
with open(best_model_file, "w") as f_out:
    f_out.write(header)
    f_out.write(best_line)


# === Beste Range aus best_line extrahieren
parts = best_line.strip().split()
ira_best = int(parts[0])
irb_best = int(parts[1])

# === Quelldateien für bestes Modell
best_r_file = os.path.join(input_dir, f"r_over_r0_{ira_best}_{irb_best}.dat")
best_V_file = os.path.join(input_dir, f"scaled_potential_{ira_best}_{irb_best}.dat")

# === Zieldateien
best_r_out = os.path.join(input_dir, "best_model_r_over_r0.dat")
best_V_out = os.path.join(input_dir, "best_model_scaled_potential.dat")

# === Kopieren, falls vorhanden
if os.path.exists(best_r_file):
    shutil.copy(best_r_file, best_r_out)
else:
    print(f"Datei fehlt: {best_r_file}")

if os.path.exists(best_V_file):
    shutil.copy(best_V_file, best_V_out)
else:
    print(f"Datei fehlt: {best_V_file}")

# === Quelldateien für grösste Range
longest_r_file = os.path.join(input_dir, f"r_over_r0_{ira_longest}_{irb_longest}.dat")
longest_V_file = os.path.join(input_dir, f"scaled_potential_{ira_longest}_{irb_longest}.dat")

# === Zieldateien
longest_r_out = os.path.join(input_dir, "longest_range_r_over_r0.dat")
longest_V_out = os.path.join(input_dir, "longest_range_scaled_potential.dat")

# === Kopieren
if os.path.exists(longest_r_file):
    shutil.copy(longest_r_file, longest_r_out)
else:
    print(f"Datei fehlt: {longest_r_file}")

if os.path.exists(longest_V_file):
    shutil.copy(longest_V_file, longest_V_out)
else:
    print(f"Datei fehlt: {longest_V_file}")


