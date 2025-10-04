import os
import glob
import re
import sys
from collections import defaultdict

# === Parameter einlesen ===
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
weights_dir = f"{base_dir}/aic_weights_potential_t"

print(f"[INFO] Suche in {weights_dir}")

radius_data = defaultdict(list)
file_count = 0

for path in glob.glob(f"{weights_dir}/weights_*.dat"):
    m = re.search(r"weights_(r\d+)_", path)
    if not m:
        print(f"[WARN] Kein Radius in Datei: {path}")
        continue
    radius = m.group(1)
    file_count += 1
    print(f"[DEBUG] Lese Datei {os.path.basename(path)} für Radius {radius}")

    with open(path) as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            t0, t1, ita, itb = map(int, parts[0:4])
            weight = float(parts[4])
            V = float(parts[5])
            error = float(parts[6])
            weighted_error = weight * error
            radius_data[radius].append((t0, t1, ita, itb, weight, V, error, weighted_error))

print(f"[INFO] {file_count} Dateien verarbeitet.")

# === Sortieren und Speichern ===
for radius, rows in radius_data.items():
    rows.sort(key=lambda x: -x[7])  # nach weighted_error absteigend

    out_path = os.path.join(weights_dir, f"sorted_weights_{radius}.dat")
    print(f"[INFO] Schreibe {len(rows)} Zeilen nach {out_path}")

    with open(out_path, "w") as f:
        header = f"{'#':<4} {'t0':>3} {'t1':>3} {'ita':>4} {'itb':>5} {'weight':>12} {'V':>10} {'error':>10} {'weighted_err':>14}\n"
        f.write(header)
        for r in rows:
            f.write(f"{'':<4} {r[0]:3} {r[1]:3} {r[2]:4} {r[3]:5} {r[4]:12.10f} {r[5]:10.6f} {r[6]:10.6f} {r[7]:14.10f}\n")
