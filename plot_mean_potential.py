import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys

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

plot_dir = f"{base_path}/plots/potential_points"
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

# === Plot nur mit Punkten ===
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

output_path = os.path.join(plot_dir, "potential_points.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

# --- zusätzlich interaktives HTML speichern ---
import plotly.graph_objects as go

fig_html = go.Figure()
fig_html.add_trace(go.Scatter(
    x=x_data, y=y_data,
    error_y=dict(type='data', array=y_errs),
    mode='markers',
    name='mean V(r)',
    marker=dict(color='black', symbol='circle-open')
))
fig_html.update_layout(
    title="Potential points",
    xaxis_title="r",
    yaxis_title="V(r)",
    hovermode="closest"
)

html_output_path = output_path.replace(".png", ".html")
fig_html.write_html(html_output_path)
