import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def V_cornell(r, V0, alpha, sigma):
    return V0 + alpha / r + sigma * r

vol_to_marker = {
    'l10t10': 'o',  # Kreis
    'l14t14': 's',  # Quadrat
    'l16t16': '^',  # Dreieck oben
    'l18t18': 'D',  # Raute
}
beta_to_color = {
    'b2.650': 'blue',
    'b2.750': 'cyan',
    'b2.800': 'green',
    'b2.850': 'lime',
    'b2.900': 'orange',
    'b2.950': 'gold',
    'b3.000': 'red',
    'b3.050': 'magenta',
    'b3.100': 'purple',
    'b3.150': 'brown',
    'b3.200': 'black',
}



plt.figure(figsize=(12, 5))

base_dir = os.path.expanduser("~/data/analysis")
for i, folder in enumerate(sorted(glob.glob(os.path.join(base_dir, "*/results_*")))):
    bpath = os.path.join(folder, "bootstrap_parameters", "mean_parameters.dat")
    mpath = os.path.join(folder, "mean_potential")

    if not os.path.isfile(bpath) or not os.path.isdir(mpath):
        continue

    mean_values = {}
    with open(bpath) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            mean_values[parts[0]] = (float(parts[1]), float(parts[2]))

    if not all(k in mean_values for k in ["V0", "alpha", "sigma", "r0"]):
        continue

    V0, V0_err = mean_values["V0"]
    alpha, alpha_err = mean_values["alpha"]
    sigma, sigma_err = mean_values["sigma"]
    r0, r0_err = mean_values["r0"]

    V_r0  = V_cornell(r0, V0, alpha, sigma)

    x_vals, y_vals, x_errs, y_errs = [], [], [], []

    for f_mean in sorted(glob.glob(os.path.join(mpath, "mean_r*.dat"))):
        r_str = os.path.basename(f_mean).split("_r")[1].split(".")[0]
        r = int(r_str)
        with open(f_mean) as f:
            lines = f.readlines()
        if len(lines) < 2:
            continue
        V_mean = float(lines[1].split()[0])
        V_err  = float(lines[1].split()[1]) 
        x_vals.append(r / r0)
        x_errs.append(r * r0_err / (r0 * r0))
        y_vals.append(r0 * (V_mean - V_r0))
        y_errs.append(r0 * V_err)

    foldername = os.path.basename(folder)
    label_parts = foldername.replace("results_", "").split("_")
    vol = label_parts[0]
    beta = label_parts[1]
    sm = label_parts[2] if len(label_parts) > 2 else "sm?"

    L = vol.replace("l", "").split("t")[0]
    T = vol.replace("l", "").split("t")[1]
    vol_pretty = rf"${L} \times {T}^3$"
    beta_val = float(beta.replace("b", ""))
    beta_pretty = f"β={beta_val:.2f}"
    label = f"{vol_pretty}, {beta_pretty}, {sm.upper()}"

    markerform = vol_to_marker.get(vol, 'o')
    farbe = beta_to_color.get(beta, 'gray')
    face = 'none' if sm == 'sm0' else farbe

    plt.errorbar(
        x_vals, y_vals,
        xerr=x_errs,  yerr=y_errs,
        fmt=markerform,
        color=farbe,
        markerfacecolor=face,
        markeredgecolor=farbe,
        ecolor='black',
        elinewidth=1.2,
        capsize=3,
        capthick=1,
        label=label
    )


plt.xlabel(r"$r/r_0$")
plt.ylabel(r"$r_0(V(r)-V(r_0))$")
plt.grid(True)
plt.xlim(0, 2.2)
plt.ylim(-2.7, 3)

handles, lbls = plt.gca().get_legend_handles_labels()
sm0 = [(h, l) for h, l in zip(handles, lbls) if "SM0" in l]
sm2 = [(h, l) for h, l in zip(handles, lbls) if "SM2" in l]
sorted_pairs = sm0 + sm2
sorted_handles, sorted_labels = zip(*sorted_pairs)

plt.legend(
    sorted_handles, sorted_labels,
    loc='center left',
    bbox_to_anchor=(1.0, 0.5),
    ncol=2
)

plt.tight_layout()
plt.savefig("static_potential.png")
plt.close()