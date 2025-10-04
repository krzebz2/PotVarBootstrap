import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import re


x_r0, xerr_r0, y_r0, yerr_r0 = [], [], [], []
x_t0, xerr_t0, y_t0, yerr_t0 = [], [], [], []
x_w0, xerr_w0, y_w0, yerr_w0 = [], [], [], []
labels = []

base_dir = os.path.expanduser("~/data/analysis")
for folder in sorted(glob.glob(os.path.join(base_dir, "*/results_*"))):
    filepath = os.path.join(folder, "bootstrap_parameters", "mean_parameters.dat")
    if not os.path.isfile(filepath):
        continue

    mean_data = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = re.split(r'\s+', line.strip())
            if len(parts) < 3:
                print(f"Skipped malformed line in {filepath}: {line.strip()}")
                continue
            try:
                mean_data[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                print(f"Failed to parse line in file: {filepath}")
                print(f"Problematic line: {line.strip()}")
                continue

            # nur wenn beide Werte vorhanden sind:
            # prüfe ob alle nötigen Werte existieren
    keys_needed = ["r0sig", "inv_r0sq", "sqrt8t0_sig", "inv_sqrt8t0_sq", "sqrt8w0_sig", "inv_sqrt8w0_sq"]
    if not all(k in mean_data for k in keys_needed):
        print(f"Skipped file {filepath}, missing keys: {[k for k in keys_needed if k not in mean_data]}")
        continue

    r0sig, r0sig_err = mean_data["r0sig"]
    inv_r0sq, inv_r0sq_err = mean_data["inv_r0sq"]
    sqrt8t0_sig, sqrt8t0_sig_err = mean_data["sqrt8t0_sig"]
    inv_sqrt8t0_sq, inv_sqrt8t0_sq_err = mean_data["inv_sqrt8t0_sq"]
    sqrt8w0_sig, sqrt8w0_sig_err = mean_data["sqrt8w0_sig"]
    inv_sqrt8w0_sq, inv_sqrt8w0_sq_err = mean_data["inv_sqrt8w0_sq"]

    label = os.path.basename(folder).replace("results_", "").rsplit("_rmax", 1)[0]

    x_r0.append(inv_r0sq)
    xerr_r0.append(inv_r0sq_err)
    y_r0.append(r0sig)
    yerr_r0.append(r0sig_err)

    x_t0.append(inv_sqrt8t0_sq)
    xerr_t0.append(inv_sqrt8t0_sq_err)
    y_t0.append(sqrt8t0_sig)
    yerr_t0.append(sqrt8t0_sig_err)

    x_w0.append(inv_sqrt8w0_sq)
    xerr_w0.append(inv_sqrt8w0_sq_err)
    y_w0.append(sqrt8w0_sig)
    yerr_w0.append(sqrt8w0_sig_err)

    labels.append(label)
    print(f"Loaded values from: {filepath}")

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

def compute_axis_range(values, errors, labels, sm_filter, include_errors):
    if sm_filter == "sm0":
        vals = [values[i] for i in range(len(values)) if "sm0" in labels[i].lower()]
        vmin = min(vals)
        vmax = max(vals)
    elif sm_filter == "sm2":
        vals = [values[i] for i in range(len(values)) if "sm2" in labels[i].lower()]
        errs = [errors[i] for i in range(len(errors)) if "sm2" in labels[i].lower()]
        if include_errors:
            # Median-based outlier filter
            abs_errs = np.array(errs)
            median_err = np.median(abs_errs)
            factor = 5
            filtered = [(v, e) for v, e in zip(vals, errs) if e <= factor * median_err]
            print(f"{len(filtered)} of {len(errs)} points kept after outlier filtering (threshold = {factor} × median)")
            if filtered:
                vmin = min([v - e for v, e in filtered])
                vmax = max([v + e for v, e in filtered])
            else:
                vmin = min(vals)
                vmax = max(vals)
        else:
            vmin = min(vals)
            vmax = max(vals)

    else:
        vals_sm0 = [values[i] for i in range(len(values)) if "sm0" in labels[i].lower()]
        vals_sm2 = [values[i] for i in range(len(values)) if "sm2" in labels[i].lower()]
        errs_sm2 = [errors[i] for i in range(len(errors)) if "sm2" in labels[i].lower()]
        vmin = min(vals_sm0 + [v - e for v, e in zip(vals_sm2, errs_sm2)])
        vmax = max(vals_sm0 + [v + e for v, e in zip(vals_sm2, errs_sm2)])
    vrange = vmax - vmin
    pad = 0.05 * vrange
    return vmin - pad, vmax + pad

def plot_scaling(xvals, xerrs, yvals, yerrs, labels, sm_filter=None, filename="scaling.png", legend_cols=2):
    plt.figure(figsize=(12, 5))

    for i in range(len(xvals)):
        label_parts = labels[i].split("_")
        vol = label_parts[0]
        beta = label_parts[1]
        sm = label_parts[2]

        if sm_filter and sm != sm_filter:
            continue

        L = vol.split("t")[0].replace("l", "")
        T = vol.split("t")[1]
        vol_pretty = rf"${L} \times {T}^3$"

        beta_val = float(beta.replace("b", ""))
        beta_pretty = f"β={beta_val:.2f}"
        label_nice = f"{vol_pretty}, {beta_pretty}, {sm.upper()}"

        markerform = vol_to_marker.get(vol, 'o')
        farbe = beta_to_color.get(beta, 'gray')
        face = 'none' if sm == 'sm0' else farbe
        print(f"Plotting point: {label_nice}")
        plt.errorbar(xvals[i], yvals[i], xerr=xerrs[i], yerr=yerrs[i],
                     marker=markerform,
                     color=farbe,
                     markerfacecolor=face,
                     markeredgecolor=farbe,
                     ecolor='black',
                     elinewidth=1.2,
                     capsize=3,
                     capthick=1,
                     label=label_nice)

    handles, lbls = plt.gca().get_legend_handles_labels()
    if sm_filter is None:
        sm0 = [(h, l) for h, l in zip(handles, lbls) if "SM0" in l]
        sm2 = [(h, l) for h, l in zip(handles, lbls) if "SM2" in l]
        sorted_pairs = sm0 + sm2
    else:
        sorted_pairs = list(zip(handles, lbls))

    sorted_handles, sorted_labels = zip(*sorted_pairs)

    y_lower, y_upper = compute_axis_range(yvals, yerrs, labels, sm_filter, include_errors=True)
    x_lower, x_upper = compute_axis_range(xvals, xerrs, labels, sm_filter, include_errors=True)


    # plt.xlim(0.0, 0.5)
    plt.xlim(x_lower, x_upper)
    # plt.ylim(1.10, 1.25)
    plt.ylim(y_lower, y_upper)
    if "_r0_" in filename:
        plt.xlabel(r'$(a/r_0)^2$')
        plt.ylabel(r'$r_0 \cdot \sqrt{\sigma}$')
    elif "_t0_" in filename:
        plt.xlabel(r'$(1/\sqrt{8t_0})^2$')
        plt.ylabel(r'$\sqrt{8t_0} \cdot \sqrt{\sigma}$')
    elif "_w0_" in filename:
        plt.xlabel(r'$(1/\sqrt{8}w_0)^2$')
        plt.ylabel(r'$\sqrt{8}w_0 \cdot \sqrt{\sigma}$')
    plt.grid(True)
    plt.legend(
        sorted_handles, sorted_labels,
        loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=legend_cols
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.close()

plot_scaling(x_r0, xerr_r0, y_r0, yerr_r0, labels, sm_filter=None, filename="scaling_r0_all.png", legend_cols=2)
plot_scaling(x_r0, xerr_r0, y_r0, yerr_r0, labels, sm_filter="sm0", filename="scaling_r0_sm0.png", legend_cols=1)
plot_scaling(x_r0, xerr_r0, y_r0, yerr_r0, labels, sm_filter="sm2", filename="scaling_r0_sm2.png", legend_cols=1)

plot_scaling(x_t0, xerr_t0, y_t0, yerr_t0, labels, sm_filter=None, filename="scaling_t0_all.png", legend_cols=2)
plot_scaling(x_t0, xerr_t0, y_t0, yerr_t0, labels, sm_filter="sm0", filename="scaling_t0_sm0.png", legend_cols=1)
plot_scaling(x_t0, xerr_t0, y_t0, yerr_t0, labels, sm_filter="sm2", filename="scaling_t0_sm2.png", legend_cols=1)

plot_scaling(x_w0, xerr_w0, y_w0, yerr_w0, labels, sm_filter=None, filename="scaling_w0_all.png", legend_cols=2)
plot_scaling(x_w0, xerr_w0, y_w0, yerr_w0, labels, sm_filter="sm0", filename="scaling_w0_sm0.png", legend_cols=1)
plot_scaling(x_w0, xerr_w0, y_w0, yerr_w0, labels, sm_filter="sm2", filename="scaling_w0_sm2.png", legend_cols=1)


