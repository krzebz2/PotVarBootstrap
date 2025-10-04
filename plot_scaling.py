import os
import glob
import matplotlib.pyplot as plt
import numpy as np

x, xerr, y, yerr, labels = [], [], [], [], []

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
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                mean_data[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                print(f"Failed to parse line in file: {filepath}")
                print(f"Problematic line: {line.strip()}")
                continue

            # nur wenn beide Werte vorhanden sind:
            if "r0sig" not in mean_data or "inv_r0sq" not in mean_data:
                continue

            xval, xerrval = mean_data["inv_r0sq"]
            yval, yerrval = mean_data["r0sig"]
            label = os.path.basename(folder).replace("results_", "")
            label = label.rsplit("_rmax", 1)[0]
            x.append(xval)
            xerr.append(xerrval)
            y.append(yval)
            yerr.append(yerrval)
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

def plot_scaling(sm_filter=None, filename="scaling.png", legend_cols=2):
    plt.figure(figsize=(12, 5))

    for i in range(len(x)):
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
        plt.errorbar(x[i], y[i], xerr=xerr[i], yerr=yerr[i],
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

    y_lower, y_upper = compute_axis_range(y, yerr, labels, sm_filter, include_errors=True)
    x_lower, x_upper = compute_axis_range(x, xerr, labels, sm_filter, include_errors=True)

    # plt.xlim(0.0, 0.5)
    plt.xlim(x_lower, x_upper)
    # plt.ylim(1.10, 1.25)
    plt.ylim(y_lower, y_upper)
    plt.xlabel(r'$(a/r_0)^2$')
    plt.ylabel(r'$r_0 \cdot \sqrt{\sigma}$')
    plt.grid(True)
    plt.legend(
        sorted_handles, sorted_labels,
        loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=legend_cols
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.close()


plot_scaling(sm_filter=None, filename="scaling_all.png", legend_cols=2)
plot_scaling(sm_filter="sm0", filename="scaling_sm0.png", legend_cols=1)
plot_scaling(sm_filter="sm2", filename="scaling_sm2.png", legend_cols=1)


