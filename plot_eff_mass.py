import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import plotly.graph_objects as go


def save_plotly_html(t, m, err_lower, err_upper, is_half, is_int, title, outpath):
    fig_html = go.Figure()
    fig_html.add_trace(go.Scatter(
        x=t[is_half], y=m[is_half],
        error_y=dict(type='data', array=err_upper[is_half], arrayminus=err_lower[is_half]),
        mode='markers+lines',
        name='halbzahlig',
        marker=dict(color='blue')
    ))
    fig_html.add_trace(go.Scatter(
        x=t[is_int], y=m[is_int],
        error_y=dict(type='data', array=err_upper[is_int], arrayminus=err_lower[is_int]),
        mode='markers+lines',
        name='ganzzahlig',
        marker=dict(color='red')
    ))
    fig_html.update_layout(
        title=title,
        xaxis_title='t/a',
        yaxis_title='Effektive Masse',
        hovermode='closest'
    )
    fig_html.write_html(outpath)


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

# Plot mode: "all" = alle t0/t1, "best" = nur best_model
plot_mode = "best"

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
input_dir = f"{base_dir}/eff_mass"


if plot_mode == "all":
    print("Plot mode = all → using all t0/t1 combinations")
    files = glob.glob(f"{input_dir}/eff_mass_in_wl_{ensemble}_r*.dat")

elif plot_mode == "best":
    print("Plot mode = best → using only best_model t0/t1")
    files = []
    for radius in range(1, rmax+1):
        t0, t1 = get_best_t0_t1(base_dir, f"r{radius:02d}")
        if t0 and t1:
            print(f"Using best model for radius {radius}: t0={t0}, t1={t1}")
            pattern = f"{input_dir}/eff_mass_in_wl_{ensemble}_r{radius:02d}_{t0}_{t1}.dat"
            files.extend(glob.glob(pattern))
        else:
            print(f"No best_model found for radius {radius}")


else:
    print("Error: plot_mode must be 'all' or 'best'")
    sys.exit(1)


plot_dir = f"{base_dir}/plots/eff_mass"
os.makedirs(plot_dir, exist_ok=True)

# Dynamische Regeln zur Auswahl der Datenpunkte
min_t = 1.5  # Mindest-t-Wert, darunterliegende Werte werden ignoriert
err_factor = 1  # Maximal erlaubtes Verhältnis zum Medianfehler

print(f"Suche mit Pattern: {pattern}")
print("Gefundene Dateien:", files)

for filepath in files:
    rows = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "" or line.startswith("k="):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                print(f"[WARN] Ungültige Zeile: {line.strip()}")
                continue
            rows.append([float(x) for x in parts[:5]])
    data = np.array(rows)
    t = data[:, 0]
    m = data[:, 1]
    err_lower = -data[:, 2]
    err_upper = data[:, 3]

    # Energies auslesen (optional, falls vorhanden)
    energies_path = os.path.join(os.path.dirname(filepath), "energies.dat")
    G = V = G_err = V_err = None
    if os.path.exists(energies_path):
        with open(energies_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    if parts[1] == 'G':
                        G = float(parts[2])
                        G_err = float(parts[4])
                    elif parts[1] == 'V':
                        V = float(parts[2])
                        V_err = float(parts[4])

    # Indizes für ganzzahliges und halbzahliges t
    is_int = np.abs(t % 1) < 0.01
    is_half = np.abs(t % 1 - 0.5) < 0.01

    # Plot
    plt.errorbar(t[is_half], m[is_half],
                 yerr=[err_lower[is_half], err_upper[is_half]],
                 fmt='o', color='blue', ecolor='black', capsize=3, label='halbzahlig')

    plt.errorbar(t[is_int], m[is_int],
                 yerr=[err_lower[is_int], err_upper[is_int]],
                 fmt='o', color='red', ecolor='black', capsize=3, label='ganzzahlig')

    if G is not None:
        plt.axhline(G, color='blue', linestyle='--', label='G')
        plt.axhspan(G - G_err, G + G_err, color='blue', alpha=0.2)

    if V is not None:
        plt.axhline(V, color='red', linestyle='--', label='V')
        plt.axhspan(V - V_err, V + V_err, color='red', alpha=0.2)

    plt.xlabel('t/a')
    plt.ylabel('Effektive Masse')
    y_all_lower = np.concatenate([m[is_int] - err_lower[is_int], m[is_half] - err_lower[is_half]])
    y_all_upper = np.concatenate([m[is_int] + err_upper[is_int], m[is_half] + err_upper[is_half]])

    ymin = np.min(y_all_lower)
    ymax = np.max(y_all_upper)
    padding = 0.05 * (ymax - ymin)
    plt.ylim(ymin - padding, ymax + padding)
    plt.title('Effektive Masse vs. Zeit')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Plot speichern
    filename = os.path.splitext(os.path.basename(filepath))[0] + ".png"
    outpath = os.path.join(plot_dir, filename)
    plt.savefig(outpath)
    html_outpath = outpath.replace(".png", ".html")
    save_plotly_html(t, m, err_lower, err_upper, is_half, is_int, "Effektive Masse vs. Zeit", html_outpath)
    plt.close()

    # Referenzfehler bestimmen (Median)
    reference_err = np.median(err_upper)

    # Indizes auswählen nach dynamischer Regel
    is_int = (np.abs(t % 1) < 0.01) & (t >= min_t) & (err_upper <= err_factor * reference_err)
    is_half = (np.abs(t % 1 - 0.5) < 0.01) & (t >= min_t) & (err_upper <= err_factor * reference_err)

    # ---- Zusätzlicher zugeschnittener Plot ---- #
    plt.errorbar(t[is_half], m[is_half],
                yerr=[err_lower[is_half], err_upper[is_half]],
                fmt='o', color='blue', ecolor='black', capsize=3, label='halbzahlig')

    plt.errorbar(t[is_int], m[is_int],
                yerr=[err_lower[is_int], err_upper[is_int]],
                fmt='o', color='red', ecolor='black', capsize=3, label='ganzzahlig')

    if G is not None:
        plt.axhline(G, color='blue', linestyle='--', label='G')
        plt.axhspan(G - G_err, G + G_err, color='blue', alpha=0.2)

    if V is not None:
        plt.axhline(V, color='red', linestyle='--', label='V')
        plt.axhspan(V - V_err, V + V_err, color='red', alpha=0.2)

    plt.xlabel('t/a (zugeschnitten)')
    plt.ylabel('Effektive Masse')
    plt.title('Effektive Masse vs. Zeit (zugeschnitten)')
    plt.grid(True)
    plt.legend()

    # Dynamischer y-Bereich für zugeschnittene Werte
    y_all_lower_cut = np.concatenate([m[is_int] - err_lower[is_int], m[is_half] - err_lower[is_half]])
    y_all_upper_cut = np.concatenate([m[is_int] + err_upper[is_int], m[is_half] + err_upper[is_half]])
    ymin_cut = np.min(y_all_lower_cut)
    ymax_cut = np.max(y_all_upper_cut)
    padding_cut = 0.05 * (ymax_cut - ymin_cut)
    plt.ylim(ymin_cut - padding_cut, ymax_cut + padding_cut)
    plt.tight_layout()

    # Speichern des zugeschnittenen Plots
    cut_filename = os.path.splitext(os.path.basename(filepath))[0] + "_cut.png"
    cut_outpath = os.path.join(plot_dir, cut_filename)
    plt.savefig(cut_outpath)
    plt.close()
