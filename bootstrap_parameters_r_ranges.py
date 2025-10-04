import os
import numpy as np
import math
import sys
import re
from scipy.optimize import minimize
from scipy.optimize import least_squares
from numpy.random import normal

# Fitted parameters (from Cornell fit):
# V0, alpha, sigma

# Derived from fit parameters:
# r0, r0*sqrt(sigma), 1/r0^2, V(r0), r/r0, r0*(V(r) - V(r0))

# Input from external data:
# t0, t0_err, w0, w0_err

# Derived from t0, w0 and sigma:
# sqrt(8t0), 1/sqrt(8t0)^2, sqrt(8t0)*sqrt(sigma)
# sqrt(8)w0, 1/(sqrt(8)w0)^2, sqrt(8)w0*sqrt(sigma)

# Bootstrap errors:
# Computed for all above quantities

# Fit quality:
# chi², ndf, chi²/ndf, AIC, AIC weight

debug = True  # Debug nur bei erster Iteration
CHECK_ERROR_LIMIT = False
REL_ERR_LIMIT = 1
n_bootstrap = 500

# === Parse input and ensemble name ===
if len(sys.argv) == 4:
    ENSEMBLE = sys.argv[1]
    smearing = int(sys.argv[2])
    rmax = int(sys.argv[3])
else:
    with open("input_ensemble.txt") as f:
        ENSEMBLE = f.readline().strip()
        smearing = int(f.readline())
        rmax = int(f.readline())

match = re.search(r"l(\d+)t\d+_b(\d+\.\d+)", ENSEMBLE)
if match:
    N_latt = int(match.group(1))
    beta = float(match.group(2))
else:
    raise ValueError(f"Could not extract N_latt and beta from ENSEMBLE name: '{ENSEMBLE}'")

ensemble = f"{ENSEMBLE}_sm{smearing}_rmax{rmax:02d}"
base_dir = os.path.expanduser(f"~/data/analysis/{ensemble}/results_{ensemble}")

input_dir = f"{base_dir}/mean_potential"      
bootstrap_dir = f"{base_dir}/bootstrap_parameters"
os.makedirs(bootstrap_dir, exist_ok=True)

t0_w0_data = {
    (10, 2.65): {"t0": (1.2321, 0.0027), "w0": (1.1455, 0.0015)},
    (10, 2.75): {"t0": (1.5709, 0.0035), "w0": (1.2848, 0.0019)},
    (16, 2.75): {"t0": (1.5702, 0.0013), "w0": (1.28442, 0.00067)},
    (14, 2.80): {"t0": (1.7655, 0.0019), "w0": (1.3606, 0.0010)},
    (14, 2.85): {"t0": (1.9798, 0.0026), "w0": (1.4394, 0.0013)},
    (14, 2.90): {"t0": (2.2028, 0.0031), "w0": (1.5171, 0.0014)},
    (16, 2.90): {"t0": (2.2085, 0.0025), "w0": (1.5192, 0.0012)},
    (14, 2.95): {"t0": (2.4687, 0.0044), "w0": (1.6067, 0.0019)},
    (16, 2.95): {"t0": (2.4667, 0.0033), "w0": (1.6062, 0.0015)},
    (16, 3.00): {"t0": (2.7433, 0.0043), "w0": (1.6923, 0.0019)},
    (16, 3.05): {"t0": (3.0379, 0.0048), "w0": (1.7809, 0.0019)},
    (18, 3.05): {"t0": (3.0435, 0.0038), "w0": (1.7828, 0.0016)},
    (18, 3.10): {"t0": (3.3840, 0.0052), "w0": (1.8799, 0.0020)},
    (18, 3.15): {"t0": (3.7456, 0.0076), "w0": (1.9777, 0.0028)},
    (18, 3.20): {"t0": (4.1508, 0.0083), "w0": (2.0811, 0.0030)},
}

# Stores results of initial fits for comparison between fit methods
null_fit_results = []

# === Define fit functions and chi² ===

def cornell_potential(r, V0, alpha, sigma):
    return V0 + alpha / r + sigma * r

def residuals_fn(params, r, V_vals, errors):
    V_model = cornell_potential(r, *params)
    return (V_model - V_vals) / errors

def chi2_fn(params, r, V, err):
    scale = 1000
    V_model = cornell_potential(r, *params) * scale
    V_data  = V * scale
    err_scaled = err * scale
    return np.sum(((V_data - V_model) / err_scaled) ** 2)

def run_bootstrap(ira, irb, use_bounds=True):

    # Lists for input data (r values and potentials)
    r_vals = []
    means = []
    errors = []

    # Lists to store bootstrap results (one entry per sample)
    params = []
    r0_samples = []
    r0sig_samples = []
    inv_r0sq_samples = []
    chi2_samples = []
    ndf_samples = []
    r_over_r0_samples = []
    scaled_potential_samples = []
    inv_sqrt8t0_sq_samples = []
    sqrt8t0_sig_samples = []
    inv_sqrt8w0_sq_samples = []
    sqrt8w0_sig_samples = []

    # Counters for diagnostics
    n_fits_ok = 0
    n_skipped_t0w0 = 0

    try:
        t0_val, t0_err = t0_w0_data[(N_latt, beta)]["t0"]
    except KeyError:
        raise ValueError(f"No t0 value found for (N_latt={N_latt}, beta={beta})")
    try:
        w0_val, w0_err = t0_w0_data[(N_latt, beta)]["w0"]
    except KeyError:
        raise ValueError(f"No w0 value found for (N_latt={N_latt}, beta={beta})")

    # === Load measurement points ===
    print(f"\n→ Loading input data for r = {ira} to {irb}")
    output_file = os.path.join(bootstrap_dir, f"bootstrap_{ira}_{irb}.dat")
    # === Load means and errors from files ===
    r_vals = []
    means = []
    errors = []

    for r in range(ira, irb + 1):
        radius = f"r{r:02d}"
        filepath = os.path.join(input_dir, f"mean_{radius}.dat")
        if not os.path.isfile(filepath):
            print(f"File missing: {filepath}")
            continue
        with open(filepath) as f:
            lines = f.readlines()
            parts = lines[1].strip().split()
            V_mean = float(parts[0])
            # V_err = float(parts[1])  # err_total
            V_err = float(parts[2])  # err_stat
            # V_err = float(parts[3])  # err_sys
            r_vals.append(r)
            means.append(V_mean)
            errors.append(V_err)

    r_vals = np.array(r_vals)
    means = np.array(means)
    errors = np.array(errors)

    if len(r_vals) == 0 or len(means) == 0 or len(errors) == 0:
        raise ValueError("No input values found. Check input_dir and ira–irb.")

    for r_val, mean, err in zip(r_vals, means, errors):
        print(f"r{r_val:02d}: mean = {mean:.6f}, error = {err:.6f}")

    # === Bootstrap with Gaussian sampling ===

    # Initial guess and parameter bounds for Cornell fit
    x0 = [0.1, -0.2, 0.01]
    c=1.65
    if use_bounds:
        bounds = ([-np.inf, 1e-12 - c, 1e-12], [np.inf, 1e-12, np.inf])
    else:
        bounds = (-np.inf, np.inf)

    # === Perform null-sample fit ===
    try:
        methods = ["trf", "dogbox"]
        results = {}

        for m in methods:
            res0 = least_squares(
                fun=residuals_fn,
                x0=x0,
                args=(r_vals, means, errors),
                method=m,
                xtol=1e-8,
                bounds=bounds
            )
            if res0.success:
                results[m] = res0.x
            else:
                results[m] = None

        if all(results[m] is not None for m in methods):
            base = results["trf"]
            diffs = {m: np.array(results[m]) - base for m in methods}
            null_fit_results.append((ira, irb, results, diffs))

        V0_null, alpha_null, sigma_null = res0.x
    except Exception as e:
        print("Error during null-sample fit:", e)
        exit()

    # Central fit parameters from the initial null fit
    V0_null, alpha_null, sigma_null = res0.x

    # === Run bootstrap resampling === 
    for _ in range(n_bootstrap):
        V_sample = normal(loc=means, scale=errors)

        try:
            # res = minimize(
            #     chi2_fn,
            #     x0=x0,
            #     args=(r_vals, V_sample, errors),
            #     method="Powell",
            #     options={"gtol": 1e-6, "maxiter": 1000, "disp": True}
            # )
            res = least_squares(
                fun=residuals_fn,
                x0=res0.x,
                args=(r_vals, V_sample, errors),
                method="trf",       # robust, auch mit Bounds möglich
                xtol=1e-8,
                bounds=bounds
            )

            if not res.success or np.array_equal(res.x, x0):
                if debug:
                    reason = res.message if not res.success else "Initial guess returned, fit ignored"
                    print("Fit skipped:", reason)
                continue

            # Erfolgreicher Fit
            n_fits_ok += 1
            # if debug:
            #     print("Fit ok → popt:", res.x)
                
            popt = res.x
            # chi2 = chi2_fn(popt, r_vals, V_sample, errors)
            # ndf = len(r_vals) - 3
            # if chi2 / ndf < 2.0:
            #     n_fits_chi2_pass += 1
            # else:
            #     continue

            params.append(popt)
            V0, alpha, sigma = popt
            if (c + alpha) / sigma <= 0:
                reason = []
                if sigma <= 0:
                    reason.append("sigma ≤ 0")
                if c + alpha <= 0:
                    reason.append("c + alpha ≤ 0")
                print(f"Invalid r0 → alpha: {alpha:.4f}, sigma: {sigma:.4f} | Reason: {', '.join(reason)}")
                continue

            # === Compute and store derived bootstrap quantities ===

            # Derived quantities from fitted parameters
            r0_sample = np.sqrt((c + alpha) / sigma)
            r0sig_sample = r0_sample * np.sqrt(sigma)
            inv_r0sq_sample = 1 / r0_sample**2
            V_r0_sample = cornell_potential(r0_sample, *popt)

            # Derived quantities from t0 and w0
            t0_sample = normal(loc=t0_val, scale=t0_err)
            w0_sample = normal(loc=w0_val, scale=w0_err)
            if t0_sample <= 0 or w0_sample <= 0:
                n_skipped_t0w0 += 1
                continue

            sqrt8t0 = math.sqrt(8 * t0_sample)
            sqrt8w0 = math.sqrt(8) * w0_sample

            sqrt8t0_sig_sample = sqrt8t0 * np.sqrt(sigma)
            sqrt8w0_sig_sample = sqrt8w0 * np.sqrt(sigma)

            inv_sqrt8t0_sq_sample = 1 / (sqrt8t0 ** 2)         
            inv_sqrt8w0_sq_sample = 1 / (sqrt8w0 ** 2)
                       
            # Chi² and rescaled coordinates
            chi2 = chi2_fn(popt, r_vals, V_sample, errors)
            ndf = len(r_vals) - len(popt)
            

            r_over_r0_sample = [r / r0_sample for r in r_vals]
            scaled_potential_sample = [r0_sample * (V - V_r0_sample) for V in V_sample]

            # r_over_r0_sample = []
            # scaled_potential_sample = []

            # for r_val, V_val in zip(r_vals, V_sample):
            #     r_over_r0_sample.append(r_val / r0_sample)
            #     scaled_potential_sample.append(r0_sample * (V_val - V_r0_sample))

            # Store bootstrap from fitted parameters 
            r0_samples.append(r0_sample)
            r0sig_samples.append(r0sig_sample)
            inv_r0sq_samples.append(inv_r0sq_sample)
            chi2_samples.append(chi2)
            ndf_samples.append(ndf)
            r_over_r0_samples.append(r_over_r0_sample)
            scaled_potential_samples.append(scaled_potential_sample)

            # Store bootstrap from t0 and w0 
            inv_sqrt8t0_sq_samples.append(inv_sqrt8t0_sq_sample)
            sqrt8t0_sig_samples.append(sqrt8t0_sig_sample)
            inv_sqrt8w0_sq_samples.append(inv_sqrt8w0_sq_sample)
            sqrt8w0_sig_samples.append(sqrt8w0_sig_sample)

        except:
            continue

    print(f"Fits successful: {n_fits_ok} out of {n_bootstrap}")
    print(f"Valid r0 values: {len(r0_samples)}")
    print(f"Excluded due to (c + alpha) / sigma ≤ 0: {n_fits_ok - len(r0_samples)}")
    print(f"Skipped samples due to non-positive t0 or w0: {n_skipped_t0w0}")

    # === Calculate bootstrap errors ===

    # Fitted Cornell parameters
    params = np.array(params)
    print("params.shape:", params.shape)
    r0_samples = np.array(r0_samples)
    r0sig_samples = np.array(r0sig_samples)
    inv_r0sq_samples = np.array(inv_r0sq_samples)

    # t0 and w0 based observables
    sqrt8t0_sig_samples = np.array(sqrt8t0_sig_samples)
    sqrt8w0_sig_samples = np.array(sqrt8w0_sig_samples)

    inv_sqrt8t0_sq_samples = np.array(inv_sqrt8t0_sq_samples)
    inv_sqrt8w0_sq_samples = np.array(inv_sqrt8w0_sq_samples)
    
    # Rescaled coordinates and potential
    r_over_r0_samples = np.array(r_over_r0_samples)
    scaled_potential_samples = np.array(scaled_potential_samples)

    # Fit quality metrics
    chi2_samples = np.array(chi2_samples)
    ndf_samples = np.array(ndf_samples)

    # === Save bootstrap fit parameters ===
    subdir = os.path.join(bootstrap_dir, f"bootstrap_{ira}_{irb}")
    os.makedirs(subdir, exist_ok=True)
    bootstrap_file_params = os.path.join(subdir, f"bootstrapsamples_{ira}_{irb}.dat")

    with open(bootstrap_file_params, "w") as f_out:
        f_out.write("# sample    V0         alpha      sigma      r0         r0sig      inv_r0sq   sqrt8t0_sig  inv_sqrt8t0_sq  sqrt8w0_sig  inv_sqrt8w0_sq\n")
        for i in range(len(r0_samples)):
            f_out.write(f"{i:6d}  {params[i,0]:10.6f}  {params[i,1]:10.6f}  {params[i,2]:10.6f}  "
                        f"{r0_samples[i]:10.6f}  {r0sig_samples[i]:10.6f}  {inv_r0sq_samples[i]:10.6f}  "
                        f"{sqrt8t0_sig_samples[i]:10.6f}  {inv_sqrt8t0_sq_samples[i]:10.6f}  "
                        f"{sqrt8w0_sig_samples[i]:10.6f}  {inv_sqrt8w0_sq_samples[i]:10.6f}\n")

    # Bootstrap errors for Cornell parameters
    V0_boot_err, alpha_boot_err, sigma_boot_err = np.std(params, axis=0, ddof=1)

    # Derived central values from null fit
    r0 = np.sqrt((c + alpha_null) / sigma_null)
    r0sig = r0 * np.sqrt(sigma_null)
    inv_r0sq = 1 / r0**2

    sqrt8t0 = math.sqrt(8 * t0_val)
    sqrt8t0_sig = sqrt8t0 * sigma_null
    inv_sqrt8t0_sq = 1 / (sqrt8t0 ** 2)

    sqrt8w0 = math.sqrt(8) * w0_val
    sqrt8w0_sig = sqrt8w0 * sigma_null
    inv_sqrt8w0_sq = 1 / (sqrt8w0 ** 2)

    # Bootstrap errors for derived quantities
    r0_boot_err = np.std(r0_samples, ddof=1)
    r0sig_boot_err = np.std(r0sig_samples, ddof=1)
    inv_r0sq_boot_err  = np.std(inv_r0sq_samples, ddof=1)

    # Bootstrap errors for rescaled potential
    r_over_r0_err  = np.std(r_over_r0_samples, axis=0, ddof=1)
    scaled_potential_err  = np.std(scaled_potential_samples, axis=0, ddof=1)

    # Chi² values from null fit and bootstrap
    ndf_null = len(r_vals) - len(res0.x)
    chi2_null = chi2_fn(res0.x, r_vals, means, errors)
    chi2_boot_err = np.std(chi2_samples, ddof=1)
    chi2_per_dof = chi2_null / ndf_null

    inv_sqrt8t0_sq_boot_err = np.std(inv_sqrt8t0_sq_samples, ddof=1)
    sqrt8t0_sig_boot_err = np.std(sqrt8t0_sig_samples, ddof=1)

    inv_sqrt8w0_sq_boot_err = np.std(inv_sqrt8w0_sq_samples, ddof=1)
    sqrt8w0_sig_boot_err = np.std(sqrt8w0_sig_samples, ddof=1)

    # === Evaluate AIC weight ===
    k = 3
    n = len(r_vals)
    ntot = rmax
    aic = chi2_null + 2 * k + 2 * (ntot - n)

    fails = []
    def rel_err(err, val):
        return float('inf') if abs(val) == 0 else abs(err)/abs(val)

    if CHECK_ERROR_LIMIT:
        if rel_err(alpha_boot_err, alpha_null) > REL_ERR_LIMIT:
            fails.append("alpha")
        if rel_err(sigma_boot_err, sigma_null) > REL_ERR_LIMIT:
            fails.append("sigma")
        if rel_err(V0_boot_err, V0_null) > REL_ERR_LIMIT:
            fails.append("V0")


    print(f"Debug Range {ira}-{irb}:")
    print(f"V0 = {V0_null:.4f} ± {V0_boot_err:.4f}; rel = {rel_err(V0_boot_err, V0_null):.2f}")
    print(f"alpha = {alpha_null:.4f} ± {alpha_boot_err:.4f}; rel = {rel_err(alpha_boot_err, alpha_null):.2f}")
    print(f"sigma = {sigma_null:.4f} ± {sigma_boot_err:.4f}; rel = {rel_err(sigma_boot_err, sigma_null):.2f}")

    if fails:
        print(f"Range {ira}-{irb} → weight = 0 (high error in {', '.join(fails)}): "
            f"alpha_err={alpha_boot_err:.4g} (rel {rel_err(alpha_boot_err, alpha_null):.2f}), "
            f"sigma_err={sigma_boot_err:.4g} (rel {rel_err(sigma_boot_err, sigma_null):.2f}), "
            f"V0_err={V0_boot_err:.4g} (rel {rel_err(V0_boot_err, V0_null):.2f})")
        weight = 0.0
    else:
        weight = math.exp(-aic/2)

    V_r0_null = cornell_potential(r0, V0_null, alpha_null, sigma_null)

    r_over_r0_null = []
    scaled_potential_null = []

    for r_val, V_val in zip(r_vals, means):
        r_over_r0_null.append(r_val / r0)
        scaled_potential_null.append(r0 * (V_val - V_r0_null))


    r0_prop_err = 0.5 * r0 * np.sqrt((alpha_boot_err**2 + sigma_boot_err**2) / (c + alpha_null)**2)
    r0sig_prop_err = np.sqrt(
        (np.sqrt(sigma_null) * r0_prop_err) ** 2 +
        ((r0 / (2 * np.sqrt(sigma_null))) * sigma_boot_err) ** 2)
    inv_r0sq_prop_err = 2 * r0_boot_err / r0**3


    # === Save fit results and bootstrap samples ===
    with open(output_file, "w") as f_out:
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
        f_out.write(header)

        line = (
            f"{'':<4} {ira:6} {irb:6} "
            f"{V0_null:14.10f} {V0_boot_err:14.10f} "
            f"{alpha_null:14.10f} {alpha_boot_err:14.10f} "
            f"{sigma_null:14.10f} {sigma_boot_err:14.10f} "
            f"{r0:14.10f} {r0_boot_err:14.10f} "       
            f"{r0sig:14.10f} {r0sig_boot_err:14.10f} "
            f"{inv_r0sq:18.10f} {inv_r0sq_boot_err:18.10f} "
            f"{sqrt8t0_sig:14.10f} {sqrt8t0_sig_boot_err:14.10f} "
            f"{inv_sqrt8t0_sq:18.10f} {inv_sqrt8t0_sq_boot_err:18.10f} "
            f"{sqrt8w0_sig:14.10f} {sqrt8w0_sig_boot_err:14.10f} "
            f"{inv_sqrt8w0_sq:18.10f} {inv_sqrt8w0_sq_boot_err:18.10f} "
            f"{chi2_null:12.6f} {n:6} {k:6} {ntot:8} "
            f"{weight:14.10f}\n"
        )
        f_out.write(line)

    # === Save r/r0 and scaled potential ===
    outfile_r = os.path.join(bootstrap_dir, f"r_over_r0_{ira}_{irb}.dat")
    outfile_V = os.path.join(bootstrap_dir, f"scaled_potential_{ira}_{irb}.dat")

    with open(outfile_r, "w") as f_r, open(outfile_V, "w") as f_v:
        f_r.write("# r_label   r_over_r0    r_over_r0_err\n")
        f_v.write("# r_label   scaled_pot   scaled_pot_err\n")

        for i, r in enumerate(r_vals):
            r_label = f"r{r:02d}"
            f_r.write(f"{r_label:<8} {r_over_r0_null[i]:12.6f} {r_over_r0_err[i]:12.6f}\n")
            f_v.write(f"{r_label:<8} {scaled_potential_null[i]:12.6f} {scaled_potential_err[i]:12.6f}\n")

# === Loop over all fit ranges ===
rmin = 2 if smearing == 2 else 1
print(f"Using rmin = {rmin} for smearing = {smearing}")
for ira in range(rmin, rmax - 2):
    for irb in range(ira + 3, rmax + 1):
        print(f"Starting bootstrap for ira = {ira}, irb = {irb}")
        try:
            run_bootstrap(ira, irb, use_bounds=True) #True with bounderies, False without boundaries 
        except Exception as e:
            print(f"Error at ira = {ira}, irb = {irb}: {e}")


# === Compare fit methods ===
print("\n=== Comparison of all null-sample parameters ===")
for ira, irb, results, diffs in null_fit_results:
    print(f"\nira={ira}, irb={irb}")
    for m in results:
        print(f"{m}: {results[m]}   Δ to trf: {diffs[m]}")
print("done")
