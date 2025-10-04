import numpy as np
import matplotlib.pyplot as plt

def V_cornell(r, V0, alpha, sigma):
    return V0 + alpha / r + sigma * r

# r-Werte (ohne 0 wegen Division durch r)
r_vals = np.linspace(0.1, 5, 500)

# Varianten zum Testen
parameter_sets = [
    {"label": r"$\alpha>0, \sigma>0$", "alpha": 0.3,  "sigma": 0.5,  "color": "blue"},
    {"label": r"$\alpha<0, \sigma>0$", "alpha": -0.3, "sigma": 0.5,  "color": "green"},
    {"label": r"$\alpha>0, \sigma<0$", "alpha": 0.3,  "sigma": -0.5, "color": "orange"},
    {"label": r"$\alpha<0, \sigma<0$", "alpha": -0.3, "sigma": -0.5, "color": "red"},
    {"label": r"$\alpha>0, \sigma=0$", "alpha": 0.3,  "sigma": 0.0,  "color": "purple"},
    {"label": r"$\alpha=0, \sigma>0$", "alpha": 0.0,  "sigma": 0.5,  "color": "brown"},
]

plt.figure(figsize=(7, 5))

for p in parameter_sets:
    V = V_cornell(r_vals, V0=0.0, alpha=p["alpha"], sigma=p["sigma"])
    plt.plot(r_vals, V, label=p["label"], color=p["color"])

plt.xlabel(r"$r$")
plt.ylabel(r"$V(r)$")
plt.title("Cornell-Potenziale für verschiedene Parameter")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("cornell_varianten.png", dpi=300, bbox_inches="tight")
plt.close()

