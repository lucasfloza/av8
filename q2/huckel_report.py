
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from matplotlib.lines import Line2D

# Parameters
ALPHA_C = 0.0
BETA_CC = -2.5
ALPHA_N = 0.0
BETA_CN = -2.5
BETA_NN = -2.5
N_ELECTRONS_PER_SITE = 1

NUM_SITES = 10
EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,0),
    (5,6),(6,7),(7,8),(8,9),(9,4)
]
SITE_LABELS = [f"{i}" for i in range(1, NUM_SITES+1)]

def build_huckel_matrix(atom_types: List[str]) -> np.ndarray:
    H = np.zeros((NUM_SITES, NUM_SITES), dtype=float)
    for i, t in enumerate(atom_types):
        H[i,i] = ALPHA_C if t == 'C' else ALPHA_N
    for i,j in EDGES:
        ti, tj = atom_types[i], atom_types[j]
        if ti == 'C' and tj == 'C':
            beta = BETA_CC
        elif ti == 'N' and tj == 'N':
            beta = BETA_NN
        else:
            beta = BETA_CN
        H[i,j] = H[j,i] = beta
    return H

def diagonalize(H: np.ndarray):
    E, C = np.linalg.eigh(H)
    idx = np.argsort(E)
    return E[idx], C[:, idx]

def fill_mos(E: np.ndarray, n_sites: int = NUM_SITES):
    n_e = n_sites * N_ELECTRONS_PER_SITE
    n_occ = n_e // 2
    occ = np.zeros_like(E, dtype=int)
    occ[:n_occ] = 2
    homo = n_occ - 1
    lumo = n_occ if n_occ < len(E) else None
    return homo, lumo, occ

def pi_populations_and_bond_orders(C: np.ndarray, occ: np.ndarray):
    occ_cols = np.where(occ == 2)[0]
    Cocc = C[:, occ_cols]
    pop = 2.0 * np.sum(Cocc**2, axis=1)
    bond_orders = {}
    for (i,j) in EDGES:
        bond_orders[(i+1,j+1)] = float(2.0 * np.sum(Cocc[i,:] * Cocc[j,:]))
    return pop, bond_orders

def spectrum_figure(E: np.ndarray, homo: int, lumo: int, title: str, savepath: str):
    eps = (E - ALPHA_C) / BETA_CC
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ypad = 0.35
    for k, ek in enumerate(eps):
        ax.hlines(ek, 0.2, 0.8, linewidth=2, linestyles="-" if k<=homo else "--")
        if k <= homo:
            ax.plot([0.45, 0.55], [ek, ek], "o", ms=6)
        else:
            ax.plot(0.5, ek, "o", ms=6, mfc="none")
        label = f"MO{k+1}"
        if k == homo: label += " (HOMO)"
        if k == lumo: label += " (LUMO)"
        ax.text(1.02, ek, f"{label}\nε={ek:.3f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", clip_on=False, fontsize=9)
    if lumo is not None:
        y1, y2 = eps[homo], eps[lumo]
        ax.vlines(0.18, min(y1,y2), max(y1,y2), linestyles=":", colors="k")
        ax.text(0.17, (y1+y2)/2, f"Δε = {abs(y2-y1):.3f}", va="center", ha="right", fontsize=9)
    ax.set_ylabel("Energia reduzida ε = (E − α_C)/β_CC")
    ax.set_xticks([])
    ax.set_ylim(min(eps)-ypad, max(eps)+ypad)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    legend_elems = [
        Line2D([0],[0], marker='o', linestyle='-', label='ocupado (2e⁻)'),
        Line2D([0],[0], marker='o', mfc='none', linestyle='--', label='desocupado (0e⁻)')
    ]
    legend = ax.legend(handles=legend_elems, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                       ncol=2, frameon=False)
    fig.subplots_adjust(right=0.80, bottom=0.26)
    fig.savefig(savepath, dpi=300, bbox_inches="tight", bbox_extra_artists=[legend], pad_inches=0.2)
    plt.close(fig)

def bar_map(C: np.ndarray, idx: int, title: str, savepath: str):
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.axhline(0.0, linewidth=1)
    ax.bar(np.arange(NUM_SITES), C[:, idx])
    ax.set_xticks(np.arange(NUM_SITES))
    ax.set_xticklabels(SITE_LABELS)
    ax.set_ylabel("coeficiente c_i")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def numbering_figure(atom_types: List[str], title: str, savepath: str):
    theta = np.linspace(0, 2*np.pi, NUM_SITES, endpoint=False)
    x = 1.6*np.cos(theta)
    y = 1.0*np.sin(theta)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for (i,j) in EDGES:
        ax.plot([x[i], x[j]],[y[i], y[j]], '-')
    for i in range(NUM_SITES):
        ax.plot(x[i], y[i], 'o', ms=8)
        label = f"{i+1}"
        if atom_types[i] == 'N':
            label += " (N)"
        ax.text(x[i], y[i]+0.14, label, ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def run_molecule(name: str, atom_types: List[str], prefix: str):
    H = build_huckel_matrix(atom_types)
    E, C = diagonalize(H)
    homo, lumo, occ = fill_mos(E, NUM_SITES)
    spectrum_figure(E, homo, lumo, f"Espectro de energias — {name}", f"{prefix}_spectrum.png")
    eps = (E - ALPHA_C)/BETA_CC
    df_eig = pd.DataFrame({"MO": np.arange(1, len(E)+1), "E/β": eps, "E (eV)": E, "ocupação": occ})
    df_eig.to_csv(f"{prefix}_eigs.csv", index=False)
    pop, bonds = pi_populations_and_bond_orders(C, occ)
    pd.DataFrame({"sitio": np.arange(1, NUM_SITES+1), "tipo": atom_types, "qi": pop}).to_csv(f"{prefix}_pop.csv", index=False)
    pd.DataFrame({"i":[ij[0] for ij in bonds.keys()],"j":[ij[1] for ij in bonds.keys()],"pij": list(bonds.values())}).to_csv(f"{prefix}_pij.csv", index=False)
    bar_map(C, homo, f"Mapa HOMO — {name}", f"{prefix}_homo_map.png")
    bar_map(C, lumo, f"Mapa LUMO — {name}", f"{prefix}_lumo_map.png")
    return E,C,homo,lumo,occ

if __name__ == "__main__":
    naph = ['C']*NUM_SITES
    quin = ['N'] + ['C']*(NUM_SITES-1)
    quino = ['N','C','C','C','C','N','C','C','C','C']  # Ns nos sítios 1 e 6 (1-based)

    numbering_figure(naph,  "Figura 1 — Naftaleno numerado", "./q2/fig1_numbering_naphthalene.png")
    numbering_figure(quin,  "Figura 2 — Quinolina (mono-aza)", "./q2/fig2_numbering_quinoline.png")
    numbering_figure(quino, "Figura 3 — Quinoxalina (1,4-diaza)", "./q2/fig3_numbering_quinoxaline.png")

    run_molecule("Naftaleno", naph, "./q2/fig4_naphthalene")
    run_molecule("Quinolina (mono-aza)", quin, "./q2/fig6_quinoline")
    run_molecule("Quinoxalina (1,4-diaza)", quino, "./q2/fig6b_quinoxaline")
