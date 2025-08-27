# -*- coding: utf-8 -*-
"""
Hückel (π, pz) para o 2H-Pirano.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Tuple, Dict, List
from collections import OrderedDict
from matplotlib.lines import Line2D

# Numeração e conectividade (0: O, 1..5: C)
ATOM_LABELS = ["O(1)", "C(2)", "C(3)", "C(4)", "C(5)", "C(6)"]
RING_EDGES = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)]  # ligações do anel (1–2–3–4–5–6–1)

# -------------------------
# Constantes
# -------------------------
ALPHA_CARBON: float = 0.0              # α (referência de energia dos C)
BETA_CC: float = -1.0                  # β (C–C), negativo
H_OXYGEN: float = 1.5                  # h_O  (α_O = α + h_O * β)
K_CO: float = 1.0                      # k_CO (β_CO = k_CO * β)
N_PI_ELECTRONS: int = 6                # nº de elétrons π (2H-pirano)

# -------------------------
# 1 - Determinante secular
# -------------------------
def print_secular_determinant():
    try:
        import sympy as sp
    except Exception as e:
        print("[1] Determinante secular: sympy indisponível.", e)
        return

    eps = sp.symbols("eps")
    alpha_C, beta_CC, h_O, k_CO = sp.symbols("alpha_C beta_CC h_O k_CO")

    matriz = sp.zeros(6)
    matriz[0,0] = alpha_C + h_O*beta_CC
    for i in range(1,6):
        matriz[i,i] = alpha_C
    for i,j in RING_EDGES:
        matriz[i,j] = matriz[j,i] = (k_CO*beta_CC) if (0 in (i,j)) else beta_CC

    E = alpha_C + eps*beta_CC
    det_eps = sp.factor((E*sp.eye(6) - matriz).det())
    # remove o fator beta^6 para mostrar só P(ε)
    P_eps = sp.factor(sp.simplify(det_eps / (beta_CC**6)))

    print("\n[1] Determinante secular (forma reduzida): P(ε) =")
    print(P_eps)

    subs = {alpha_C: ALPHA_CARBON, beta_CC: BETA_CC, h_O: H_OXYGEN, k_CO: K_CO}
    print("\n[1] P(ε) com parâmetros atuais:")
    print(sp.factor(P_eps.subs(subs)))

# -------------------------
# Construção da matriz com o método de Hückel
# -------------------------
def build_huckel_matrix() -> np.ndarray:

    huckel_matrix = np.zeros((6, 6), dtype=float)

    # termos on-site (diagonais)
    huckel_matrix[0, 0] = ALPHA_CARBON + H_OXYGEN * BETA_CC  # O(1)
    for atom_index in range(1, 6):                            # C(2) ... C(6)
        huckel_matrix[atom_index, atom_index] = ALPHA_CARBON

    # acoplamentos de primeiros vizinhos (fora da diagonal)
    for atom_i, atom_j in RING_EDGES:
        coupling = (K_CO * BETA_CC) if (0 in (atom_i, atom_j)) else BETA_CC
        huckel_matrix[atom_i, atom_j] = coupling
        huckel_matrix[atom_j, atom_i] = coupling

    return huckel_matrix

# -------------------------
# Autovalores/autovetores via QR com shift simples
# -------------------------
def qr_eig(matrix: np.ndarray,
           tolerance: float = 1e-12,
           max_iterations: int = 5000) -> tuple[np.ndarray, np.ndarray]:

    current_matrix = matrix.copy().astype(float)
    dimension = current_matrix.shape[0]
    accumulated_Q = np.eye(dimension)

    for iteration in range(max_iterations):
        off_diagonal_norm = np.linalg.norm(current_matrix - np.diag(np.diag(current_matrix)))
        if off_diagonal_norm < tolerance:
            break

        shift = current_matrix[-1, -1]
        q_factor, r_factor = np.linalg.qr(current_matrix - shift * np.eye(dimension))
        current_matrix = r_factor @ q_factor + shift * np.eye(dimension)
        accumulated_Q = accumulated_Q @ q_factor

    eigenvalues_unsorted = np.diag(current_matrix).copy()
    order_indices = np.argsort(eigenvalues_unsorted)
    eigenvalues = eigenvalues_unsorted[order_indices]
    eigenvectors = accumulated_Q[:, order_indices]

    return eigenvalues, eigenvectors

# -------------------------
# 2 - Calculando níveis de energia e plotando diagrama de níveis
# -------------------------
def calculating_energy_levels(eigenvalues):
    # energias reduzidas ε = (E − α)/β
    reduced_energies: np.ndarray = (eigenvalues - ALPHA_CARBON) / BETA_CC

    # ocupação eletrônica (2 e− por MO π)
    num_occupied_mos: int = N_PI_ELECTRONS // 2
    homo_index: int = num_occupied_mos - 1
    lumo_index: int | None = num_occupied_mos if num_occupied_mos < len(eigenvalues) else None
    print("\n[2] Níveis de energia (Hückel)")
    for k in range(len(eigenvalues)):
        eps_k = reduced_energies[k]
        E_k = eigenvalues[k]
        occ_str = "2e⁻" if k <= homo_index else "0e⁻"
        marks = []
        if k == homo_index:
            marks.append("HOMO")
        if lumo_index is not None and k == lumo_index:
            marks.append("LUMO")
        mark_str = ", ".join(marks)
        print(f"    MO{(k+1):>2}   {eps_k:+8.3f}          {E_k:+8.3f}          {occ_str:>3}    {mark_str}")
    
    plot_level_diagram(reduced_energies, homo_index, lumo_index)
    return reduced_energies, homo_index, lumo_index

# -------------------------
# 3 - Calculando as Populações π e ordens π
# -------------------------
def compute_pi_properties(
    energy_eigenvalues: np.ndarray,
    mo_coefficients: np.ndarray,
    atom_labels: List[str] = ATOM_LABELS,
    ring_edges: List[Tuple[int, int]] = RING_EDGES,
    n_pi_electrons: int = N_PI_ELECTRONS,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
 
    # Ordena por energia crescente e extrai MOs ocupados
    sort_indices = np.argsort(energy_eigenvalues)
    mo_coeffs_sorted = mo_coefficients[:, sort_indices]

    num_occupied_mos = n_pi_electrons // 2
    occupied_mos = mo_coeffs_sorted[:, :num_occupied_mos]  # colunas = MOs ocupados

    # Populações π por átomo
    pi_populations = 2.0 * np.sum(occupied_mos ** 2, axis=1)

    # Ordens de ligação π nas ligações do anel
    pi_bond_orders_indexed: Dict[Tuple[int, int], float] = OrderedDict()
    for atom_i, atom_j in ring_edges:
        order_ij = 2.0 * np.sum(occupied_mos[atom_i, :] * occupied_mos[atom_j, :])
        # guarda como 1-based no dicionário de saída
        pi_bond_orders_indexed[(atom_i + 1, atom_j + 1)] = float(order_ij)

    print("\n[4] Propriedades π (apenas MOs ocupados)")
    print(f"    Elétrons π total = {n_pi_electrons}  →  MOs ocupados = {num_occupied_mos}")

    # Populações π
    print("\n[4.1] Populações π por átomo:")
    width_name = max(len(lbl) for lbl in atom_labels)
    for label, pop in zip(atom_labels, pi_populations):
        print(f"    {label:<{width_name}} : {pop:8.5f}")

    # Ordens de ligação π
    print("\n[4.2] Ordens de ligação π (ligações do anel):")
    for (i1, j1), value in pi_bond_orders_indexed.items():
        label_i = atom_labels[i1 - 1]
        label_j = atom_labels[j1 - 1]
        print(f"    {i1}-{j1} ({label_i}–{label_j}) : {value:8.5f}")

# -------------------------
# Visualizações
# -------------------------
def plot_level_diagram(eps: np.ndarray, homo: int, lumo: int | None, path="./q1/level_diagram_pyran.png"):
    occupied_color = "#1f77b4"
    vacant_color   = "#7f7f7f"
    ypad = 0.35
    tol = 1e-6

    # Agrupa níveis degenerados
    unique_levels: list[float] = []
    groups: list[list[int]] = []
    for k, e in enumerate(eps):
        for gi, ge in enumerate(unique_levels):
            if abs(e - ge) < tol:
                groups[gi].append(k)
                break
        else:
            unique_levels.append(e)
            groups.append([k])

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    right_texts = []

    for g in groups:
        e_level = eps[g[0]]
        deg = len(g)

        # Ocupação do grupo:
        # - fully_occ: todos índices <= HOMO
        # - partially: algum <= HOMO e algum > HOMO
        fully_occ = all(idx <= homo for idx in g)
        partially = (any(idx <= homo for idx in g) and not fully_occ)

        line_style = "-" if fully_occ else "--"
        line_color = occupied_color if fully_occ else vacant_color

        ax.hlines(e_level, 0.2, 0.8, colors=line_color,
                  linestyles=line_style, linewidth=2)

        if fully_occ:
            ax.plot([0.44, 0.56], [e_level, e_level], "o", ms=6, color=occupied_color)
        elif partially:
            ax.plot(0.44, e_level, "o", ms=6, color=occupied_color)
            ax.plot(0.56, e_level, "o", ms=6, mfc="none", mec=vacant_color, color=vacant_color)
        else:
            ax.plot(0.5, e_level, "o", ms=6, mfc="none", mec=vacant_color, color=vacant_color)

        label = f"MO{g[0]+1}"
        if homo in g:
            label += " (HOMO)"
        if lumo is not None and lumo in g:
            label += " (LUMO)"
        if deg > 1:
            label += f" ×{deg}"

        txt = ax.text(1.02, e_level, f"{label}\nε={e_level:.3f}",
                      va="center", ha="left",
                      transform=ax.get_yaxis_transform(), clip_on=False)
        right_texts.append(txt)

    # Gap HOMO–LUMO (à esquerda)
    if lumo is not None and homo is not None:
        y_homo = eps[homo]
        y_lumo = eps[lumo]
        y_low, y_high = (min(y_homo, y_lumo), max(y_homo, y_lumo))
        x_gap = 0.2
        ax.vlines(x_gap, y_low, y_high, linestyles=":", colors="k")
        delta_eps = abs(y_lumo - y_homo)
        ax.text(x_gap + 0.12, (y_low + y_high)/2, f"Δε = {delta_eps:.3f}",
                va="center", ha="right", fontsize=9)

    ax.set_ylabel("Energia reduzida ε = (E − α)/β")
    ax.set_xticks([])
    ax.set_ylim(min(eps) - ypad, max(eps) + ypad)
    ax.set_title("Diagrama de níveis (Hückel) – Pirano (2H)")
    ax.grid(axis="y", alpha=0.2)

    legend_elems = [
        Line2D([0],[0], marker='o', linestyle='-',  color=occupied_color, label='ocupado (2e⁻)'),
        Line2D([0],[0], marker='o', mfc='none', mec=vacant_color,
               linestyle='--', color=vacant_color, label='desocupado (0e⁻)'),
    ]
    legend = ax.legend(handles=legend_elems, loc='upper center',
                       bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)

    # Margens para caber rótulos e legenda
    fig.subplots_adjust(right=0.82, bottom=0.22)

    fig.savefig(path, dpi=300, bbox_inches="tight",
                bbox_extra_artists=right_texts + [legend], pad_inches=0.2)
    plt.show()

def plot_mo_bars(C: np.ndarray, labels: list[str], k: int, title: str, path: str):
    plt.figure(figsize=(6,3))
    plt.axhline(0, lw=1)
    plt.bar(range(len(labels)), C[:,k])
    plt.xticks(range(len(labels)), labels)
    plt.ylabel("coeficiente c_i"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.show()

def main():
    print("Molécula: 2H-Pirano (anel de 6 membros com 1 O). Numeração: O(1), C(2)–C(6).")
    print("Presente como heteroaromático em intermediários de açúcares e derivados furânicos.\n")

    # (1) Determinante secular
    print_secular_determinant()

    # (2) Níveis de energia via QR (fallback para eigh) + diagrama de níveis
    huckel_matrix: np.ndarray = build_huckel_matrix()
    try:
        eigenvalues, eigenvectors = qr_eig(huckel_matrix)
    except Exception:
        # fallback numérico
        eigenvalues, eigenvectors = np.linalg.eigh(huckel_matrix)

    reduced_energies, homo_index, lumo_index = calculating_energy_levels(eigenvalues)


    print("\n[3] Orbitais moleculares (ε e combinação linear):")
    C = eigenvectors.copy()
    for j in range(C.shape[1]):
        i_max = np.argmax(np.abs(C[:,j]))
        if C[i_max,j] < 0: C[:,j] *= -1

    for k in range(C.shape[1]):
        terms = "  ".join([f"{C[i,k]:+0.3f}|{ATOM_LABELS[i]}>" for i in range(C.shape[0])])
        print(f"MO{k+1} (ε={reduced_energies[k]:.3f}): {terms}")

    # (4) populações π por átomo e ordens π por ligação
    compute_pi_properties(
        energy_eigenvalues=eigenvalues,
        mo_coefficients=eigenvectors,
        atom_labels=ATOM_LABELS,
        ring_edges=RING_EDGES,
        n_pi_electrons=N_PI_ELECTRONS,
    )    

    # (5) esboços HOMO e LUMO
    plot_mo_bars(C, ATOM_LABELS, homo_index, "HOMO – coeficientes por átomo", "./q1/pyran_HOMO_coeffs.png")
    if lumo_index is not None:
        plot_mo_bars(C, ATOM_LABELS, lumo_index, "LUMO – coeficientes por átomo", "./q1/pyran_LUMO_coeffs.png")

    # resumo final
    print("\nResumo:")
    print("• Matriz de Hückel:\n", huckel_matrix)
    print("• Níveis (ε):", ", ".join([f"{e:.6f}" for e in reduced_energies]))
    print(f"• HOMO = MO{homo_index+1}, LUMO = MO{lumo_index+1 if lumo_index is not None else '—'}")

if __name__ == "__main__":
    main()
