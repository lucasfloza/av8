# -*- coding: utf-8 -*-
"""
Hückel para o Pirano (2H), apenas orbitais pz.
Constrói a matriz de Hückel, o determinante secular (quando sympy está disponível),
calcula níveis de energia, diagrama de níveis, preenchimento eletrônico, MOs,
populações π e ordens de ligação π.
"""
import numpy as np
import matplotlib.pyplot as plt

# --- Parâmetros do modelo Hückel ---
coulomb_integral_carbon = 0.0
resonance_integral_CC   = -1.0
coulomb_shift_oxygen    = 1.5      # h_O (α_O = α + h_O * β)
resonance_factor_CO     = 1.0      # k_CO (β_CO = k_CO * β)
num_pi_electrons        = 6

# Conectividade e rótulos
ring_connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
atom_labels = ['O(1)', 'C(2)', 'C(3)', 'C(4)', 'C(5)', 'C(6)']

# --- Construção da matriz de Hückel ---
num_atoms = 6
huckel_matrix = np.zeros((num_atoms, num_atoms), dtype=float)

# Termo diagonal do oxigênio
huckel_matrix[0, 0] = coulomb_integral_carbon + coulomb_shift_oxygen * resonance_integral_CC

# Termos diagonais dos carbonos
for i in range(1, num_atoms):
    huckel_matrix[i, i] = coulomb_integral_carbon

# Ligações do anel
for i, j in ring_connections:
    if 0 in (i, j):  # envolve oxigênio
        huckel_matrix[i, j] = huckel_matrix[j, i] = resonance_factor_CO * resonance_integral_CC
    else:            # ligação C-C
        huckel_matrix[i, j] = huckel_matrix[j, i] = resonance_integral_CC


# --- Decomposição QR para autovalores e autovetores ---
def qr_eigenvalues(matrix, max_iter=5000, tol=1e-12, with_vectors=True):
    current_matrix = matrix.copy().astype(float)
    n = current_matrix.shape[0]
    accumulated_Q = np.eye(n) if with_vectors else None
    for _ in range(max_iter):
        off_diag_norm = np.sqrt(np.sum((current_matrix - np.diag(np.diag(current_matrix)))**2))
        if off_diag_norm < tol:
            break
        shift = current_matrix[-1, -1]
        Q, R = np.linalg.qr(current_matrix - shift*np.eye(n))
        current_matrix = R @ Q + shift*np.eye(n)
        if with_vectors:
            accumulated_Q = accumulated_Q @ Q
    eigenvalues = np.diag(current_matrix).copy()
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = accumulated_Q[:, idx] if with_vectors else None
    return eigenvalues, eigenvectors


eigenvalues, eigenvectors = qr_eigenvalues(huckel_matrix, with_vectors=True)
reduced_energies = (eigenvalues - coulomb_integral_carbon) / resonance_integral_CC

# --- Preenchimento eletrônico ---
num_occupied_MOs = num_pi_electrons // 2
HOMO_index = num_occupied_MOs - 1
LUMO_index = num_occupied_MOs if num_occupied_MOs < num_atoms else None

# --- Ajuste de sinal dos vetores próprios ---
MO_coefficients = eigenvectors.copy()
for j in range(num_atoms):
    max_index = np.argmax(np.abs(MO_coefficients[:, j]))
    if MO_coefficients[max_index, j] < 0:
        MO_coefficients[:, j] *= -1.0


# --- Populações e ordens de ligação π ---
def pi_populations_and_bond_orders(eigenvalues, eigenvectors, ring_connections, num_electrons):
    idx = np.argsort(eigenvalues)
    occupied_coeffs = eigenvectors[:, idx][:, :num_electrons//2]
    pi_populations = 2.0 * np.sum(occupied_coeffs**2, axis=1)
    bond_orders = {}
    for i, j in ring_connections:
        bond_orders[(i+1, j+1)] = 2.0 * np.sum(occupied_coeffs[i, :] * occupied_coeffs[j, :])
    return pi_populations, bond_orders


pi_populations, pi_bond_orders = pi_populations_and_bond_orders(
    eigenvalues, eigenvectors, ring_connections, num_pi_electrons
)


# --- Impressões e gráficos ---
def print_populations_and_bond_orders():
    print("\nPopulações π:")
    for atom, pop in zip(atom_labels, pi_populations):
        print(f"{atom:4s}: {pop:0.5f}")

    print("\nOrdens de ligação π (ligações do anel):")
    for (i, j), val in pi_bond_orders.items():
        print(f"{i}-{j}: {val:0.5f}")


def plot_level_diagram():
    plt.figure(figsize=(6, 6))
    for k, energy in enumerate(reduced_energies):
        plt.hlines(energy, xmin=0.2, xmax=0.8, linewidth=2)
        occupancy = 2 if k <= HOMO_index else 0
        for m in range(occupancy):
            plt.plot(0.5 + (m-0.5)*0.15, energy, 'o')
        label = f"MO{k+1}"
        if k == HOMO_index:
            label += " (HOMO)"
        if LUMO_index is not None and k == LUMO_index:
            label += " (LUMO)"
        plt.text(0.82, energy, f"{label}\nε={energy:.3f}", va='center', fontsize=9)
    plt.ylim(min(reduced_energies)-0.5, max(reduced_energies)+0.5)
    plt.xlim(0, 1)
    plt.ylabel("Energia reduzida ε = (E - α)/β")
    plt.title("Diagrama de níveis (Hückel) – Pirano (2H)")
    plt.xticks([])
    plt.tight_layout()
    plt.savefig()
    plt.show()


def print_huckel_matrix_and_levels():
    print("\nMatriz de Hückel H:\n", huckel_matrix)
    print("\nNíveis (ε):")
    for k, energy in enumerate(reduced_energies):
        mark = ""
        if k == HOMO_index:
            mark = "  <= HOMO"
        elif LUMO_index is not None and k == LUMO_index:
            mark = "  <= LUMO"
        print(f"MO{k+1}: ε = {energy:.6f}{mark}")


# --- Execução ---
print_populations_and_bond_orders()
plot_level_diagram()
print_huckel_matrix_and_levels()
