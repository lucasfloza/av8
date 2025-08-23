# -*- coding: utf-8 -*-
"""
Hückel para o Pirano (2H), apenas orbitais pz.
- Constrói a matriz de Hückel
- Calcula autovalores/autovetores (QR)
- Gera o diagrama de níveis e salva a figura
- Faz o preenchimento eletrônico
- Calcula coeficientes dos MOs, populações π e ordens de ligação π
"""
import numpy as np
import matplotlib.pyplot as plt

# --- Parâmetros do modelo Hückel ---
coulomb_integral_carbon = 0.0           # α (referência de energia dos carbonos)
resonance_integral_CC   = -1.0          # β (C–C), geralmente negativo
coulomb_shift_oxygen    = 1.5           # h_O  (α_O = α + h_O * β)
resonance_factor_CO     = 1.0           # k_CO (β_CO = k_CO * β)
num_pi_electrons        = 6             # nº de elétrons π

# Conectividade e rótulos
ring_connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
atom_labels = ['O(1)', 'C(2)', 'C(3)', 'C(4)', 'C(5)', 'C(6)']

# --- Construção da matriz de Hückel ---
num_atoms = 6
huckel_matrix = np.zeros((num_atoms, num_atoms), dtype=float)

# Diagonais (O e Cs)
huckel_matrix[0, 0] = coulomb_integral_carbon + coulomb_shift_oxygen * resonance_integral_CC
for i in range(1, num_atoms):
    huckel_matrix[i, i] = coulomb_integral_carbon

# Ligações do anel
for i, j in ring_connections:
    if 0 in (i, j):  # envolve oxigênio
        huckel_matrix[i, j] = huckel_matrix[j, i] = resonance_factor_CO * resonance_integral_CC
    else:            # ligação C–C
        huckel_matrix[i, j] = huckel_matrix[j, i] = resonance_integral_CC


def qr_eigenvalues(matrix: np.ndarray, max_iter: int = 5000, tol: float = 1e-12, with_vectors: bool = True):
    """
    Calcula autovalores (e opcionalmente autovetores) via iteração QR com shift simples.

    Parâmetros
    ----------
    matrix : np.ndarray
        Matriz quadrada real (simétrica no caso Hückel).
    max_iter : int
        Máximo de iterações do método.
    tol : float
        Tolerância para a norma fora da diagonal (critério de parada).
    with_vectors : bool
        Se True, acumula Q para retornar autovetores.

    Retorna
    -------
    eigenvalues : np.ndarray
        Autovalores em ordem crescente.
    eigenvectors : np.ndarray | None
        Autovetores por colunas, alinhados aos autovalores. None se with_vectors=False.
    """
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


def pi_populations_and_bond_orders(eigenvalues: np.ndarray,
                                   eigenvectors: np.ndarray,
                                   connections: list[tuple[int, int]],
                                   num_electrons: int):
    """
    Calcula populações π por centro e ordens de ligação π (Hückel) usando MOs ocupados.

    Parâmetros
    ----------
    eigenvalues : np.ndarray
        Autovalores (energias) dos MOs.
    eigenvectors : np.ndarray
        Autovetores (coeficientes dos MOs por coluna).
    connections : list[tuple[int,int]]
        Lista de pares (i, j) conectados no anel (0-index).
    num_electrons : int
        Número total de elétrons π (2 por MO ocupado).

    Retorna
    -------
    pi_populations : np.ndarray
        População π em cada centro (2 * soma dos c_i^2 nos MOs ocupados).
    pi_bond_orders : dict[(int,int), float]
        Ordem de ligação π para cada par (i+1, j+1) da lista de conexões.
    """
    idx = np.argsort(eigenvalues)
    occupied_coeffs = eigenvectors[:, idx][:, :num_electrons//2]
    pi_populations = 2.0 * np.sum(occupied_coeffs**2, axis=1)

    pi_bond_orders = {}
    for i, j in connections:
        pi_bond_orders[(i+1, j+1)] = 2.0 * np.sum(occupied_coeffs[i, :] * occupied_coeffs[j, :])
    return pi_populations, pi_bond_orders


def print_populations_and_bond_orders(pi_populations: np.ndarray, pi_bond_orders: dict):
    """
    Imprime populações π por centro e ordens de ligação π para as ligações do anel.
    """
    print("\nPopulações π:")
    for atom, pop in zip(atom_labels, pi_populations):
        print(f"{atom:4s}: {pop:0.5f}")

    print("\nOrdens de ligação π (ligações do anel):")
    for (i, j), val in pi_bond_orders.items():
        print(f"{i}-{j}: {val:0.5f}")


def plot_level_diagram(reduced_energies: np.ndarray,
                       HOMO_index: int,
                       LUMO_index: int | None,
                       save_path: str = "level_diagram_pyran.png",
                       dpi: int = 300):
    """
    Plota e salva o diagrama de níveis (energias reduzidas ε) com ocupação eletrônica.

    Parâmetros
    ----------
    reduced_energies : np.ndarray
        Energias reduzidas ε = (E - α) / β em ordem crescente.
    HOMO_index : int
        Índice do HOMO (0-index).
    LUMO_index : int | None
        Índice do LUMO (ou None se inexistente).
    save_path : str
        Caminho/arquivo para salvar a figura (PNG).
    dpi : int
        Resolução da imagem salva.
    """
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

    # >>> salvar figura <<<
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"Diagrama salvo em: {save_path}")


def print_huckel_matrix_and_levels(huckel_matrix: np.ndarray, reduced_energies: np.ndarray,
                                   HOMO_index: int, LUMO_index: int | None):
    """
    Imprime a matriz de Hückel e a lista de níveis (ε), marcando HOMO/LUMO.
    """
    print("\nMatriz de Hückel H:\n", huckel_matrix)
    print("\nNíveis (ε):")
    for k, energy in enumerate(reduced_energies):
        mark = ""
        if k == HOMO_index:
            mark = "  <= HOMO"
        elif LUMO_index is not None and k == LUMO_index:
            mark = "  <= LUMO"
        print(f"MO{k+1}: ε = {energy:.6f}{mark}")


# --- Autovalores/autovetores e energias reduzidas ---
eigenvalues, eigenvectors = qr_eigenvalues(huckel_matrix, with_vectors=True)
reduced_energies = (eigenvalues - coulomb_integral_carbon) / resonance_integral_CC

# --- Preenchimento eletrônico ---
num_occupied_MOs = num_pi_electrons // 2
HOMO_index = num_occupied_MOs - 1
LUMO_index = num_occupied_MOs if num_occupied_MOs < num_atoms else None

# --- Ajuste de sinal dos vetores próprios para consistência visual ---
MO_coefficients = eigenvectors.copy()
for j in range(num_atoms):
    max_index = np.argmax(np.abs(MO_coefficients[:, j]))
    if MO_coefficients[max_index, j] < 0:
        MO_coefficients[:, j] *= -1.0

# --- Populações π e ordens de ligação π ---
pi_populations, pi_bond_orders = pi_populations_and_bond_orders(
    eigenvalues, eigenvectors, ring_connections, num_pi_electrons
)

# --- Execução ---
print_populations_and_bond_orders(pi_populations, pi_bond_orders)
plot_level_diagram(reduced_energies, HOMO_index, LUMO_index,
                   save_path="level_diagram_pyran.png", dpi=300)
print_huckel_matrix_and_levels(huckel_matrix, reduced_energies, HOMO_index, LUMO_index)
