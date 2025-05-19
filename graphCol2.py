import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qutip import *
from scipy.optimize import minimize
from tqdm import tqdm

# Parametri
n_nodes = 3
edges = [(0, 1), (1, 2), (2, 0)]
graph = nx.Graph(edges)
positions = nx.spring_layout(graph, seed=1)
nx.draw(graph, with_labels=True, pos=positions)
plt.show()

#FOR K_COLORS IN RANGE(30) GIRERÀ CERCANDO IL NUM CROMATICO DEL GRAFO
for k_colors in range(2,30):

    # Stato iniziale: prodotto di stati uniformi
    def initial_state(n_nodes, d):
        psi0 = tensor([basis(d, 0) for _ in range(n_nodes)])
        hadamard_d = (1 / np.sqrt(d)) * sum([basis(d, i) for i in range(d)])
        return tensor([hadamard_d for _ in range(n_nodes)])

    # Hamiltoniano di costo: penalizza se due nodi hanno stesso colore
    def cost_hamiltonian(n_nodes, d, edges):
        H = 0
        for (i, j) in edges:
            for c in range(d):
                proj_i = basis(d, c) * basis(d, c).dag()
                proj_j = basis(d, c) * basis(d, c).dag()
                term = tensor([proj_i if k == i else proj_j if k == j else qeye(d) for k in range(n_nodes)])
                H += term
        return H

    # Mixer Hamiltonian: somma di permutazioni fra i livelli
    def mixer_hamiltonian(n_nodes, d):
        H = 0
        for i in range(n_nodes):
            for m in range(d):
                for n in range(m+1, d):
                    flip = basis(d, m) * basis(d, n).dag() + basis(d, n) * basis(d, m).dag()
                    op = tensor([flip if j == i else qeye(d) for j in range(n_nodes)])
                    H += op
        return H

    # Stato iniziale
    psi0 = initial_state(n_nodes, k_colors)

    # Hamiltoniani
    Hc = cost_hamiltonian(n_nodes, k_colors, edges)
    Hm = mixer_hamiltonian(n_nodes, k_colors)
    
    # Funzione costo: aspettazione dell'H_cost sullo stato finale
    loss_history = []

    def qaoa_expectation(params):
        depth = len(params) // 2
        gammas = params[:depth]
        alphas = params[depth:]
        psi = psi0
        for gamma, alpha in zip(gammas, alphas):
            psi = (-1j * gamma * Hc).expm() * psi
            psi = (-1j * alpha * Hm).expm() * psi
        expectation = expect(Hc, psi)
        loss_history.append(expectation)  # salva loss
        return expectation


    # Preparazione della barra di caricamento per l'ottimizzazione (con callback da "minimize")
    n_steps = 100  # massimo numero di iterazioni dell’ottimizzatore
    pbar = tqdm(total=n_steps, desc=f"Training Progress k={k_colors}...")
    def optimization_callback(xk):
        pbar.update(1)

    # Inizializzazione random
    depth = 1
    init_params = np.random.uniform(0, np.pi, 2 * depth)

    # Ottimizzazione
    result = minimize(
        qaoa_expectation,
        init_params,
        method='COBYLA',
        callback=optimization_callback,
        options={'maxiter': n_steps}
    )
    pbar.close()
    opt_params = result.x
    opt_expectation = result.fun


    # Stato finale con parametri ottimizzati
    final_state = psi0
    for gamma, alpha in zip(opt_params[:depth], opt_params[depth:]):
        final_state = (-1j * gamma * Hc).expm() * final_state
        final_state = (-1j * alpha * Hm).expm() * final_state

    # Probabilità finali
    probs = np.abs(final_state.full().flatten())**2

    # Istogramma
    n_levels = k_colors
    labels = [np.base_repr(i, base=n_levels).zfill(n_nodes) for i in range(n_levels**n_nodes)]

    # Decodifica e analisi delle stringhe di output
    def decode_qudit_string(s):
        """Trasforma una stringa tipo '012' in un dizionario nodo → colore"""
        return {node: int(color) for node, color in enumerate(s)}

    def is_valid_coloring_qudit(assignment, edges):
        """Controlla che due nodi adiacenti non abbiano lo stesso colore"""
        for u, v in edges:
            if assignment[u] == assignment[v]:
                return False
        return True

    threshold = max(probs) - 0.0001
    print("Bitstring | Assegnamento | Valido | Probabilità")
    print("-" * 50)

    valid_count = 0
    best_assignment = None

    for i, p in enumerate(probs):
        if p > threshold:
            s = np.base_repr(i, base=k_colors).zfill(n_nodes)
            assignment = decode_qudit_string(s)
            valid = is_valid_coloring_qudit(assignment, edges)
            print(f"{s} | {assignment} | {valid} | {p:.4f}")
            if valid:
                valid_count += 1
                best_assignment = assignment

    #Output: numero cromatico
    if valid_count > 0:
        print(f"Il numero cromatico del grafo è {k_colors} e si può colorare in {valid_count} modi diversi")
        break
    else:
        print(f"\nNessuna colorazione valida trovata con {k_colors} colori. Provo con {k_colors+1}...")


#STAMPE E PLOTS
print("Migliori parametri trovati:", opt_params)
print("Valore minimo atteso H_cost:", opt_expectation)

# Plot convergenza 
plt.plot(loss_history)
plt.xlabel("Step")
plt.ylabel("Costo")
plt.title("Convergenza funzione costo")
plt.grid(True)
plt.show()

# Istogramma
plt.figure(figsize=(10, 4))
plt.bar(labels, probs)
plt.xticks(rotation=90)
plt.xlabel("Stati (bitstring in base colori)")
plt.ylabel("Probabilità")
plt.title("Distribuzione delle probabilità - QAOA con qudit")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualizza grafo colorato
def plot_colored_graph(graph, assignment, positions=None, cmap=plt.cm.Set3):
    node_colors = [assignment[n] for n in graph.nodes]
    unique_colors = sorted(set(node_colors))
    
    n_colors = len(unique_colors)
    color_list = [cmap(i / max(1, n_colors - 1)) for i in range(n_colors)]
    color_map = {c: color_list[i] for i, c in enumerate(unique_colors)}
    final_colors = [color_map[c] for c in node_colors]

    if positions is None:
        positions = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(6, 4))
    nx.draw(
        graph,
        pos=positions,
        with_labels=True,
        node_color=final_colors,
        edge_color="gray",
        node_size=800,
        font_color="black",
        font_weight="bold"
    )
    plt.title("Grafo colorato secondo l'assegnazione QAOA", fontsize=14)
    plt.axis('off')
    plt.show()

if best_assignment:
    plot_colored_graph(graph, best_assignment, positions)
