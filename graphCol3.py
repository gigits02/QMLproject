import pennylane as qml
from pennylane import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
from functools import reduce
import operator

#GRAFO
# Parametri
n_nodes = 3
k_colors = 3
edges = [(0, 1), (1, 2), (2, 0)]
graph = nx.Graph(edges)
positions = nx.spring_layout(graph, seed=1)
nx.draw(graph, with_labels=True, pos=positions)
plt.show()

#FOR K_COLORS IN RANGE(30) GIRERÀ CERCANDO IL NUM CROMATICO DEL GRAFO
for k_colors in range(2, 30):

    m = int(np.ceil(np.log2(k_colors)))  # qubits per nodo
    n_qubits = n_nodes * m
    wires = list(range(n_qubits))

    # Qubit per nodo
    def qubits_for_node(v):
        return [v * m + i for i in range(m)]

    # Hamiltoniano di costo
    cost_h = qml.Hamiltonian([], [])  # inizializzazione esplicita

    # --- Penalità: nodi adiacenti con stesso colore ---
    for (u, v) in edges:
        terms = []
        for i in range(m):
            op = (qml.PauliZ(qubits_for_node(u)[i]) @ qml.PauliZ(qubits_for_node(v)[i]))
            terms.append((1 + op) / 2)

        # Prodotto dei termini
        penalty = reduce(operator.matmul, terms)
        cost_h += penalty

    # --- Penalità: codifiche non valide (se k < 2^m) ---
    invalid_bitstrings = [b for b in product([0, 1], repeat=m) if int("".join(map(str, b)), 2) >= k_colors]

    for v in range(n_nodes):
        q_v = qubits_for_node(v)
        for b in invalid_bitstrings:
            proj_terms = []
            for i in range(m):
                z = qml.PauliZ(q_v[i])
                coeff = (-1)**b[i]
                proj_terms.append((1 + coeff * z) / 2)

            proj = reduce(operator.matmul, proj_terms)
            cost_h += proj

    # Mixer Hamiltonian
    mixer_h = qml.Hamiltonian([], [])
    for w in wires:
        mixer_h += qml.PauliX(w)
    
    # QAOA layer
    def qaoa_layer(gamma, alpha):
        qml.ApproxTimeEvolution(cost_h, gamma, 1)
        qml.ApproxTimeEvolution(mixer_h, alpha, 1)

    # Circuito
    depth = 2
    def circuit(params, **kwargs):
        for w in wires:
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, depth, params[0], params[1])
    
    # Device
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def cost_function(params):
        circuit(params)
        return qml.expval(cost_h)
    
    # Ottimizzazione
    optimizer = qml.AdamOptimizer()
    steps = 100
    params = np.array([[0.5] * depth, [0.5] * depth], requires_grad=True)

    patience = 3           
    min_delta = 0.01        
    counter = 0
    cost_history = []
    best_cost = np.inf
    best_params = None

    for step in tqdm(range(steps), desc=f"Training Progress k={k_colors}..."):
        cost = cost_function(params)
        cost_history.append(cost)

        # Early stopping logic
        if best_cost - cost > min_delta:
            best_cost = cost
            best_params = params.copy()
            counter = 0
        else:
            counter += 1

        # Check stop condition
        if counter >= patience:
            print(f"Early stopping at step {step}")
            break

        params = optimizer.step(cost_function, params)


    # Probabilità finali
    @qml.qnode(dev)
    def probability_circuit(gamma, alpha):
        circuit([gamma, alpha])
        return qml.probs(wires=wires)

    probs = probability_circuit(best_params[0], best_params[1])

    def decode_binary(bitstring, m):
        return int(bitstring, 2)

    def analyze_binary_results(probs, n_nodes, m, k_colors, edges, threshold=None):
        if threshold is None:
            threshold = max(probs) - 1e-4

        print("Bitstring | Assegnamento | Valido | Probabilità")
        print("-" * 50)
        deg = 0
        outcome = False
        best_assignment = None

        for i, p in enumerate(probs):
            if p < threshold:
                continue

            bitstring = format(i, f"0{n_nodes * m}b")
            assignment = {}
            valid = True

            for v in range(n_nodes):
                bits = bitstring[v * m:(v + 1) * m]
                color = decode_binary(bits, m)
                assignment[v] = color
                if color >= k_colors:
                    valid = False

            for u, v in edges:
                if assignment[u] == assignment[v]:
                    valid = False

            print(f"{bitstring} | {assignment} | {valid} | {p:.4f}")
            if valid:
                deg += 1
                best_assignment = assignment
                outcome = True

        return best_assignment, outcome, deg

    assignment, outcome, deg = analyze_binary_results(probs, n_nodes, m, k_colors, edges)
    
    #Output: numero cromatico
    if outcome:
        print(f"Il numero minimo di colori per colorare il grafo è {k_colors} e si può fare in {deg} modi diversi")
        break
    else:
        print(f"\nNessuna colorazione valida trovata con {k_colors} colori. Provo con {k_colors+1}...")        

#STAMPE E PLOTS
print("Miglior costo trovato:", best_cost)
print("Parametri corrispondenti:", best_params)
# Plot convergenza
plt.plot(cost_history)
plt.title("Convergenza funzione costo")
plt.xlabel("Step")
plt.ylabel("Costo")
plt.grid(True)
plt.show()

# Istogramma
bitstrings = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]

plt.figure(figsize=(10, 4))
plt.bar(bitstrings, probs)
plt.xticks(rotation=90)
plt.xlabel("Stati")
plt.ylabel("Probabilità")
plt.title("Distribuzione delle probabilità - QAOA con encoding binario")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualizzazione grafo colorato
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

if assignment:
    plot_colored_graph(graph, assignment, positions)
