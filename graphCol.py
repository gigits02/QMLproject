import pennylane as qml
from pennylane import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

#GRAFO
# Parametri
n_nodes = 4
edges = [(0, 1), (1, 2), (2, 3)] 
graph = nx.Graph(edges)
positions = nx.spring_layout(graph, seed=1)
nx.draw(graph, with_labels=True, pos=positions)
plt.show()

#FOR K_COLORS IN RANGE(30) GIRERÀ CERCANDO IL NUM CROMATICO DEL GRAFO
for k_colors in range(1,30):

    n_qubits = n_nodes * k_colors
    wires = list(range(n_qubits))

    # Mappa nodo-colore → qubit
    def qubit_index(node, color):
        return node * k_colors + color

    # Hamiltoniano di costo
    cost_h = 0

    # Penalità: ogni nodo deve avere un solo colore
    for node in range(n_nodes):
        terms = []
        for color in range(k_colors):
            wire = qubit_index(node, color)
            terms.append((qml.Identity(wire) - qml.PauliZ(wire)) / 2)
        # Somma dei bit attivati
        sum_x = terms[0]
        for t in terms[1:]:
            sum_x += t
        # (sum - 1)^2 = sum^2 - 2sum + 1
        cost_h += sum_x @ sum_x - 2 * sum_x + 1


    # Penalità: nodi adiacenti non devono avere lo stesso colore
    for (u, v) in edges:
        for color in range(k_colors):
            i = qubit_index(u, color)
            j = qubit_index(v, color)

            penalty = (1 - qml.PauliZ(i)) / 2 @ (1 - qml.PauliZ(j)) / 2
            cost_h += penalty

    # Hamiltoniano di mixer
    mixer_h = 0
    for qubit in wires:
        mixer_h += qml.PauliX(qubit)

    # QAOA Layer
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
    dev = qml.device("qulacs.simulator", wires=n_qubits)

    @qml.qnode(dev)
    def cost_function(params):
        circuit(params)
        return qml.expval(cost_h)

    # Ottimizzazione
    optimizer = qml.GradientDescentOptimizer()
    steps = 500
    params = np.array([[0.5] * depth, [0.5] * depth], requires_grad=True)

    cost_history = []
    best_cost = np.inf
    best_params = None

    for i in tqdm(range(steps), desc=f"Training Progress k={k_colors}...", unit="epoch"):
        cost = cost_function(params)
        cost_history.append(cost)
        
        # Aggiorna i migliori parametri trovati
        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()

        # Step dell'ottimizzatore
        params = optimizer.step(cost_function, params)

    # Probabilità finali
    @qml.qnode(dev)
    def probability_circuit(gamma, alpha):
        circuit([gamma, alpha])
        return qml.probs(wires=wires)

    probs = probability_circuit(best_params[0], best_params[1])

    def decode_bitstring(bitstring, n_nodes, k_colors):
        assignment = {}
        for node in range(n_nodes):
            for color in range(k_colors):
                index = node * k_colors + color
                if bitstring[index] == '1':
                    if node in assignment:
                        assignment[node].append(color)
                    else:
                        assignment[node] = [color]
        return assignment

    def is_valid_coloring(assignment, edges):
        # Ogni nodo deve avere esattamente un colore
        for node, colors in assignment.items():
            if len(assignment) != n_nodes or len(colors) != 1:
                return False
        # Nodi adiacenti non devono avere lo stesso colore
        for u, v in edges:
            color_u = assignment.get(u, [-1])[0]
            color_v = assignment.get(v, [-2])[0]
            if color_u == color_v:
                return False
        return True

    def analyze_results(probs, n_nodes, k_colors, edges, threshold=np.max(probs)-0.00005):
        print("Bitstring | Assegnamento | Valido | Probabilità")
        print("-" * 50)
        outcome = False
        deg = 0
        for idx, prob in enumerate(probs):
            if prob > threshold:
                bitstring = format(idx, f"0{n_nodes * k_colors}b")
                assignment = decode_bitstring(bitstring, n_nodes, k_colors)
                valid = is_valid_coloring(assignment, edges)
                print(f"{bitstring} | {assignment} | {valid} | {prob:.4f}")
                if valid:
                    outcome = True
                    deg += 1
        return assignment, outcome, deg

    assignment, outcome, deg = analyze_results(probs, n_nodes, k_colors, edges)

    #Output: numero cromatico
    if outcome:
        print(f"Il numero minimo di colori per colorare il grafo è {k_colors} e si può fare in {deg} modi diversi")
        break
    else:
        print(f"\nNessuna colorazione valida trovata con {k_colors} colori.")


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
plt.figure(figsize=(10, 4))
plt.bar(range(2**n_qubits), probs)
plt.xticks(rotation=90)
plt.xlabel("Stati")
plt.ylabel("Probabilità")
plt.title("Distribuzione delle probabilità - QAOA con qudit")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualizza grafo colorato
def plot_colored_graph(graph, assignment, positions=None, cmap=plt.cm.Set3):
    node_colors = [assignment[n][0] if len(assignment[n]) == 1 else -1 for n in graph.nodes]
    unique_colors = sorted(set(c for c in node_colors if c != -1))
    
    n_colors = len(unique_colors)
    color_list = [cmap(i / max(1, n_colors - 1)) for i in range(n_colors)]
    color_map = {c: color_list[i] for i, c in enumerate(unique_colors)}

    final_colors = [color_map.get(c, (0.7, 0.7, 0.7)) for c in node_colors]

    if positions is None:
        positions = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(6, 4))
    nx.draw(
        graph,
        pos=positions,
        with_labels=True,
        node_color=final_colors,
        edge_color="gray",
        cmap=cmap,
        node_size=800,
        font_color="black",
        font_weight="bold"
    )
    plt.title("Grafo colorato secondo l'assegnazione QAOA", fontsize=14)
    plt.axis('off')
    plt.show()

plot_colored_graph(graph, assignment, positions)