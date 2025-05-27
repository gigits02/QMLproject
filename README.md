Graph Coloring with QAOA
=========================

Questa repository contiene diverse implementazioni del protocollo QAOA (Quantum Approximate Optimization Algorithm) applicato al problema del **graph coloring**. Le varianti presenti utilizzano differenti codifiche dei qubit (one-hot e binaria) e un'estensione concettuale basata su **qudit**, ovvero stati quantistici con d livelli.

Contenuto della repository
--------------------------
I seguenti file sono presenti nella repository:

1. `graphColOne.ipynb` - Notebook Jupyter con implementazione e spiegazione del metodo QAOA con codifica *one-hot*.
2. `graphColOne.py`    - Versione compatta e automatizzata del codice *one-hot*. Restituisce il numero cromatico dato un grafo in input.

3. `graphColBin.ipynb` - Notebook con metodo QAOA usando codifica *binaria*. Contiene sezioni teoriche ed esempi.
4. `graphColBin.py`    - Versione compatta del codice *binario*. Calcola il numero cromatico del grafo.

5. `graphColQd.ipynb`  - Notebook esplorativo dell’estensione del graph coloring a *qudit*. Include teoria e codice dimostrativo.
6. `graphColQd.py`     - Implementazione compatta del metodo basato su qudit. Ritorna il numero cromatico stimato.

Caratteristiche principali
--------------------------
- **QAOA** come strategia di risoluzione del problema combinatorio.
- **Codifiche multiple** per confrontare efficienza e consumo di risorse:
  - One-hot (più qubit, codifica semplice)
  - Binaria (meno qubit, codifica più compatta)
  - Qudit (versatilità teorica, ancora esplorativa)
- Ogni file `.ipynb` contiene spiegazioni teoriche, visualizzazioni e test.
- Ogni file `.py` è progettato per l'esecuzione diretta su un grafo definito, restituendo il **numero cromatico** stimato.

Requisiti
---------
- Python 3.8+
- [PennyLane](https://pennylane.ai/) (per circuiti quantistici e ottimizzazione)
- NumPy, NetworkX, Matplotlib (per grafo e visualizzazione)
- (Opzionale) Jupyter Notebook per l'esecuzione interattiva
