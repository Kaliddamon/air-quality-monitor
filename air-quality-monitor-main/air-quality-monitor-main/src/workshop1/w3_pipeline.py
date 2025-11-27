import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from workshop3.markov_model import MarkovModel
from workshop3.spatial_markov import SpatialMarkov

def plot_markov_chain(states, matrix):
    plt.figure(figsize=(8, 6))
    G = nx.DiGraph()

    # nodos
    for s in states:
        G.add_node(s)

    # aristas con pesos
    for i, s_from in enumerate(states):
        for j, s_to in enumerate(states):
            w = matrix[i][j]
            if w > 0.05:  # filtrar ruido
                G.add_edge(s_from, s_to, weight=round(w, 2))

    pos = nx.circular_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos,
            with_labels=True,
            node_size=1800,
            node_color="#8cc8ff",
            arrows=True,
            arrowsize=20,
            font_size=12,
            arrowstyle='-|>')

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Diagrama de Transición - Cadena de Markov PM2.5")

    return plt


# ------------------------------------------------------------------
# W3 MAIN PIPELINE
# ------------------------------------------------------------------
def build_w3_structures(records):
    # ----------------------------
    # 1. FORMAT RECORDS
    # ----------------------------
    formatted = []
    for r in records:
        aq = r.get("airQualityData", {}) or {}
        pm = aq.get("PM25")
        ts = r.get("timestamp")

        if pm is None or ts is None:
            continue

        formatted.append({
            "timestamp": ts,
            "PM25": float(pm)
        })

    # ----------------------------
    # 2. MARKOV MODEL
    # ----------------------------
    markov = MarkovModel()
    markov.fit(formatted)
    stationary = markov.compute_stationary()

    # predicted state
    states = markov.states
    if stationary is not None:
        i = int(np.argmax(stationary))
        predicted_state = states[i]
    else:
        predicted_state = None

    # ----------------------------
    # 3. SPATIAL MARKOV
    # ----------------------------
    spatial = SpatialMarkov()
    spatial.load_sensors(records)
    spatial_transition = spatial.build_spatial_transition()

    if spatial_transition is not None:
        n = len(spatial_transition)
        initial = np.ones(n) / n
        diffusion = spatial.simulate_diffusion(initial, steps=5).tolist()
    else:
        diffusion = None

    # ----------------------------
    # 4. STATE SEQUENCE
    # ----------------------------
    seq = []
    for f in formatted:
        pm = f["PM25"]
        if pm < 12:
            seq.append("bueno")
        elif pm < 35:
            seq.append("moderado")
        elif pm < 55:
            seq.append("insalubre")
        else:
            seq.append("peligroso")

    # ----------------------------
    # 5. EXPORT FINAL
    # ----------------------------
    return {
        "transition_matrix": markov.transition_matrix.tolist(),
        "stationary_distribution": stationary.tolist() if stationary is not None else None,
        "predicted_state": predicted_state,
        "states_sequence": seq[:200],
        "spatial_probabilities": diffusion,
        "spatial_matrix": spatial_transition.tolist() if spatial_transition is not None else None,
        "states": states  # NECESARIO PARA EL GRÁFICO
    }


def run_w3_from_records(records):
    return build_w3_structures(records)
