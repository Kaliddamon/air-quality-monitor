def build_similarity_graph(sensors, k=5, threshold=20.0, use_mean_difference=False):
    """
    Construye lista de adyacencia como pares (i, j) si la distancia de polucion <= threshold
    Se usa knn_pollution para acotar candidatos (mejor rendimiento en datasets grandes).
    threshold interpreta:
      - si use_mean_difference: threshold se aplica a diferencia de medias absolutas
      - else: threshold se aplica a euclidiana sobre vectores de 7 días
    Devuelve adjacency dict: id -> lista de ids vecinos
    """
    ids = [s["id"] for s in sensors]
    lookup = {s["id"]: s for s in sensors}
    adjacency = {sid: [] for sid in ids}
    for s in sensors:
        qid = s["id"]
        # pedir más candidatos para tener seguridad (k*2)
        neighbors = knn_pollution(sensors, query_id=qid, k=max(5, k*2), use_average=use_mean_difference)
        for dist, nb in neighbors:
            if use_mean_difference:
                val = dist  # aquí dist ya es diferencia media absoluta
            else:
                val = dist  # euclidiana sobre 7 días
            if val <= threshold:
                adjacency[qid].append(nb["id"])
                adjacency[nb["id"]].append(qid)  # arista bidireccional
    # quitar duplicados en listas
    for k0 in adjacency:
        seen = set()
        newl = []
        for x in adjacency[k0]:
            if x not in seen and x != k0:
                seen.add(x)
                newl.append(x)
        adjacency[k0] = newl
    return adjacency

def connected_components(adjacency):
    """
    adjacency: dict id -> list of neighbor ids
    Retorna lista de componentes: cada una es lista de ids
    """
    visited = set()
    components = []
    for node in adjacency:
        if node in visited:
            continue
        # BFS/DFS
        stack = [node]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for nb in adjacency[cur]:
                if nb not in visited:
                    stack.append(nb)
        components.append(comp)
    return components

def group_sensors_by_similarity(sensors, k=5, thresholds=[10.0, 20.0, 30.0], use_mean_difference=False):
    """
    Para cada threshold construye grafo y devuelve componentes.
    """
    result = []
    for th in thresholds:
        adj = build_similarity_graph(sensors, k=k, threshold=th, use_mean_difference=use_mean_difference)
        comps = connected_components(adj)
        result.append({
            "threshold": th,
            "n_groups": len(comps),
            "groups": comps
        })
    return result
