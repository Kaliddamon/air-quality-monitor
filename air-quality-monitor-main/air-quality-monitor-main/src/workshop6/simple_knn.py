from workshop6.utils import *

# --------------------------
# Algorithm 1: Basic Pollution Similarity (Simple KNN)
# - usa Euclidean sobre vectores de 7 días, o compara medias si use_average=True
# --------------------------
def knn_pollution(sensors, query, k=5, use_average=False):
    """
    sensors: lista de dict con clave 'pollutionLevels7'
    query_id: si se pasa, tomará ese sensor como consulta
    query_vector: si se pasa, usa ese vector de 7 valores
    k: número de vecinos
    use_average: si True compara absolutos de medias en lugar de euclidiana
    """

    if query is None:
        raise ValueError("query no encontrado")

    query_id = query["_id"]
    query_vector = query["pollutionLevels7"]

    results = []
    if use_average:
        qavg = mean_vector(query_vector)
        for s in sensors:
            if s.get("pollutionLevels7") is None:
                continue
            avg_s = mean_vector(s["pollutionLevels7"])
            score = abs(qavg - avg_s)  # menor = más similar
            results.append((score, s))
    else:
        for s in sensors:
            if s.get("pollutionLevels7") is None:
                continue
            score = euclidean(query_vector, s["pollutionLevels7"])
            results.append((score, s))

    # ordenar y retornar k vecinos (excluye el propio si coincide exactamente con id)
    results.sort(key=lambda x: x[0])
    # filtrar out self si query_id dado
    filtered = []
    for score, s in results:
        if query_id is not None and s["_id"] == query_id:
            continue
        filtered.append((score, s))
        if len(filtered) >= k:
            break
    return filtered



