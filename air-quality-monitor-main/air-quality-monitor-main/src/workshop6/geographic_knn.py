from workshop6.utils import *

def knn_geographic(sensors, query_id=None, k=5, weight_type_penalty=200.0, weight_source=50.0, weight_coord=1.0):
    """
    Distancia compuesta:
      - penalty grande si geographicType distinto (para priorizar mismo tipo)
      - distancia euclidiana entre coordenadas (proximidad)
      - diferencia codificada de pollutionSource
    """
    if query_id is None:
        raise ValueError("Necesito query_id para knn_geographic")
    query = None
    for s in sensors:
        if s["id"] == query_id:
            query = s
            break
    if query is None:
        raise ValueError("query_id no encontrado")

    q_type = query["geographicType"]
    q_coords = query["coords"]
    q_source_code = encode_source(query["pollutionSource"])

    results = []
    for s in sensors:
        if s["id"] == query_id:
            continue
        penalty = 0.0 if s["geographicType"] == q_type else 1.0
        coord_d = dist2d(q_coords, s["coords"])
        source_d = abs(q_source_code - encode_source(s["pollutionSource"]))
        # distancia compuesta
        score = penalty * weight_type_penalty + coord_d * weight_coord + source_d * weight_source
        results.append((score, s))
    results.sort(key=lambda x: x[0])
    return results[:k]

