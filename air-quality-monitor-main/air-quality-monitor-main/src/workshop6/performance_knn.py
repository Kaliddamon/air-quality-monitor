from workshop6.utils import *
from workshop6.alert_knn import knn_alert
from workshop6.geographic_knn import knn_geographic
from workshop6.simple_knn import knn_pollution

import random

def add_next_day_pollution(sensors, seed=999):
    random.seed(seed)
    for s in sensors:
        base = mean_vector(s["pollutionLevels7"])
        # ruido proporcional a la media (5% std), más efecto de fuente/vento
        noise = random.gauss(0, base * 0.05)
        wind = s["weatherInfluence"]["windSpeed"]
        wind_effect = - (wind / 30.0) * random.uniform(0, base * 0.03)  # viento reduce algo
        source = s["pollutionSource"]
        source_effect = 0.0
        if source == "industrial":
            source_effect = random.uniform(0, base * 0.04)
        elif source == "traffic":
            source_effect = random.uniform(0, base * 0.02)
        next_day = max(0.0, base + noise + wind_effect + source_effect)
        s["nextDayPollution"] = round(next_day, 2)

def mae(values_true, values_pred):
    n = len(values_true)
    if n == 0:
        return None
    s = 0.0
    for t, p in zip(values_true, values_pred):
        s += abs(t - p)
    return s / n

def rmse(values_true, values_pred):
    n = len(values_true)
    if n == 0:
        return None
    s = 0.0
    for t, p in zip(values_true, values_pred):
        d = t - p
        s += d * d
    return math.sqrt(s / n)

# requiere numpy; sklearn es opcional (mejor rendimiento si está presente)
import numpy as np

# intentamos usar sklearn si está disponible (mejor para datasets grandes)
try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

def build_neighbors_index(sensors, max_k=7, selection_methods=('pollution','geographic')):
    """
    Precomputar vecinos para cada sensor:
      - devuelve dicts: neighbors_pollution[qid] = [(pdist, idx), ... up to max_k]
                        neighbors_geo[qid] = [(coord_dist, idx), ... up to max_k]
    idx es el índice en la lista sensors (más barato que copiar objetos).
    """
    n = len(sensors)
    id_to_idx = {s["_id"]: i for i, s in enumerate(sensors)}
    idx_to_id = {i: s["_id"] for i, s in enumerate(sensors)}

    # Prepara arrays
    # pollution vectors: shape (n, 7) or variable length -> we will pad/handle missing
    pollution_list = []
    coords_list = []
    for s in sensors:
        pl = s.get("pollutionLevels7")
        if pl is None:
            pollution_list.append([np.nan]*7)
        else:
            pollution_list.append(list(pl))
        coords_list.append(tuple(s.get("coords", (np.nan, np.nan))))
    P = np.array(pollution_list, dtype=float)   # (n,7)
    C = np.array(coords_list, dtype=float)      # (n,2)

    neighbors_pollution = {s["_id"]: [] for s in sensors}
    neighbors_geo = {s["_id"]: [] for s in sensors}

    if SKLEARN_AVAILABLE:
        # Pollution NN (euclidean on 7-dim). We ask for max_k+1 because result includes self.
        nn_poll = NearestNeighbors(n_neighbors=min(max_k+1, n), metric='euclidean')
        nn_poll.fit(np.nan_to_num(P, nan=0.0))  # nan->0 fallback; ensure behavior acceptable
        dists_p, idxs_p = nn_poll.kneighbors(np.nan_to_num(P, nan=0.0), return_distance=True)

        nn_geo = NearestNeighbors(n_neighbors=min(max_k+1, n), metric='euclidean')
        nn_geo.fit(C)
        dists_g, idxs_g = nn_geo.kneighbors(C, return_distance=True)

        for i in range(n):
            qid = idx_to_id[i]
            # build pollution neighbors, skip self (first neighbor usually self with dist 0)
            pairs_p = []
            for dist, j in zip(dists_p[i], idxs_p[i]):
                if i == j:
                    continue
                pairs_p.append((float(dist), int(j)))
                if len(pairs_p) >= max_k:
                    break
            neighbors_pollution[qid] = pairs_p

            pairs_g = []
            for dist, j in zip(dists_g[i], idxs_g[i]):
                if i == j:
                    continue
                pairs_g.append((float(dist), int(j)))
                if len(pairs_g) >= max_k:
                    break
            neighbors_geo[qid] = pairs_g

    else:
        # Fallback: vectorized pairwise distances (cost O(n^2) memory/time for large n)
        # Use this only para n moderado (<~5000). Otherwise instalar sklearn.
        # Pollution distances
        # compute squared distances efficiently
        # handle NaNs by replacing them with zeros (or better: mask)
        P0 = np.nan_to_num(P, nan=0.0)
        # coords
        C0 = np.nan_to_num(C, nan=0.0)
        # compute pairwise distances
        # pollution
        diff_p = P0[:, None, :] - P0[None, :, :]  # shape n,n,7 (may be large)
        dists_p = np.sqrt(np.sum(diff_p * diff_p, axis=2))  # n x n
        np.fill_diagonal(dists_p, np.inf)
        # geographic
        diff_g = C0[:, None, :] - C0[None, :, :]
        dists_g = np.sqrt(np.sum(diff_g * diff_g, axis=2))
        np.fill_diagonal(dists_g, np.inf)

        for i in range(n):
            qid = idx_to_id[i]
            idxs_p = np.argsort(dists_p[i])[:max_k]
            neighbors_pollution[qid] = [(float(dists_p[i, j]), int(j)) for j in idxs_p]

            idxs_g = np.argsort(dists_g[i])[:max_k]
            neighbors_geo[qid] = [(float(dists_g[i, j]), int(j)) for j in idxs_g]

    # return also useful helpers
    return {
        "id_to_idx": id_to_idx,
        "idx_to_id": idx_to_id,
        "neighbors_pollution": neighbors_pollution,
        "neighbors_geo": neighbors_geo,
        "P": P,
        "C": C
    }


def predict_next_day_knn_with_neighbors(sensors, neighbors_index, k=5, selection_method='pollution', weighting='uniform'):
    """
    Versión que usa vecinos precomputados (neighbors_index).
    neighbors_pollution[qid] -> list of (pdist, idx)
    neighbors_geo[qid] -> list of (coord_dist, idx)
    """
    id_to_idx = neighbors_index["id_to_idx"]
    idx_to_id = neighbors_index["idx_to_id"]
    neigh_poll = neighbors_index["neighbors_pollution"]
    neigh_geo = neighbors_index["neighbors_geo"]

    true_vals = []
    pred_vals = []

    for s in sensors:
        qid = s["_id"]
        true = s.get("nextDayPollution")
        if true is None:
            continue

        if selection_method == 'pollution':
            neigh = neigh_poll.get(qid, [])[:k]
            # neigh: (pdist, idx)
            weights = []
            vals = []
            for pdist, idx in neigh:
                nb = sensors[idx]
                v = nb.get("nextDayPollution")
                if v is None:
                    continue
                vals.append(v)
                if weighting == 'uniform':
                    weights.append(1.0)
                elif weighting == 'pollution_distance':
                    weights.append(1.0 / (1.0 + pdist))
                else:  # geo_distance fallback
                    coordd = float(np.linalg.norm(np.array(s.get("coords", (0,0))) - np.array(nb.get("coords", (0,0)))))
                    weights.append(1.0 / (1.0 + coordd))
        else:
            neigh = neigh_geo.get(qid, [])[:k]
            weights = []
            vals = []
            for coordd, idx in neigh:
                nb = sensors[idx]
                v = nb.get("nextDayPollution")
                if v is None:
                    continue
                vals.append(v)
                if weighting == 'uniform':
                    weights.append(1.0)
                elif weighting == 'geo_distance':
                    weights.append(1.0 / (1.0 + coordd))
                else:  # pollution_distance fallback
                    # compute pollution euclid on the fly (cheap since k small)
                    pdist = float(np.linalg.norm(np.array(s.get("pollutionLevels7", [0]*7)) - np.array(nb.get("pollutionLevels7", [0]*7))))
                    weights.append(1.0 / (1.0 + pdist))

        if not vals or not weights:
            continue
        total_w = sum(weights)
        if total_w == 0:
            continue
        pred = sum(w * v for w, v in zip(weights, vals)) / total_w
        true_vals.append(true)
        pred_vals.append(pred)
    return true_vals, pred_vals


def evaluate_knn_performance(sensors, ks=[1,3,5,7], selection_methods=['pollution','geographic'], weightings=['uniform','pollution_distance','geo_distance']):
    # Precompute neighbors once with max_k = max(ks)
    max_k = max(ks)
    neigh_idx = build_neighbors_index(sensors, max_k=max_k, selection_methods=selection_methods)

    resultsSel = {}
    for sel in selection_methods:
        resultsWeight = {}
        for w in weightings:
            resultsK = {}
            # We can compute for each k using the SAME neighbors index
            for k in ks:
                truev, predv = predict_next_day_knn_with_neighbors(sensors, neigh_idx, k=k, selection_method=sel, weighting=w)
                m = mae(truev, predv)
                r = rmse(truev, predv)
                resultsK[k] = {
                    "MAE": m,
                    "RMSE": r,
                    "n_points": len(truev)
                }
            resultsWeight[w] = resultsK
        resultsSel[sel] = resultsWeight

    return resultsSel

def test_geo_distance_effect_optimized(sensors, neighbors_index, k=5, thresholds=[5,15,30,60], selection_method='pollution', weighting='uniform'):
    """
    Versión optimizada de test_geo_distance_effect que reutiliza neighbors_index.
    neighbors_index debe provenir de build_neighbors_index(sensors, max_k=20, ...)
    neighbors_index["neighbors_pollution"][qid] -> list of (pdist, idx)
    neighbors_index["C"] -> array shape (n,2) de coords
    neighbors_index["id_to_idx"] -> map _id -> idx
    """
    out = {}
    id_to_idx = neighbors_index["id_to_idx"]
    neigh_poll = neighbors_index["neighbors_pollution"]
    C = neighbors_index["C"]  # numpy array (n,2)
    
    # Precompute for speed: map qid -> list[(pdist, coordd, idx)] using neigh_poll candidates
    # This avoids recomputing coordd each threshold loop.
    precomputed = {}
    for s in sensors:
        qid = s["_id"]
        i = id_to_idx[qid]
        cand = neigh_poll.get(qid, [])  # list of (pdist, idx)
        combos = []
        for pdist, idx in cand:
            # compute coord distance between sensor i and neighbor idx using C array
            coordd = float(np.linalg.norm(C[i] - C[idx]))
            combos.append((float(pdist), coordd, int(idx)))
        # keep combos sorted by pdist (ascending) so choosing top-k is faster
        combos.sort(key=lambda x: x[0])
        precomputed[qid] = combos

    # Now for each threshold we scan the short candidate lists
    for th in thresholds:
        true_vals = []
        pred_vals = []
        for s in sensors:
            qid = s["_id"]
            true_val = s.get("nextDayPollution")
            if true_val is None:
                continue  # no ground truth; saltar

            combos = precomputed.get(qid, [])
            # filter by coord distance threshold
            filtered = [(pdist, coordd, idx) for pdist, coordd, idx in combos if coordd <= th]
            if not filtered:
                continue

            # already sorted by pdist, take up to k best
            chosen = filtered[:k]
            weights = []
            vals = []
            for pdist, coordd, idx in chosen:
                nb = sensors[idx]
                v = nb.get("nextDayPollution")
                if v is None:
                    continue
                vals.append(v)
                if weighting == 'uniform':
                    weights.append(1.0)
                elif weighting == 'pollution_distance':
                    weights.append(1.0 / (1.0 + pdist))
                else:  # geo_distance
                    weights.append(1.0 / (1.0 + coordd))

            # filter out None values already done; check
            if not vals or not weights:
                continue
            total_w = sum(weights)
            if total_w <= 0:
                continue
            pred = sum(w * v for w, v in zip(weights, vals)) / total_w

            true_vals.append(true_val)
            pred_vals.append(pred)

        out[th] = {
            "MAE": mae(true_vals, pred_vals),
            "RMSE": rmse(true_vals, pred_vals),
            "n_points": len(true_vals)
        }
    return out











