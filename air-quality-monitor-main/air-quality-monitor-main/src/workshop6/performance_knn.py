def add_next_day_pollution(sensors, seed=999):
    random.seed(seed)
    sensorsNextDay = []
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
        sensorsNextDay.append(s)
    return sensorsNextDay        

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

def predict_next_day_knn(sensors, k=5, selection_method='pollution', weighting='uniform'):
    """
    selection_method: 'pollution' or 'geographic'
    weighting: 'uniform', 'pollution_distance' (1/(1+poll_dist)), 'geo_distance' (1/(1+coord_dist))
    Retorna (list_true, list_pred)
    """
    true_vals = []
    pred_vals = []

    sensorsND = add_next_day_pollution(sensors, seed=25)
    
    for s in sensors:
        qid = s["id"]
        if selection_method == 'pollution':
            neigh = knn_pollution(sensors, query_id=qid, k=k, use_average=False)
            # neigh: list of (pollution_distance, sensor)
            # prepare arrays
            if not neigh:
                continue
            weights = []
            vals = []
            for pdist, nb in neigh:
                vals.append(nb.get("nextDayPollution", None))
                if weighting == 'uniform':
                    weights.append(1.0)
                elif weighting == 'pollution_distance':
                    weights.append(1.0 / (1.0 + pdist))
                else:  # geo_distance fallback
                    coordd = dist2d(s["coords"], nb["coords"])
                    weights.append(1.0 / (1.0 + coordd))
        else:  # geographic selection
            neigh = knn_geographic(sensors, query_id=qid, k=k)
            if not neigh:
                continue
            weights = []
            vals = []
            for gscore, nb in neigh:
                vals.append(nb.get("nextDayPollution", None))
                if weighting == 'uniform':
                    weights.append(1.0)
                elif weighting == 'geo_distance':
                    # approximate: use coord distance from gscore components? need coords
                    coordd = dist2d(s["coords"], nb["coords"])
                    weights.append(1.0 / (1.0 + coordd))
                else:  # pollution_distance fallback: compute pollution euclid
                    pdist = euclidean(s["pollutionLevels7"], nb["pollutionLevels7"])
                    weights.append(1.0 / (1.0 + pdist))

        # filter out neighbors without nextDayPollution (shouldn't happen here)
        combined = []
        for w, v in zip(weights, vals):
            if v is None:
                continue
            combined.append((w, v))
        if not combined:
            continue
        # weighted average
        total_w = sum(w for w, _ in combined)
        pred = sum(w * v for w, v in combined) / total_w
        true_vals.append(s.get("nextDayPollution"))
        pred_vals.append(pred)
    return true_vals, pred_vals

def evaluate_knn_performance(sensors, ks=[1,3,5,7], selection_methods=['pollution','geographic'], weightings=['uniform','pollution_distance','geo_distance']):
    results = []
    for sel in selection_methods:
        for w in weightings:
            for k in ks:
                truev, predv = predict_next_day_knn(sensors, k=k, selection_method=sel, weighting=w)
                m = mae(truev, predv)
                r = rmse(truev, predv)
                results.append({
                    "selection": sel,
                    "weighting": w,
                    "k": k,
                    "MAE": m,
                    "RMSE": r,
                    "n_points": len(truev)
                })
    return results

def test_geo_distance_effect(sensors, k=5, thresholds=[5,15,30,60], selection_method='pollution', weighting='uniform'):
    """
    Prueba cómo el limitar vecinos por distancia geográfica afecta la predicción.
    Para cada threshold, selecciona vecinos por pollution-similarity pero filtra
    aquellos con coord distance <= threshold. Si un sensor no tiene vecinos en el umbral,
    lo excluye.
    """
    out = []
    for th in thresholds:
        true_vals = []
        pred_vals = []
        for s in sensors:
            qid = s["id"]
            neigh = knn_pollution(sensors, query_id=qid, k=20, use_average=False)  # obtener muchos candidatos
            # filtrar por distancia geográfica
            filtered = []
            for pdist, nb in neigh:
                coordd = dist2d(s["coords"], nb["coords"])
                if coordd <= th:
                    filtered.append((pdist, nb))
            if not filtered:
                continue
            # tomar los k mejores según polution_distance entre los filtrados
            filtered.sort(key=lambda x: x[0])
            chosen = filtered[:k]
            weights = []
            vals = []
            for pdist, nb in chosen:
                vals.append(nb.get("nextDayPollution"))
                if weighting == 'uniform':
                    weights.append(1.0)
                elif weighting == 'pollution_distance':
                    weights.append(1.0 / (1.0 + pdist))
                else:
                    coordd = dist2d(s["coords"], nb["coords"])
                    weights.append(1.0 / (1.0 + coordd))
            combined = [(w,v) for w,v in zip(weights, vals) if v is not None]
            if not combined:
                continue
            total_w = sum(w for w,_ in combined)
            pred = sum(w*v for w,v in combined) / total_w
            true_vals.append(s.get("nextDayPollution"))
            pred_vals.append(pred)
        out.append({
            "geo_threshold": th,
            "MAE": mae(true_vals, pred_vals),
            "RMSE": rmse(true_vals, pred_vals),
            "n_points": len(true_vals)
        })
    return out


