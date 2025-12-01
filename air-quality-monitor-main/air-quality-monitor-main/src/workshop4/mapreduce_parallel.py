import multiprocessing
import time
import random
import math
from collections import defaultdict, Counter
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# ------------------------------
# Shuffle (agrupación) simple
# ------------------------------
def shuffle(mapped):
    """
    Agrupa una lista de tuplas (clave, valor) en un diccionario:
    { clave: [valores,...], ... }
    """
    groups = {}
    for k, v in mapped:
        if k in groups:
            groups[k].append(v)
        else:
            groups[k] = [v]
    return groups

def _shuffle_bucket(pairs):
    """
    pairs: list of (k, v)
    devuelve dict: k -> [v,...]
    """
    d = {}
    for k, v in pairs:
        d.setdefault(k, []).append(v)
    return d

def _merge_dicts_sum_semantics(dest, src):
    """
    Mezcla src en dest in-place.
    Si el valor es numérico (int/float), suma; si no, sobrescribe.
    """
    for k, v in src.items():
        if isinstance(v, (int, float)):
            dest[k] = dest.get(k, 0) + v
        else:
            dest[k] = v
    return dest

def testMapReduce(input_data, worker_configs):
    df, counts, averages, max_city, min_city = run_benchmark(input_data=input_data, worker_configs=worker_configs)
        
    # Asegurarnos de que df no esté vacío
    if df.empty:
        return {
            "chart_data" : pd.DataFrame(),
            "counts" : pd.DataFrame(counts),
            "averages" : pd.DataFrame(averages),
            "max_city" : max_city,
            "min_city" : min_city
        }
        
    # Obtener tiempos como lista (convertir la Series a lista)
    serial_time = df["Serial time (s)"].iloc[0]
    parallel_times = df["Parallel time (s)"].tolist()
        
    # Etiquetas: "Serial" y luego "<m+r> workers" por cada fila
    labels = ["Serial"] + [f"{m+r} workers" for m, r in zip(df["Map workers"], df["Reduce workers"])]
        
    # Construir DataFrame con índice = labels y una columna 'Time (s)'
    chart_data = pd.DataFrame({"Time (s)": [serial_time] + parallel_times}, index=labels)

    result = {
        "chart_data" : chart_data,
        "counts" : pd.DataFrame(counts),
        "averages" : pd.DataFrame(averages),
        "max_city" : max_city,
        "min_city" : min_city
    }
        
    return result

# ------------------------------
# Algoritmo 1: Contador de Lecturas
# ------------------------------
def map_count_reads(records):
    """
    Map: por cada registro emite (ubicacion_sensor, 1)
    Usa 'sensorLocation' como ubicación.
    """
    out = []
    for r in records:
        loc = r.get("sensorLocation")
        if loc is not None:
            out.append((loc, 1))
    return out

def reduce_count_reads(shuffled):
    """
    Reduce: suma los 1s por ubicación -> total de lecturas.
    Devuelve un diccionario {ubicacion: total}
    """
    df = []
    for k, values in shuffled.items():
        s = 0
        for val in values:
            s += val
        df.append({
            "Ciudad": k,
            "Conteo": s,
        })
    return df

def top_n_sensors(counts, n=10):
    """
    Devuelve una lista con las n ubicaciones con más lecturas:
    [(ubicacion, conteo), ...]
    """
    items = list(counts.items())
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[:n]

# ------------------------------
# Algoritmo 2: Promedio de Contaminación (AQI)
# ------------------------------
def map_avg_aqi(records):
    """
    Map: por cada registro emite (ciudad, aqiValue)
    Usa 'sensorLocation' y 'aqiValue'.
    """
    out = []
    for r in records:
        loc = r.get("sensorLocation")
        aqi = r.get("aqiValue")
        if loc is not None and (isinstance(aqi, int) or isinstance(aqi, float)):
            out.append((loc, float(aqi)))
    return out

def reduce_avg_aqi(shuffled):
    """
    Reduce: calcula promedio por ciudad.
    Devuelve {ciudad: promedio}
    """
    df = []
    averages = {}
    for k, vals in shuffled.items():    # Bucle (k, vals), se gestiona por las keys 
        average = {}
        total = 0.0
        count = 0
        for v in vals:        # Bucle para recorrer todos los valores 'vals' de la key 'k'
            total += v
            count += 1
        if count > 0:
            average[k] = total / count
            averages[k] = total / count
        else:
            average[k] = float('nan')
            averages[k] = float('nan')
        df.append({
            "Ciudad": k,
            "Promedio": average.get(k),
        })
    return df

def city_with_max_min(averages):
    """
    Retorna ((ciudad_max, val), (ciudad_min, val)).
    Si averages está vacío retorna (("N/A", nan), ("N/A", nan)).
    """
    if not averages:
        return (("N/A", float('nan')), ("N/A", float('nan')))
    # Encontrar max y min manualmente
    primer_elemento = averages[0]
    ciudad = primer_elemento.get("Ciudad")
    promedio = primer_elemento.get("Promedio")
    
    max_item = (ciudad, promedio)
    min_item = (ciudad, promedio)
    for average in averages:
        promedio = average.get("Promedio")
        if promedio > max_item[1]:
            max_item = (average.get("Ciudad"), promedio)
        if promedio < min_item[1]:
            min_item = (average.get("Ciudad"), promedio)
    return max_item, min_item

# ------------------------------
# Map para cada fuente
# ------------------------------
def map_pollution(records, key_field="sensorLocation"):
    """
    Map para lecturas de contaminación.
    Emite (key_field, {'type':'pollution', 'data': <registro_sin_clave>})
    """
    out = []
    for r in records:
        key = r.get(key_field)
        if key is None:
            continue
        # copia del registro sin romper el original (opcional)
        data = {}
        for kk, vv in r.items():
            if kk != key_field:
                data[kk] = vv
        out.append((key, {"type": "pollution", "data": data}))
    return out

def map_weather(records, key_field="sensorLocation"):
    """
    Map para lecturas de clima.
    Emite (key_field, {'type':'weather', 'data': <registro_sin_clave>})
    """
    out = []
    for r in records:
        key = r.get(key_field)
        if key is None:
            continue
        data = {}
        for kk, vv in r.items():
            if kk != key_field:
                data[kk] = vv
        out.append((key, {"type": "weather", "data": data}))
    return out

# ------------------------------
# Reduce: realizar el join
# ------------------------------
def reduce_join(shuffled, join_type="inner"):
    """
    join_type: 'inner' (default), 'left', 'right', 'outer'
    - inner: solo claves que tengan ambos tipos
    - left: todas las pollution, si no hay weather -> weather = None
    - right: todas las weather, si no hay pollution -> pollution = None
    - outer: todas las claves; faltantes -> None
    Retorna lista de reportes combinados: cada reporte es dict con keys:
      { 'location': clave, 'pollution': <dict or None>, 'weather': <dict or None> }
    Nota: si hay múltiples registros de cada tipo para una ubicación se hace cross-product.
    """
    reports = []

    for key, values in shuffled.items():
        polls = []
        weathers = []
        for item in values:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            d = item.get("data")
            if t == "pollution":
                polls.append(d)
            elif t == "weather":
                weathers.append(d)

        if join_type == "inner":
            if not polls or not weathers:
                continue
            # cross-product
            for p in polls:
                for w in weathers:
                    reports.append({"location": key, "pollution": p, "weather": w})
        elif join_type == "left":
            if polls:
                if weathers:
                    for p in polls:
                        for w in weathers:
                            reports.append({"location": key, "pollution": p, "weather": w})
                else:
                    for p in polls:
                        reports.append({"location": key, "pollution": p, "weather": None})
        elif join_type == "right":
            if weathers:
                if polls:
                    for p in polls:
                        for w in weathers:
                            reports.append({"location": key, "pollution": p, "weather": w})
                else:
                    for w in weathers:
                        reports.append({"location": key, "pollution": None, "weather": w})
        elif join_type == "outer":
            if polls and weathers:
                for p in polls:
                    for w in weathers:
                        reports.append({"location": key, "pollution": p, "weather": w})
            elif polls and not weathers:
                for p in polls:
                    reports.append({"location": key, "pollution": p, "weather": None})
            elif weathers and not polls:
                for w in weathers:
                    reports.append({"location": key, "pollution": None, "weather": w})
        else:
            # join_type desconocido -> tratar como inner
            if polls and weathers:
                for p in polls:
                    for w in weathers:
                        reports.append({"location": key, "pollution": p, "weather": w})

    return reports

# ------------------------------
# Función de utilidad: pipeline completo
# ------------------------------
def mapreduce_join(pollution_records, weather_records, key_field="sensorLocation", join_type="inner"):
    mapped = []
    mapped.extend(map_pollution(pollution_records, key_field=key_field))
    mapped.extend(map_weather(weather_records, key_field=key_field))
    shuffled = shuffle(mapped)
    reports = reduce_join(shuffled, join_type=join_type)
    return reports

# ------------------------------
# Map: contar sensores por ciudad
# ------------------------------
def map_city_sensors(records):
    """
    Emite (ciudad, 1) por cada registro que represente un sensor/lectura.
    """
    out = []
    for r in records:
        city = r.get("sensorLocation")
        if city is not None:
            out.append((city, 1))
    return out                      # En resumen, guarda en tuplas (ciudad, 1), para marcar que ciudades se leyeron

# Reduce genérico: sumar valores por clave
def reduce_sum(shuffled):
    totals = {}
    for k, vals in shuffled.items():
        s = 0
        for v in vals:
            s += v
        totals[k] = s
    return totals

# ------------------------------
# Util: Amdahl (speedup)
# ------------------------------
def effective_nodes_amdahl(nodes, serial_fraction):
    """
    Devuelve el speedup (aceleración relativa) según Amdahl:
      speedup = 1 / (s + (1-s)/N)
    - nodes: número de nodos (N >= 1)
    - serial_fraction: fracción serial s en [0,1]
    """
    if nodes <= 0:
        return 0.0
    if serial_fraction < 0.0:
        serial_fraction = 0.0
    if serial_fraction > 1.0:
        serial_fraction = 1.0
    s = float(serial_fraction)
    N = float(nodes)
    # speedup
    denom = s + (1.0 - s) / N
    if denom == 0.0:
        return float('inf')
    return 1.0 / denom

# ------------------------------
# Pipeline de costos
# ------------------------------
def estimate_monitoring_costs(records,
                              cost_per_sensor_per_year=50.0,
                              fixed_cost_per_city_per_year=1000.0,
                              record_size_bytes=800,      # tamaño estimado por lectura JSON
                              storage_overhead=1.10,      # 10% overhead (metadatos, índices)
                              nodes_cost_per_hour=0.20,   # costo por nodo/hora (USD)
                              processing_time_per_record_sec=0.05,
                              serial_fraction=0.05):
    """
    Parámetros:
      - cost_per_sensor_per_year: costo anual por mantener un sensor (licencias, mantenimiento)
      - fixed_cost_per_city_per_year: costo fijo por ciudad (infraestructura local)
      - record_size_bytes: tamaño aproximado de cada registro
      - storage_overhead: multiplicador para overhead
      - nodes_cost_per_hour: costo por nodo/hora para estimaciones de procesamiento
      - processing_time_per_record_sec: tiempo medio (segundos) para procesar una lectura
    Devuelve diccionario con estimaciones.
    """
    # MapReduce: contar sensores por ciudad
    mapped = map_city_sensors(records)
    shuffled = shuffle(mapped)
    sensors_by_city = reduce_sum(shuffled)   # {city: num_sensors}

    # Estadísticas básicas
    num_cities = len(sensors_by_city)

    total_sensors = 0                       # Promedio de sensores por ciudad
    for c in sensors_by_city:
        total_sensors += sensors_by_city[c]
    avg_sensors_per_city = (total_sensors / num_cities) if num_cities > 0 else 0.0

    # 1) Costos anuales de monitoreo por ciudad y total
    costs_per_city = {}
    total_cost = 0.0
    for city, ns in sensors_by_city.items():    # ns: número de sensores
        cost = fixed_cost_per_city_per_year + ns * cost_per_sensor_per_year
        costs_per_city[city] = cost
        total_cost += cost

    # 2) Comparación almacenamiento: hourly vs daily (por año)
    # Asumimos:
    #  - hourly: 24 lecturas/día por sensor
    #  - daily: 1 lectura/día por sensor
    days_per_year = 365
    hourly_per_sensor_per_year = 24 * days_per_year
    daily_per_sensor_per_year = 1 * days_per_year

    storage_hourly_bytes_per_sensor = hourly_per_sensor_per_year * record_size_bytes * storage_overhead
    storage_daily_bytes_per_sensor  = daily_per_sensor_per_year  * record_size_bytes * storage_overhead

    total_storage_hourly_bytes = storage_hourly_bytes_per_sensor * total_sensors
    total_storage_daily_bytes  = storage_daily_bytes_per_sensor  * total_sensors

    # Convertir a GB
    BYTES_PER_GB = 1024.0**3
    storage_hourly_gb = total_storage_hourly_bytes / BYTES_PER_GB
    storage_daily_gb  = total_storage_daily_bytes  / BYTES_PER_GB

    # 3) Estimación de costos de procesamiento para distintos tamaños de red (nodos)
    # Calculamos tiempo total de procesamiento sec = total_records * processing_time_per_record_sec
    total_records_per_year = hourly_per_sensor_per_year * total_sensors  # usar horario como caso "máximo"
    total_proc_seconds = total_records_per_year * processing_time_per_record_sec

    # Escenarios de nodos a evaluar
    node_scenarios = [1, 2, 5, 10, 20, 50]
    processing_estimates = {}
    for nodes in node_scenarios:
        speedup = effective_nodes_amdahl(nodes, serial_fraction)
        if speedup <= 0:
            # evita división por cero
            wall_time_seconds = float('inf')
        else:
            wall_time_seconds = total_proc_seconds / speedup
        wall_time_hours = wall_time_seconds / 3600.0 if wall_time_seconds != float('inf') else float('inf')
        cost_processing = nodes * wall_time_hours * nodes_cost_per_hour if wall_time_hours != float('inf') else float('inf')
        processing_estimates[nodes] = {
            "speedup": speedup,
            "wall_time_hours": wall_time_hours,
            "cost_usd": cost_processing
        }

    result = {
        "num_cities": num_cities,
        "total_sensors": total_sensors,
        "avg_sensors_per_city": avg_sensors_per_city,
        "costs_per_city": costs_per_city,
        "total_monitoring_cost_usd_per_year": total_cost,
        "storage_hourly_gb_per_year": storage_hourly_gb,
        "storage_daily_gb_per_year": storage_daily_gb,
        "processing_total_seconds_per_year": total_proc_seconds,
        "processing_estimates_by_nodes": processing_estimates
    }
    return result

# Algoritmo 5: Rendimiento del Procesamiento de Calidad del Aire

# ------------------------------
# Map para procesamiento: genera (key, processing_time_sec)
# key será 'proc' (agregamos todo), values son tiempos por lectura
# ------------------------------
def map_processing(records, base_time_per_record=0.05, event_type="normal", event_multipliers=None):
    """
    event_type: 'normal' o 'critical'
    event_multipliers: dict opcional para ajustar tiempos, e.g. {'critical': 2.5}
    """
    if event_multipliers is None:
        event_multipliers = {"normal": 1.0, "critical": 2.5}

    multiplier = event_multipliers.get(event_type, 1.0)
    out = []
    for r in records:
        # si registro incluye campo que sugiere evento crítico local, se podría ajustar por registro;
        # en este ejemplo aplicamos el mismo multiplicador global del evento
        proc_time = float(base_time_per_record) * multiplier
        out.append(("proc", proc_time))
    return out

# Reduce: sumar tiempos (igual que reduce_sum pero para la clave 'proc')
def reduce_sum_basic(shuffled):
    totals = {}
    for k, vals in shuffled.items():
        s = 0.0
        for v in vals:
            s += v
        totals[k] = s
    return totals

# Pipeline de rendimiento
def evaluate_performance(records,
                         nodes_list=[1,2,5,10],
                         base_time_per_record=0.05,
                         event_types=("normal", "critical"),
                         event_multipliers=None,
                         alert_rate_multiplier=3.0,
                         nodes_cost_per_hour=0.20,
                         serial_fraction=0.05):
    """
    - alert_rate_multiplier: durante una alerta crítica, tasa de llegada puede multiplicarse (ej. 3x)
    - event_multipliers: factor de tiempo por registro según el tipo de evento
    Devuelve métricas de rendimiento por (event_type, nodes).
    """
    if event_multipliers is None:
        event_multipliers = {"normal": 1.0, "critical": 2.5}

    results = {}

    # número base de registros
    base_num_records = len(records)

    for event in event_types:
        # Ajuste del número de registros si evento crítico incrementa la llegada
        if event == "critical":
            num_records = int(base_num_records * alert_rate_multiplier)
        else:
            num_records = base_num_records

        # Generamos mapeo (simulado) con tiempos por lectura ya ajustadas
        # Para eficiencia no construimos lista enorme si num_records grande; pero aquí lo hacemos simple
        # creando una lista de tuplas
        mapped = []
        for i in range(num_records):
            # Podríamos usar datos distintos por registro, aquí uniformizamos
            mapped.append(("proc", float(base_time_per_record) * event_multipliers.get(event, 1.0)))

        shuffled = shuffle(mapped)
        reduced = reduce_sum_basic(shuffled)
        total_proc_seconds = reduced.get("proc", 0.0)

        # Evaluar para cada cantidad de nodos
        results[event] = {}
        for nodes in nodes_list:
            speedup = effective_nodes_amdahl(nodes, serial_fraction)
            if speedup <= 0:
                wall_time_seconds = float('inf')
            else:
                wall_time_seconds = total_proc_seconds / speedup
            wall_time_hours = wall_time_seconds / 3600.0 if wall_time_seconds != float('inf') else float('inf')
            cost = nodes * wall_time_hours * nodes_cost_per_hour if wall_time_hours != float('inf') else float('inf')
            throughput = (num_records / wall_time_seconds) if wall_time_seconds not in (0.0, float('inf')) else 0.0
            results[event][nodes] = {
                "num_records": num_records,
                "total_proc_seconds": total_proc_seconds,
                "speedup": speedup,
                "wall_time_hours": wall_time_hours,
                "cost_usd": cost,
                "throughput_records_per_sec": throughput
            }

    return results

# ------------------------------
# ------ Paralelismo simple -----
# ------------------------------
def _map_worker(map_queue, reduce_queue, map_func, map_args, map_kwargs):
    """Worker que toma chunks de registros y manda un dict combinado al reduce_queue."""
    proc_name = multiprocessing.current_process().name
    while True:
        task = map_queue.get()
        if task is None:
            break
        records_chunk = task
        # ejecutar map sobre el chunk
        if map_args is None:
            mapped = map_func(records_chunk)
        else:
            mapped = map_func(records_chunk, *map_args, **(map_kwargs or {}))
        # combiner simple: agrupa valores por clave para reducir tráfico
        combined = {}
        for k, v in mapped:
            if k in combined:
                combined[k].append(v)
            else:
                combined[k] = [v]
        reduce_queue.put(combined)
        # pequeño log opcional
        # print(f"[{proc_name}] mapped chunk len={len(records_chunk)} -> {len(combined)} keys")
        time.sleep(random.uniform(0.0, 0.01))

def _reduce_worker(reduce_queue, result_queue, reduce_func, reduce_args, reduce_kwargs):
    """Worker que acumula dicts {k:[v...]} y al final aplica reduce_func."""
    proc_name = multiprocessing.current_process().name
    accumulated = defaultdict(list)
    while True:
        task = reduce_queue.get()
        if task is None:
            # finalizar: aplicar reduce sobre accumulated
            result = reduce_func(accumulated) if reduce_args is None else reduce_func(accumulated, *reduce_args, **(reduce_kwargs or {}))
            result_queue.put(result)
            # print(f"[{proc_name}] reduced -> {len(result)} entries")
            break
        for k, vals in task.items():
            accumulated[k].extend(vals)
        # print(f"[{proc_name}] received chunk with {len(task)} keys")
        time.sleep(random.uniform(0.0, 0.01))

def parallel_mapreduce(
    records,
    map_func,
    reduce_func,
    num_map_tasks=4,
    num_shuffle_tasks=2,
    num_reduce_tasks=2,
    chunking_override=None,
):
    """
    pipeline paralelizado en 3 fases:
      1) map: procesa chunks en procesos (cada map_func devuelve list of (k,v))
      2) shuffle: particiona por hash(k) y agrupa cada bucket (paralelizado)
      3) reduce: aplica reduce_func a cada bucket (paralelizado) y merge final

    Args:
      records: lista de registros
      map_func(chunk) -> list of (k, v)
      reduce_func(shuffled_dict) -> dict (resultado por clave)  -- debe procesar el dict local de su bucket
      num_map_tasks/num_shuffle_tasks/num_reduce_tasks: paralelismos por fase
      chunking_override: si quieres pasar un chunk_size fijo (int); si None se calcula automáticamente.
    Returns:
      dict final con k -> reduced_value
    """

    # sanity
    n = len(records)
    if n == 0:
        return {}

    # -----------------
    # 1) MAP stage
    # -----------------
    # chunkeo simple
    if chunking_override is None:
        chunk_size = int(math.ceil(n / float(num_map_tasks)))
    else:
        chunk_size = int(chunking_override)
    chunks = [records[i:i+chunk_size] for i in range(0, n, chunk_size)]

    mapped_pairs = []  # lista de (k,v) provenientes de todos los map workers
    with ProcessPoolExecutor(max_workers=num_map_tasks) as map_executor:
        map_futures = [map_executor.submit(map_func, ch) for ch in chunks]
        for f in as_completed(map_futures):
            try:
                res = f.result()  # debe ser lista de (k,v)
                if res:
                    mapped_pairs.extend(res)
            except Exception:
                print("Exception in map worker:")
                traceback.print_exc()
                raise

    # -----------------
    # 2) SHUFFLE stage (paralelizado)
    # -----------------
    # Particionar mapped_pairs en buckets por hash(key)
    num_buckets = max(1, int(num_shuffle_tasks))
    buckets = [[] for _ in range(num_buckets)]
    for k, v in mapped_pairs:
        idx = (hash(k) & 0x7FFFFFFF) % num_buckets
        buckets[idx].append((k, v))

    # Cada bucket se transforma en dict {k: [v,...]} (esto puede correr en paralelo)
    bucket_dicts = []
    with ProcessPoolExecutor(max_workers=num_shuffle_tasks) as shuffle_executor:
        shuffle_futures = [shuffle_executor.submit(_shuffle_bucket, b) for b in buckets]
        for f in as_completed(shuffle_futures):
            try:
                bucket_dicts.append(f.result())
            except Exception:
                print("Exception in shuffle worker:")
                traceback.print_exc()
                raise

    # -----------------
    # 3) REDUCE stage (paralelizado)
    # -----------------
    # Cada bucket_dict es independiente; aplicamos reduce_func por bucket (puede ser costoso)
    partial_results = []
    with ProcessPoolExecutor(max_workers=num_reduce_tasks) as reduce_executor:
        reduce_futures = [reduce_executor.submit(reduce_func, bd) for bd in bucket_dicts]
        for f in as_completed(reduce_futures):
            try:
                partial_results.append(f.result())  # cada resultado es dict
            except Exception:
                print("Exception in reduce worker:")
                traceback.print_exc()
                raise

    # -----------------
    # Merge final
    # -----------------
    final = {}
    for part in partial_results:
        if not isinstance(part, dict):
            continue
        _merge_dicts_sum_semantics(final, part)

    return final

# ------------------------------
# Ejemplos de uso paralelo (manteniendo tus ejemplos)
# ------------------------------
# Datos de ejemplo (simulan los registros del JSON que enviaste)


import time
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Helpers para medir rendimiento
# ---------------------------------------------------------

def measure_serial(records):
    t0 = time.time()
    print("=== Serial (original) results ===")
    # Algoritmo 1: contador de lecturas (serial)
    mapped1 = map_count_reads(records)
    shuffled1 = shuffle(mapped1)
    counts = reduce_count_reads(shuffled1)
    print("Conteo de lecturas por ubicación (serial):", counts)

    # Algoritmo 2: promedio AQI por ciudad (serial)
    mapped2 = map_avg_aqi(records)
    shuffled2 = shuffle(mapped2)
    df_averages = reduce_avg_aqi(shuffled2)
    print("Promedio AQI por ciudad (serial):", df_averages)
    max_city, min_city = city_with_max_min(df_averages)
    print(f"Mayor: {max_city}, Menor: {min_city}")
    t1 = time.time()
    return t1 - t0, counts, df_averages, max_city, min_city

def measure_parallel(records, num_map=2, num_reduce=2, num_shuffle=2):
    t0 = time.time()
    print("\n=== Paralelo (demo) ===")
    # Parallel: count reads (this is additive so puede usar >1 reduce)
    par_counts = parallel_mapreduce(records, map_count_reads, reduce_count_reads, num_map_tasks=num_map, num_shuffle_tasks=num_shuffle, num_reduce_tasks=num_reduce)
    print("Conteo de lecturas por ubicación (paralelo):", par_counts)

    # Parallel: average AQI -> usar 1 reduce worker para evitar problemas de fusion de promedios
    par_avg = parallel_mapreduce(records, map_avg_aqi, reduce_avg_aqi, num_map_tasks=num_map, num_shuffle_tasks=num_shuffle, num_reduce_tasks=num_reduce)
    print("Promedio AQI por ciudad (paralelo, 1 reduce):", par_avg)
    max_city_p, min_city_p = city_with_max_min(par_avg)
    print(f"Mayor (paralelo): {max_city_p}, Menor (paralelo): {min_city_p}")
    t1 = time.time()
    return t1 - t0

# ---------------------------------------------------------
# Ejecutar pruebas
# ---------------------------------------------------------

def run_benchmark(input_data, worker_configs):
    results = []

    # medir serial
    serial_time, counts, averages, max_city, min_city = measure_serial(input_data)

    for (m, r, s) in worker_configs:
        parallel_time = measure_parallel(input_data, m, r, s)

        speedup = serial_time / parallel_time
        efficiency = speedup / (m + r + s)

        results.append({
            "Map workers": m,
            "Reduce workers": r,
            "Shuffle workers": s,
            "Total workers": m + r + s,
            "Serial time (s)": serial_time,
            "Parallel time (s)": parallel_time,
            "Speedup": speedup,
            "Efficiency": efficiency,
        })

    return pd.DataFrame(results), counts, averages, max_city, min_city

# ---------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------

if __name__ == "__main__":

    ### ALGORITMO 4
    # Datos de ejemplo: muchos sensores en algunas ciudades
    example_records = [
        {"sensorLocation": "Bogota"}, {"sensorLocation": "Bogota"}, {"sensorLocation": "Bogota"},
        {"sensorLocation": "Medellin"}, {"sensorLocation": "Medellin"},
        {"sensorLocation": "Cali"}, {"sensorLocation": "Cali"}, {"sensorLocation": "Cali"}, {"sensorLocation": "Cali"},
        {"sensorLocation": "Barranquilla"}
    ]

    costs = estimate_monitoring_costs(example_records)
    print("Resumen costos (Algoritmo 4):")
    print("  Ciudades detectadas:", costs["num_cities"])
    print("  Total sensores:", costs["total_sensors"])
    print("  Costo total estimado (USD/año): {:.2f}".format(costs["total_monitoring_cost_usd_per_year"]))
    print("  Almacenamiento anual (hourly) GB: {:.2f}".format(costs["storage_hourly_gb_per_year"]))
    print("  Almacenamiento anual (daily)  GB: {:.2f}".format(costs["storage_daily_gb_per_year"]))
    print("  Estimaciones de procesamiento (por nodos):")
    for n, info in costs["processing_estimates_by_nodes"].items():
        print("    nodos={}: tiempo(h)={:.2f}, costo_usd={:.2f}, nodos_efectivos={:.1f}".format(
            n, info["wall_time_hours"], info["cost_usd"], info["speedup"]))

    ### ALGORITMO 5
    # dataset base de ejemplo (10 registros)
    sample_records = [{"sensorLocation": "Bogota", "aqiValue": 100}] * 10

    perf = evaluate_performance(sample_records,
                                nodes_list=[1,2,5,10],
                                base_time_per_record=0.04,
                                event_types=("normal", "critical"),
                                event_multipliers={"normal": 1.0, "critical": 3.0},
                                alert_rate_multiplier=4.0,
                                nodes_cost_per_hour=0.25)

    print("\nResumen rendimiento (Algoritmo 5):")
    for event in perf:
        print("Evento:", event)
        for n in perf[event]:
            info = perf[event][n]
            print("  nodos={}: registros={}, tiempo(h)={:.3f}, costo=${:.2f}, throughput={:.2f} rec/s".format(
                n, info["num_records"], info["wall_time_hours"], info["cost_usd"], info["throughput_records_per_sec"]
            ))


    # Configuraciones de trabajadores a evaluar
    worker_configs = [(1, 1, 1),(2, 2, 2),(3, 2, 2),(4, 2, 2),(6, 2, 2)]

    input_data = [
        {"sensorLocation": "Bogota", "aqiValue": 120},
        {"sensorLocation": "Bogota", "aqiValue": 130},
        {"sensorLocation": "Medellin", "aqiValue": 80},
        {"sensorLocation": "Cali", "aqiValue": 200},
        {"sensorLocation": "Medellin", "aqiValue": 90},
        {"sensorLocation": "Cali", "aqiValue": 150},
        {"sensorLocation": "Bogota", "aqiValue": 100},
        {"sensorLocation": "Cali", "aqiValue": 220},
        {"sensorLocation": "Bogota", "aqiValue": 120},
        {"sensorLocation": "Bogota", "aqiValue": 130},
        {"sensorLocation": "Medellin", "aqiValue": 80},
        {"sensorLocation": "Cali", "aqiValue": 200},
        {"sensorLocation": "Medellin", "aqiValue": 90},
        {"sensorLocation": "Cali", "aqiValue": 150},
        {"sensorLocation": "Bogota", "aqiValue": 100},
        {"sensorLocation": "Cali", "aqiValue": 220},
        {"sensorLocation": "Bogota", "aqiValue": 120},
        {"sensorLocation": "Bogota", "aqiValue": 130},
        {"sensorLocation": "Medellin", "aqiValue": 80},
        {"sensorLocation": "Cali", "aqiValue": 200},
        {"sensorLocation": "Medellin", "aqiValue": 90},
        {"sensorLocation": "Cali", "aqiValue": 150},
        {"sensorLocation": "Bogota", "aqiValue": 100},
        {"sensorLocation": "Cali", "aqiValue": 220},
        {"sensorLocation": "Bogota"},   # sin aqiValue -> ignorado por map_avg_aqi
        {"aqiValue": 50},               # sin sensorLocation -> ignorado
    ]

    # Benchmark
    df = run_benchmark(
        input_data=input_data,
        worker_configs=worker_configs
    )

    print(df)

    # ---------------------------------------------------------
    # Visualizaciones estilo dashboard
    # ---------------------------------------------------------

    # --- Tiempo de ejecución ---
    plt.figure(figsize=(10,5))
    plt.bar(["Serial"], [df["Serial time (s)"].iloc[0]])
    plt.bar([f"{m+r} workers" for m,r in zip(df["Map workers"], df["Reduce workers"])],
            df["Parallel time (s)"])
    plt.ylabel("Tiempo (segundos)")
    plt.title("Tiempo de ejecución: Serial vs Paralelo")
    plt.show()

     # --- Speedup ---
    plt.figure(figsize=(10,5))
    plt.plot(df["Total workers"], df["Speedup"], marker="o")
    plt.xlabel("Número total de procesos (map + reduce)")
    plt.ylabel("Speedup")
    plt.title("Speedup al aumentar los Workers")
    plt.grid(True)
    plt.show()

     # --- Eficiencia ---
    plt.figure(figsize=(10,5))
    plt.plot(df["Total workers"], df["Efficiency"], marker="o")
    plt.xlabel("Número total de procesos")
    plt.ylabel("Eficiencia")
    plt.title("Eficiencia paralela")
    plt.grid(True)
    plt.show()
