# Algoritmo 4: Costos del Sistema de Monitoreo del Aire

# ------------------------------
# Util: shuffle simple
# ------------------------------
def shuffle(mapped):
    groups = {}
    for k, v in mapped:
        if k in groups:
            groups[k].append(v)
        else:
            groups[k] = [v]
    return groups

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
                                    # NOTA: Pueden haber varias tuplas repetidas (sus records llegaron en distintos tiempos)

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
# Ejemplo de uso
# ------------------------------
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
