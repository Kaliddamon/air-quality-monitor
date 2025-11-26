from datetime import datetime, timezone

def random_quicksort(arr):
    if len(arr) <= 1:
        return arr[:]
    pivot = arr[random.randrange(0, len(arr))]
    less = []
    equal = []
    greater = []
    for x in arr:
        if x < pivot:
            less.append(x)
        elif x > pivot:
            greater.append(x)
        else:
            equal.append(x)
    return random_quicksort(less) + equal + random_quicksort(greater)

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

# NO2 breakpoints are in ppb for the 1-hour standard (EPA)
NO2_BREAKPOINTS_PPB = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]

def calculate_aqi_from_breakpoints(concentration, breakpoints):
    """Interpolación lineal genérica donde concentration y breakpoints deben usar la misma unidad."""
    c = float(concentration)
    for (C_lo, C_hi, I_lo, I_hi) in breakpoints:
        if C_lo <= c <= C_hi:
            aqi = ((I_hi - I_lo) / (C_hi - C_lo)) * (c - C_lo) + I_lo
            return int(round(aqi))
    # fuera de rangos: limitar al máximo del AQI representado o 0 si por debajo
    if c > breakpoints[-1][1]:
        return 500
    return 0

def calculate_aqi_pm25(concentration_ugm3):
    return calculate_aqi_from_breakpoints(concentration_ugm3, PM25_BREAKPOINTS)

# Para NO2, asumimos que la entrada es µg/m3 -> convertimos a ppb
def no2_ugm3_to_ppb(conc_ugm3):
    # Fórmula: ppb = (µg/m3) * 24.45 / molecular_weight
    # Molecular weight NO2 ≈ 46.0055 g/mol
    try:
        mw = 46.0055
        return (float(conc_ugm3) * 24.45) / mw
    except Exception:
        return None

def calculate_aqi_no2_from_ugm3(concentration_ugm3):
    ppb = no2_ugm3_to_ppb(concentration_ugm3)
    if ppb is None:
        return None
    return calculate_aqi_from_breakpoints(ppb, NO2_BREAKPOINTS_PPB)

def median_from_sorted(sorted_list):
    n = len(sorted_list)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return sorted_list[mid]
    else:
        return (sorted_list[mid - 1] + sorted_list[mid]) / 2.0


# ------------------------ INGESTIÓN DEL JSON REAL (incluye NO2) ------------------------

def ingest_json_readings(hash_table, readings):
    """
    readings: lista de objetos con estructura real:
      { _id, sensorLocation, airQualityData: { PM25, NO2 }, timestamp }
    """
    for rec in readings:
        sid = rec.get('_id')
        if sid is None:
            continue
        loc = rec.get('sensorLocation', 'unknown')
        aq = rec.get('airQualityData', {}) or {}
        pm25 = None
        no2 = None
        try:
            if 'PM25' in aq and aq['PM25'] not in (None, ''):
                pm25 = float(aq['PM25'])
        except Exception:
            pm25 = None
        try:
            if 'NO2' in aq and aq['NO2'] not in (None, ''):
                no2 = float(aq['NO2'])
        except Exception:
            no2 = None
        timestamp = rec.get('timestamp')

        key_id = f"sensor:{sid}"
        sensor = hash_table.get(key_id)
        if sensor is None:
            sensor = {
                '_id': sid,
                'sensorLocation': loc,
                'pm25_readings': [],
                'no2_readings': [],
                'last_timestamp': timestamp
            }
            hash_table.insert(key_id, sensor)
            key_loc = f"loc:{loc}"
            existing = hash_table.get(key_loc)
            if existing is None:
                existing = []
            already = False
            for s in existing:
                if s.get('_id') == sid:
                    already = True
                    break
            if not already:
                existing.append(sensor)
            hash_table.insert(key_loc, existing)
        sensor['last_timestamp'] = timestamp
        if pm25 is not None:
            sensor['pm25_readings'].append(pm25)
        if no2 is not None:
            sensor['no2_readings'].append(no2)


def predict_aqi_for_sensor(sensor):
    pm25_list = sensor.get('pm25_readings', []) or []
    no2_list = sensor.get('no2_readings', []) or []

    result = {
        'median_pm25': None,
        'aqi_pm25': None,
        'median_no2_ugm3': None,
        'aqi_no2': None,
        'combined_aqi': None,
        'dominant_pollutant': None,
        'sorted_pm25': [],
        'sorted_no2': []
    }

    if pm25_list:
        sorted_pm25 = random_quicksort(pm25_list)
        med_pm25 = median_from_sorted(sorted_pm25)
        aqi_pm25 = calculate_aqi_pm25(med_pm25)
        result['median_pm25'] = med_pm25
        result['aqi_pm25'] = aqi_pm25
        result['sorted_pm25'] = sorted_pm25

    if no2_list:
        sorted_no2 = random_quicksort(no2_list)
        med_no2 = median_from_sorted(sorted_no2)
        aqi_no2 = calculate_aqi_no2_from_ugm3(med_no2)
        result['median_no2_ugm3'] = med_no2
        result['aqi_no2'] = aqi_no2
        result['sorted_no2'] = sorted_no2

    # combinar: AQI final es el máximo de los AQIs individuales disponibles
    aqi_candidates = []
    if result['aqi_pm25'] is not None:
        aqi_candidates.append(('PM2.5', result['aqi_pm25']))
    if result['aqi_no2'] is not None:
        aqi_candidates.append(('NO2', result['aqi_no2']))

    if aqi_candidates:
        # escoger máximo
        dominant, combined = aqi_candidates[0]
        for pol, val in aqi_candidates:
            if val is not None and val > combined:
                dominant, combined = pol, val
        result['combined_aqi'] = combined
        result['dominant_pollutant'] = dominant

    # si no hubo lecturas, devolver None
    if result['combined_aqi'] is None:
        return None

    return result


def check_and_alert(hash_table, alert_callback=None, alert_threshold=151):
    alerts = []
    for (k, v) in hash_table.items():
        if isinstance(k, str) and k.startswith("sensor:"):
            sensor = v
            pred = predict_aqi_for_sensor(sensor)
            if pred is None:
                continue
            aqi = pred['combined_aqi']
            if aqi <= 50:
                level = 'Good'
            elif aqi <= 100:
                level = 'Moderate'
            elif aqi <= 150:
                level = 'Unhealthy for Sensitive Groups'
            elif aqi <= 200:
                level = 'Unhealthy'
            elif aqi <= 300:
                level = 'Very Unhealthy'
            else:
                level = 'Hazardous'
            if aqi >= alert_threshold:
                alert = {
                    'sensor_id': sensor.get('_id'),
                    'location': sensor.get('sensorLocation'),
                    'combined_aqi': aqi,
                    'level': level,
                    'dominant_pollutant': pred.get('dominant_pollutant'),
                    'median_pm25': pred.get('median_pm25'),
                    'aqi_pm25': pred.get('aqi_pm25'),
                    'median_no2_ugm3': pred.get('median_no2_ugm3'),
                    'aqi_no2': pred.get('aqi_no2'),
                    'last_timestamp': sensor.get('last_timestamp')
                }
                alerts.append(alert)
                if alert_callback is not None:
                    alert_callback(sensor, pred)
                else:
                    print("ALERTA AUTOMÁTICA 🔴")
                    print(f"  Sensor: {sensor.get('_id')}  Location: {sensor.get('sensorLocation')}")
                    print(f"  Combined AQI: {aqi}  Level: {level}  Dominant: {pred.get('dominant_pollutant')}")
                    print(f"  PM2.5 median: {pred.get('median_pm25')}  AQI_PM25: {pred.get('aqi_pm25')}")
                    print(f"  NO2 median (µg/m3): {pred.get('median_no2_ugm3')}  AQI_NO2: {pred.get('aqi_no2')}")
                    print(f"  Last timestamp: {sensor.get('last_timestamp')}")
                    print("-" * 80)
    return alerts


# ------------------------ DEMOSTRACIÓN con datos sintéticos ------------------------

def make_random_objectid():
    return ''.join(random.choice('0123456789abcdef') for _ in range(24))

cities = ['Bogota', 'Medellin', 'Cali', 'Barranquilla', 'Cartagena', 'Bucaramanga']

sample_readings = []
for _ in range(100):
    rec = {
        '_id': make_random_objectid(),
        'sensorLocation': random.choice(cities),
        'airQualityData': {
            'PM25': round(random.uniform(0, 500), 2),
            'NO2': round(random.uniform(0, 200), 2)  # µg/m3 assumed
        },
        'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S %z")
    }
    sample_readings.append(rec)

ht = HashTable()
ingest_json_readings(ht, sample_readings)

print("HashTable estado:", ht)
print("Número de claves en HashTable (incluye loc:... y sensor:...):", len(ht.keys()))
print("Primeros 6 items (resumen):")
pprint.pprint(ht.items()[:6])

print("\nEjecutando predicciones y detectando alertas (AQI combinado >= 151):\n")
alerts = check_and_alert(ht)

print(f"\nTotal sensores con alertas: {len(alerts)}")
if alerts:
    print("Primeras 5 alertas (resumen):")
    pprint.pprint(alerts[:5])
