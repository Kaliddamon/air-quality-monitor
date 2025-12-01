from workshop6.simple_knn import knn_pollution
from workshop6.geographic_knn import knn_geographic
from workshop6.alert_knn import knn_alert


def generate_sample_sensors(n=20, seed=42):
    random.seed(seed)
    sensors = []
    geo_types = ["urban", "suburban", "rural"]
    pollution_sources = ["traffic", "industrial", "mixed"]
    air_states = ["good", "moderate", "unhealthy", "hazardous"]

    for i in range(n):
        # Coordenadas sintéticas en un plano (x, y)
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)

        # Vector de niveles de contaminación (últimos 7 días)
        pollution_levels = [round(random.uniform(10, 150), 2) for _ in range(7)]

        sensor = {
            "id": str(uuid.uuid4())[:8],
            "coords": (x, y),
            "geographicType": random.choice(geo_types),
            "pollutionSource": random.choice(pollution_sources),
            "airQualityState": random.choice(air_states),
            "alertIssued": random.choice([True, False]),
            "aqiValue": random.randint(0, 500),
            "airQualityData": {
                "PM25": round(random.uniform(0, 500), 2),
                "NO2": round(random.uniform(0, 200), 2),
                "O3": round(random.uniform(0, 300), 2),
                "CO": round(random.uniform(0, 50), 2)
            },
            "weatherInfluence": {
                "windSpeed": round(random.uniform(0, 30), 1),   # m/s
                "temperature": round(random.uniform(-10, 45), 1), # °C
                "humidity": random.randint(20, 100)               # %
            },
            "pollutionLevels7": pollution_levels  # historial 7 días
        }
        sensors.append(sensor)
    return sensors


def testKNN(sensors, qid):
    print("Sensor consulta id:", qid)
    print("GeographicType:", query["geographicType"], "PollSource:", query["pollutionSource"])
    print("AQI:", query["aqiValue"], "State:", query["airQualityState"], "AlertIssued:", query["alertIssued"])
    print()

    # Algorithm 1: Simple KNN (euclidiana sobre los 7 días)
    print("=== Algorithm 1: Simple KNN (Euclidean sobre 7 días) ===")
    neighbors1 = knn_pollution(sensors, query_id=qid, k=5, use_average=False)
    simple_knn = []
    for score, s in neighbors1:
        print(f"id:{s['id']} score:{score:.2f} avg:{mean_vector(s['pollutionLevels7']):.2f}")
        simple_knn.append({
            "Ciudad": s["sensorLocation"]
            "Score": score
        })

    print("\n=== Algorithm 2: Geographic KNN (mismo tipo geográfico preferido) ===")
    neighbors2 = knn_geographic(sensors, query_id=qid, k=5)
    geographic_knn = []
    for score, s in neighbors2:
        print(f"id:{s['id']} score:{score:.2f} type:{s['geographicType']} coords:{s['coords']} source:{s['pollutionSource']}")
        geographic_knn.append({
            "Ciudad": s["sensorLocation"]
            "Score": score
        })

    print("\n=== Algorithm 3: Alert KNN (event matching por estado/alert y clima) ===")
    neighbors3 = knn_alert(sensors, query_id=qid, k=5)
    alert_knn = []
    for score, s in neighbors3:
        print(f"id:{s['id']} score:{score:.2f} state:{s['airQualityState']} alert:{s['alertIssued']} weather:{s['weatherInfluence']} aqi:{s['aqiValue']}")
        alert_knn.append({
            "Ciudad": s["sensorLocation"]
            "Score": score
        })

    return pd.DataFrame(simple_knn), pd.DataFrame(geographic_knn), pd.DataFrame(alert_knn)

