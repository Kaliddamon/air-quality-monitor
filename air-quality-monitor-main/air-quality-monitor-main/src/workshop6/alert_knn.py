from workshop6.utils import *

def knn_alert(sensors, query=None, k=5, w_alert=100.0, w_weather=1.0, w_aqi=0.5):
    """
    score compuesto:
      - Si mismo airQualityState o mismo alertIssued -> menor penalización
      - Distancia en condiciones meteorológicas (normalizado por rangos típicos)
      - Diferencia de aqiValue (normalizada)
    """
    if query is None:
        raise ValueError("Se necesita query para knn_alert")

    q_state = query["airQualityState"]
    q_alert = query["alertIssued"]
    q_weather = query["weatherInfluence"]
    q_aqi = query["aqiValue"]

    # rangos para normalizar
    temp_range = 55.0   # -10 a 45
    humidity_range = 80.0  # 20 a 100
    wind_range = 30.0   # 0 a 30
    aqi_range = 500.0

    results = []
    for s in sensors:
        if s["_id"] == query_id:
            continue
        # Alert/State penalty: prefer mismos eventos
        state_penalty = 0.0 if s["airQualityState"] == q_state else 1.0
        alert_penalty = 0.0 if s["alertIssued"] == q_alert else 1.0
        alert_score = (state_penalty + alert_penalty) * w_alert

        # Weather distance (normalizado)
        sw = s["weatherInfluence"]
        d_temp = abs(q_weather["temperature"] - sw["temperature"]) / temp_range
        d_hum = abs(q_weather["humidity"] - sw["humidity"]) / humidity_range
        d_wind = abs(q_weather["windSpeed"] - sw["windSpeed"]) / wind_range
        weather_dist = (d_temp + d_hum + d_wind)  # suma de fracciones

        # AQI diferencia normalizada
        aqi_diff = abs(q_aqi - s["aqiValue"]) / aqi_range

        score = alert_score + w_weather * weather_dist + w_aqi * aqi_diff
        results.append((score, s))

    results.sort(key=lambda x: x[0])
    return results[:k]



