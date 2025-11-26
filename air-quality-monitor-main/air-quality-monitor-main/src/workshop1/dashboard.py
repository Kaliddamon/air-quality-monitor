import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd

from hash_table import HashTable
from prediction import run_all_and_alert
from w2_pipeline import run_w2_from_records

st.set_page_config(
    page_title="Real-Time Air Quality Monitoring",
    layout="wide",
)

#global styles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Encabezados más sobrios */
    h1, h2, h3, h4 {
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Tabla: encabezado "fijo" con fondo sólido */
    .stDataFrame tbody tr th {
        position: sticky;
        left: 0;
        z-index: 1;
    }

    .stDataFrame thead tr th {
        position: sticky;
        top: 0;
        z-index: 2;
        background-color: #f4f4f4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#load data
DATA_PATH = "./data/sensor_data_clean.json"

ht = HashTable()
if os.path.exists(DATA_PATH):
    ht.load_json(DATA_PATH)
else:
    st.error(f"No se encontró el archivo de datos en {DATA_PATH}")
    st.stop()

#converter
records = ht.all_records

def records_to_dataframe(records_list):
    rows = []
    for r in records_list:
        aq = r.get("airQualityData", {}) or {}
        rows.append({
            "_id": r.get("_id"),
            "sensorLocation": r.get("sensorLocation"),
            "PM25": aq.get("PM25"),
            "NO2": aq.get("NO2"),
            "timestamp": r.get("timestamp"),
        })
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

df = records_to_dataframe(records)

#sidebar
st.sidebar.title("Filtros")

#city filter
all_cities = sorted([c for c in df["sensorLocation"].dropna().unique()])
city_filter = st.sidebar.multiselect("Ciudades", options=all_cities, default=all_cities)

#filter contaminant
pollutant = st.sidebar.selectbox("Contaminante principal", ["PM25", "NO2"])

if not df.empty and pollutant in df.columns:
    min_val = float(df[pollutant].min()) if pd.notna(df[pollutant].min()) else 0.0
    max_val = float(df[pollutant].max()) if pd.notna(df[pollutant].max()) else 0.0
    low, high = st.sidebar.slider(
        f"Rango de {pollutant}",
        min_value=float(min_val),
        max_value=float(max_val if max_val > min_val else min_val + 1),
        value=(float(min_val), float(max_val if max_val > min_val else min_val + 1)),
        step=0.5
    )
else:
    low, high = 0.0, 0.0

show_only_outliers = st.sidebar.checkbox("Mostrar solo registros fuera de umbral", value=False)

st.title("Real-Time Air Quality Monitoring Dashboard")

stats = ht.stats()
global_stats = ht.global_stats()

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Resumen de registros")
    st.metric("Total de lecturas", stats["total_records"])
    st.metric("Sensores únicos", stats["unique_sensors"])

with col2:
    st.subheader("PM2.5 (global)")
    pm = global_stats.get("PM25", {})
    st.metric("Promedio", f"{pm.get('avg'):.2f}" if pm.get("avg") is not None else "N/D")
    st.metric("Máximo", f"{pm.get("max"):.2f}" if pm.get("max") is not None else "N/D")

with col3:
    st.subheader("NO2 (global)")
    no2 = global_stats.get("NO2", {})
    st.metric("Promedio", f"{no2.get("avg"):.2f}" if no2.get("avg") is not None else "N/D")
    st.metric("Máximo", f"{no2.get("max"):.2f}" if no2.get("max") is not None else "N/D")

st.markdown("---")

#city rank
st.subheader("Ranking de ciudades por contaminante")

ranking = ht.rank_cities_by_pollutant(pollutant=pollutant)
rank_df = pd.DataFrame(ranking, columns=["Ciudad", f"Promedio {pollutant}"])

st.dataframe(
    rank_df,
    use_container_width=True,
    height=250
)

st.markdown("---")

#table
st.subheader("Lecturas de calidad del aire")

filtered_df = df.copy()

#city filter
if city_filter:
    filtered_df = filtered_df[filtered_df["sensorLocation"].isin(city_filter)]

#filter for rank
if pollutant in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df[pollutant] >= low) &
        (filtered_df[pollutant] <= high)
    ]

#filter
if show_only_outliers:
    outliers = ht.detect_outliers()
    ids_outliers = set()
    for rec, info in outliers:
        aq = rec.get("airQualityData", {}) or {}
        if pollutant in info:
            ids_outliers.add(rec.get("_id"))
    filtered_df = filtered_df[filtered_df["_id"].isin(ids_outliers)]

#order for time
if "timestamp" in filtered_df.columns:
    filtered_df = filtered_df.sort_values("timestamp", ascending=False)

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=400
)

st.markdown("---")

#predict
st.subheader("Predicciones y alertas")

if st.button("Ejecutar predicciones y generar alertas"):
    alerts = run_all_and_alert(ht)
    if alerts:
        st.write("Alertas generadas:")
        alerts_df = pd.DataFrame(alerts)
        st.dataframe(
            alerts_df,
            use_container_width=True,
            height=250
        )
    else:
        st.write("No se generaron alertas con las condiciones actuales.")

st.markdown("---")

#w2analysis
st.subheader("Análisis probabilístico W2")

if st.button("Ejecutar análisis avanzado (W2)"):
    w2_result = run_w2_from_records(records)

    #distinct events estimate
    distinct_events = w2_result.get("distinct_events_estimate", 0)
    st.write(f"Estimación de eventos de contaminación distintos: {distinct_events}")

    #moment 1 by city (critical alerts)
    moment1_by_city = w2_result.get("moment1_by_city", {})
    if moment1_by_city:
        moment1_rows = []
        for city, info in moment1_by_city.items():
            moment1_rows.append({
                "Ciudad": city,
                "Alertas críticas": info.get("count_critical", 0),
                "Total lecturas": info.get("total_readings", 0),
                "Proporción crítica": info.get("critical_ratio", 0.0),
            })
        moment1_df = pd.DataFrame(moment1_rows)
        st.write("Frecuencia de alertas críticas por ciudad (Momento 1):")
        st.dataframe(
            moment1_df,
            use_container_width=True,
            height=250
        )

    #top zones
    top_zones = w2_result.get("top_zones", [])
    if top_zones:
        top_rows = []
        for item in top_zones:
            zone = item.get("zone")
            freq = item.get("frequency", 0)
            top_rows.append({
                "Zona": zone,
                "Frecuencia crítica": freq,
            })
        top_df = pd.DataFrame(top_rows)
        st.write("Zonas con mayor frecuencia de eventos críticos:")
        st.dataframe(
            top_df,
            use_container_width=True,
            height=250
        )

    #dgim metrics
    dgim_last_100 = w2_result.get("dgim_last_100")
    dgim_exact_last_100 = w2_result.get("dgim_exact_last_100")
    dgim_trend = w2_result.get("dgim_trend")
    dgim_prediction = w2_result.get("dgim_prediction")

    st.write("DGIM - estimación de lecturas críticas en las últimas 100 posiciones:")
    st.write(f"aproximado: {dgim_last_100} | exacto: {dgim_exact_last_100}")
    st.write("tendencia detectada (dgim):")
    st.write(dgim_trend)
    st.write("predicción próxima ventana (dgim):")
    st.write(dgim_prediction)

    #adaptive sampling stats
    sampling_stats = w2_result.get("sampling_stats", {})
    if sampling_stats:
        st.write("Resumen de muestreo adaptativo:")
        st.json(sampling_stats)
