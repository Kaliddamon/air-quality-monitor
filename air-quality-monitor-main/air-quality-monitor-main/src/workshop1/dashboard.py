import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd

from hash_table import HashTable
from prediction import run_all_and_alert
from w2_pipeline import run_w2_from_records
from w3_pipeline import run_w3_from_records, plot_markov_chain
from w4_pipeline import run_w4_from_records
import matplotlib.pyplot as plt   #plt for charts w4
import numpy as np   #needed for w4 heatmap



# ---------------------------------------------------------
# base project paths
# ---------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_dir))
DATA_PATH = os.path.join(project_root, "data", "sensor_data_clean.json")

st.set_page_config(
    page_title="Real-Time Air Quality Monitoring",
    layout="wide",
)

# ---------------------------------------------------------
# global styles
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-weight: 600;
        letter-spacing: 0.02em;
    }

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

    /* Scroll propio del sidebar para muchos filtros */
    section[data-testid="stSidebar"] > div {
        max-height: 100vh;
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# load data
# ---------------------------------------------------------
ht = HashTable()
if os.path.exists(DATA_PATH):
    ht.load_json(DATA_PATH)
else:
    st.error(f"No se encontró el archivo de datos en {DATA_PATH}")
    st.stop()

records = ht.all_records

# ---------------------------------------------------------
# dataframe converter
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# sidebar filters
# ---------------------------------------------------------
st.sidebar.title("Filtros")

all_cities = sorted([c for c in df["sensorLocation"].dropna().unique()])

# estado inicial del filtro de ciudades
if "city_filter" not in st.session_state:
    st.session_state["city_filter"] = []

# botones: seleccionar todo / limpiar
btn_col1, btn_col2 = st.sidebar.columns(2)
with btn_col1:
    if st.button("Todo"):
        st.session_state["city_filter"] = all_cities
with btn_col2:
    if st.button("Limpiar"):
        st.session_state["city_filter"] = []

# multiselect controlado por session_state
city_filter = st.sidebar.multiselect(
    "Ciudades",
    options=all_cities,
    key="city_filter"
)

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

# ---------------------------------------------------------
# title + stats
# ---------------------------------------------------------
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
    st.metric("Máximo", f"{pm.get('max'):.2f}" if pm.get("max") is not None else "N/D")

with col3:
    st.subheader("NO2 (global)")
    no2 = global_stats.get("NO2", {})
    st.metric("Promedio", f"{no2.get('avg'):.2f}" if no2.get("avg") is not None else "N/D")
    st.metric("Máximo", f"{no2.get('max'):.2f}" if no2.get("max") is not None else "N/D")

st.markdown("---")

# ---------------------------------------------------------
# ranking
# ---------------------------------------------------------
st.subheader("Ranking de ciudades por contaminante")

ranking = ht.rank_cities_by_pollutant(pollutant=pollutant)
rank_df = pd.DataFrame(ranking, columns=["Ciudad", f"Promedio {pollutant}"])

st.dataframe(rank_df, use_container_width=True, height=250)

st.markdown("---")

# ---------------------------------------------------------
# table of readings
# ---------------------------------------------------------
st.subheader("Lecturas de calidad del aire")

filtered_df = df.copy()

# si no hay ciudades seleccionadas, no se filtra por ciudad
if city_filter:
    filtered_df = filtered_df[filtered_df["sensorLocation"].isin(city_filter)]

if pollutant in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df[pollutant] >= low) & (filtered_df[pollutant] <= high)
    ]

if show_only_outliers:
    outliers = ht.detect_outliers()
    ids_outliers = set()
    for rec, info in outliers:
        if pollutant in info:
            ids_outliers.add(rec.get("_id"))
    filtered_df = filtered_df[filtered_df["_id"].isin(ids_outliers)]

if "timestamp" in filtered_df.columns:
    filtered_df = filtered_df.sort_values("timestamp", ascending=False)

st.dataframe(filtered_df, use_container_width=True, height=400)

st.markdown("---")

# ---------------------------------------------------------
# predictions W1
# ---------------------------------------------------------
st.subheader("Predicciones y alertas")

if st.button("Ejecutar predicciones y generar alertas"):
    alerts = run_all_and_alert(ht)
    if alerts:
        st.write("Alertas generadas:")
        st.dataframe(pd.DataFrame(alerts), use_container_width=True, height=250)
    else:
        st.write("No se generaron alertas con las condiciones actuales.")

st.markdown("---")

# ---------------------------------------------------------
# w2
# ---------------------------------------------------------
st.subheader("análisis probabilístico w2")

if st.button("ejecutar análisis avanzado (w2)"):

    # run the probabilistic analysis based on all records
    w2 = run_w2_from_records(records)

    # display the estimated number of distinct pollution events (flajolet-martin)
    st.write(f"estimación de eventos de contaminación distintos: {w2.get('distinct_events_estimate', 0)}")

    # ----------------------------------------------
    # moment 1: critical alert frequency per city
    # ----------------------------------------------
    moment1 = w2.get("moment1_by_city", {})

    # optional debug print to inspect structure
    # st.write("debug moment1_by_city:", moment1)

    if moment1:
        rows = []
        for city, info in moment1.items():

            # if the data is a dict, try to extract critical count
            if isinstance(info, dict):
                crit = (
                    info.get("count_critical")
                    or info.get("critical")
                    or info.get("frequency")
                    or info.get("moment1")
                    or 0
                )
            else:
                # if it's a number, use it directly
                crit = info

            rows.append({
                "Ciudad": city,
                "Alertas críticas": crit,
            })

        # display table for moment 1
        st.write("frecuencia de alertas críticas por ciudad (momento 1):")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=250)

    # ----------------------------------------------
    # top zones: zones with highest critical frequency
    # ----------------------------------------------
    top = w2.get("top_zones", [])
    if top:
        rows = []
        for item in top:
            # supports both tuples/lists and dict formats
            if isinstance(item, (tuple, list)):
                rows.append({"Zona": item[0], "Frecuencia crítica": item[1]})
            elif isinstance(item, dict):
                rows.append({
                    "Zona": item.get("zone"),
                    "Frecuencia crítica": item.get("frequency")
                })

        st.write("zonas con mayor frecuencia de eventos críticos:")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=250)

    # ----------------------------------------------
    # dgim estimations for the last 100 positions
    # ----------------------------------------------
    st.write("dgim - estimación últimas 100 posiciones:")
    st.write(f"aproximado: {w2.get('dgim_last_100')} | exacto: {w2.get('dgim_exact_last_100')}")

    # display trend from dgim
    st.write("tendencia dgim:")
    st.write(w2.get("dgim_trend"))

    # display next-window prediction
    st.write("predicción próxima ventana:")
    st.write(w2.get("dgim_prediction"))

    # ----------------------------------------------
    # adaptive sampling summary
    # ----------------------------------------------
    st.write("resumen de muestreo adaptativo:")
    st.json(w2.get("sampling_stats", {}))

# ---------------------------------------------------------
# W3
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Análisis predictivo W3 (Modelos de Markov)")

if st.button("Ejecutar análisis W3"):

    w3 = run_w3_from_records(records)

    st.write("Matriz de transición:")
    st.json(w3.get("transition_matrix"))

    st.write("Secuencia de estados PM2.5:")
    st.write(w3.get("states_sequence"))

    st.write("Predicción del próximo estado PM2.5:")
    st.write(w3.get("predicted_state"))

    st.write("Probabilidades espaciales por ciudad:")
    st.json(w3.get("spatial_probabilities"))

    st.write("Diagrama de la cadena de Markov:")
    try:
        fig = plot_markov_chain(
            w3.get("states"),
            w3.get("transition_matrix")
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"No se pudo generar el gráfico: {e}")

# ---------------------------------------------------------
# w4
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Análisis avanzado W4 (K-Means, anomalías y similitud entre ciudades)")

if st.button("Ejecutar análisis W4"):
    w4_data = run_w4_from_records(records)

    st.write("Drones por ciudad")

    counts = w4_data.get("counts")
    print(counts)
    st.dataframe(counts, use_container_width=True, height=250)

    st.write("Promedio de AQI por ciudad")

    averages = w4_data.get("averages")
    print(averages)
    st.dataframe(averages, use_container_width=True, height=250)

    col1, col2 = st.columns(2)

    max_city = w4_data.get("max_city")
    min_city = w4_data.get("min_city")

    with col1:
        st.subheader("Mayor")
        st.metric(max_city[0], f"{max_city[1]:.2f}" if max_city[1] is not None else "N/D")
    
    with col2:
        st.subheader("Menor")
        st.metric(min_city[0], f"{min_city[1]:.2f}" if min_city[1] is not None else "N/D")

    st.write("Gráfico de barras (Comparación Map Reduce Serial y Paralelo):")

    chart_data = w4_data.get("chart_data")
    st.bar_chart(chart_data)

    
    

    

    










