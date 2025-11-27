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

    w4 = run_w4_from_records(records)

    #cluster labels
    st.write("Asignación de clusters (por registro):")
    st.json(w4.get("cluster_labels"))

    #global centroids
    st.write("Centroides globales (normalizados):")
    st.json(w4.get("centroids"))

    #centroids by city
    st.write("Promedios PM2.5 y NO2 por ciudad:")
    st.json(w4.get("city_centroids"))

    #raw similarity matrix
    st.write("Matriz de similitud entre ciudades (distancia euclidiana):")
    st.json(w4.get("city_similarity_matrix"))

    #anomaly table
    st.write("Anomalías detectadas (top 5% más alejadas del cluster):")
    anomalies = w4.get("anomalies", [])
    if anomalies:
        st.dataframe(pd.DataFrame(anomalies), use_container_width=True, height=250)
    else:
        st.write("No se detectaron anomalías en este dataset.")

    #plots section
    st.markdown("### Visualizaciones W4")

    #scatter pm25 vs no2 colored by cluster
    try:
        df_plot = pd.DataFrame(w4.get("clustered_points", []))
        if not df_plot.empty:
            fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))

            #scatter plot by cluster
            for c_id, sub in df_plot.groupby("cluster"):
                ax_scatter.scatter(
                    sub["pm25"],
                    sub["no2"],
                    s=10,
                    alpha=0.6,
                    label=f"cluster {c_id}"
                )

            ax_scatter.set_xlabel("PM2.5")
            ax_scatter.set_ylabel("NO2")
            ax_scatter.set_title("distribución de lecturas por cluster")
            ax_scatter.legend(fontsize=8)
            st.pyplot(fig_scatter)
    except Exception as e:
        st.error(f"No se pudo generar el scatter de clusters: {e}")

    #elbow performance plot
    try:
        #get k values and inertia list
        k_values = w4.get("elbow_k_values", [])
        inertias = w4.get("elbow_inertias", [])

        if k_values and inertias:
            fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))

            #plot elbow curve
            ax_elbow.plot(k_values, inertias, marker="o", linestyle="-")
            ax_elbow.set_xlabel("k")
            ax_elbow.set_ylabel("inertia")
            ax_elbow.set_title("rendimiento k-means (elbow method)")
            ax_elbow.grid(True)

            st.pyplot(fig_elbow)
        else:
            st.write("No se pudo calcular el rendimiento (elbow method).")
    except Exception as e:
        st.error(f"No se pudo generar el gráfico de rendimiento: {e}")

    #heatmap of city similarity (only top cities for readability)
    try:
        sim_matrix = np.array(w4.get("city_similarity_matrix", []))
        city_centroids = w4.get("city_centroids", {}) or {}

        if sim_matrix.size > 0 and city_centroids:
            #get city names in a stable order
            if "city_names" in w4:
                city_names = w4.get("city_names", [])
            else:
                city_names = list(city_centroids.keys())

            #build dataframe with average pm25 to choose most critical cities
            rows = []
            for name, vals in city_centroids.items():
                rows.append({
                    "city": name,
                    "pm25": vals.get("PM25", 0.0),
                    "no2": vals.get("NO2", 0.0)
                })
            cent_df = pd.DataFrame(rows)

            #select top n cities by pm25
            top_n = 25
            top_cities = cent_df.sort_values("pm25", ascending=False)["city"].head(top_n).tolist()

            #indices of those cities in the similarity matrix
            indices = [city_names.index(c) for c in top_cities if c in city_names]

            if indices:
                sim_sub = sim_matrix[np.ix_(indices, indices)]
                labels = [city_names[i] for i in indices]

                fig_heat, ax_heat = plt.subplots(figsize=(7, 6))
                im = ax_heat.imshow(sim_sub, cmap="Reds")
                ax_heat.set_xticks(range(len(labels)))
                ax_heat.set_xticklabels(labels, rotation=90, fontsize=6)
                ax_heat.set_yticks(range(len(labels)))
                ax_heat.set_yticklabels(labels, fontsize=6)
                ax_heat.set_title("city similarity matrix (euclidean distance) - top cities")
                fig_heat.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
                st.pyplot(fig_heat)
                st.caption("solo se muestran las ciudades más críticas para hacer legible el mapa.")
    except Exception as e:
        st.error(f"No se pudo generar el heatmap de similitud: {e}")

