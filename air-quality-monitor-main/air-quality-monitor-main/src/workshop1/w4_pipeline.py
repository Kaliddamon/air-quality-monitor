import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

#convertrecordsintoadf
def records_to_df(records):
    rows = []
    for r in records:
        aq = r.get("airQualityData", {}) or {}
        rows.append({
            "sensorLocation": r.get("sensorLocation"),
            "PM25": aq.get("PM25"),
            "NO2": aq.get("NO2"),
        })
    df = pd.DataFrame(rows)
    df = df.dropna()
    return df

#normalizecolumnsbetween0and1
def normalize(df, cols):
    return (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min() + 1e-9)

#mainpipelineforw4
def run_w4_from_records(records):

    #buildcleanframe
    df = records_to_df(records)

    #nodatafilter
    if df.empty:
        return {
            "cluster_labels": [],
            "centroids": None,
            "city_centroids": {},
            "city_similarity_matrix": None,
            "anomalies": [],
        }

    #applynormalization
    norm_df = normalize(df, ["PM25", "NO2"])

    #runclustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(norm_df[["PM25", "NO2"]])

    #extractcentroids
    centroids = kmeans.cluster_centers_.tolist()

    #computeaveragesbycity
    city_centroids = (
        df.groupby("sensorLocation")[["PM25", "NO2"]].mean().to_dict(orient="index")
    )

    #buildsimilaritymatrix
    city_df = pd.DataFrame(city_centroids).T.fillna(0)
    if len(city_df) > 1:
        similarity = pairwise_distances(city_df, metric="euclidean")
        similarity_matrix = similarity.tolist()
    else:
        similarity_matrix = None

    #detectanomaliesbasedondistance
    distances = kmeans.transform(norm_df[["PM25", "NO2"]])
    cluster_dist = np.min(distances, axis=1)
    threshold = np.percentile(cluster_dist, 95)

    #exportanomalouspoints
    anomalies = df[cluster_dist >= threshold]
    anomalies_export = anomalies[["sensorLocation", "PM25", "NO2"]].to_dict("records")

    #finaloutput
    output = {
        "cluster_labels": df["cluster"].astype(int).tolist(),
        "centroids": centroids,
        "city_centroids": city_centroids,
        "city_similarity_matrix": similarity_matrix,
        "anomalies": anomalies_export,
    }

    return output
