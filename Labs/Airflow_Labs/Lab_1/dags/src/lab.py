import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import pickle
import os
import base64


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("Loading data from file.csv")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a DBSCAN clustering model on the preprocessed data and saves it.
    Returns a dict with model info (JSON-serializable).
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    # DBSCAN: eps controls neighborhood radius, min_samples controls density threshold
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan.fit(df)

    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")

    #save fitted model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(dbscan, f)

    #moel info
    return {"n_clusters": n_clusters, "n_noise": n_noise}


def load_model_elbow(filename: str, model_info: dict):
    """
    Loads the saved DBSCAN model and reports clustering results.
    Returns the first prediction label (as a plain int) for test.csv.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    with open(output_path, "rb") as f:
        loaded_model = pickle.load(f)

    # results 
    print(f"DBSCAN Results: {model_info['n_clusters']} clusters, {model_info['n_noise']} noise points")
    print(f"Core sample count: {len(loaded_model.core_sample_indices_)}")

    #for DBSCAN, we use the training labels since DBSCAN doesn't have a .predict() method.
    #  report  label assigned to the first core sample as a demo.
    labels = loaded_model.labels_
    first_label = int(labels[0])
    print(f"First sample cluster label: {first_label}")

    return first_label
