import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
from dataclasses import dataclass

from temporalio import activity

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
TASK_QUEUE_NAME = "yellow-taxi-training-pipeline-queue"

@activity.defn
async def download_file():
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    df = pd.read_parquet(url)
    print(f"Number of rows downloaded {df.shape[0]}")
    file_name = 'yellow_tripdata.parquet'
    df.to_parquet(
        file_name,
        engine='pyarrow',
        compression='gzip',
        index=False
    )
    return file_name

@activity.defn
async def preprocess_dataframe(file_name):
    df = pd.read_parquet(file_name)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Number of rows after prep {df.shape[0]}")
    file_name = 'yellow_tripdata_prep.parquet'
    df.to_parquet(
        file_name,
        engine='pyarrow',
        compression='gzip',
        index=False
    )
    return file_name

@activity.defn
async def train_model(file_name):
    df = pd.read_parquet(file_name)
    x_dict = df[['PULocationID', 'DOLocationID']].astype(str).to_dict(orient='records')
    y = df['duration']
    X_train, X_val, y_train, y_val = train_test_split(x_dict, y, test_size=0.2, random_state=42)

    dv = DictVectorizer()
    X_encode = dv.fit_transform(X_train)
    X_val_encode = dv.transform(X_val)
    model = LinearRegression()
    model.fit(X_encode, y_train)
    print(model.intercept_)
    y_pred = model.predict(X_val_encode)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("rmse: ", rmse)

    with open("dv.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)

    with open("model.pkl", "wb") as f_out:
        pickle.dump(model, f_out)

@activity.defn
async def register_model():
    model = load_pickle("model.pkl")
    dv = load_pickle("dv.pkl")

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.sklearn.log_model(model, artifact_path="model")

        mlflow.log_artifact("dv.pkl")
        return run.info.run_id


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
