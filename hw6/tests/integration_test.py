import os
import pandas as pd
import subprocess
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def generate_file():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    input_file = 's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
    options = {
        'client_kwargs': {
            'endpoint_url': s3_endpoint_url
        }
    }
    df.to_parquet(
        input_file.format(year=2023, month=1),
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def run_batch():
    result = subprocess.run([
        "python", "batch.py", "2023", "1"
    ], env={
        **os.environ,
        "INPUT_FILE_PATTERN": 's3://nyc-duration/in/{year:04d}-{month:02d}.parquet',
        "OUTPUT_FILE_PATTERN": 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet',
        "S3_ENDPOINT_URL": 'http://localhost:4566/'
    }, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

run_batch()
