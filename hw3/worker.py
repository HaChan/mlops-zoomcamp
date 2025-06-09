import asyncio

from temporalio.client import Client
from temporalio.worker import Worker
from activities import (TASK_QUEUE_NAME, download_file, preprocess_dataframe,
    train_model, register_model)
from workflow import YellowTaxiTrainWorkflow

async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE_NAME,
        workflows=[YellowTaxiTrainWorkflow],
        activities=[download_file, preprocess_dataframe, train_model, register_model],
    )
    print("Worker start")
    await worker.run()

asyncio.run(main())
