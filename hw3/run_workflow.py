import asyncio
from temporalio.client import Client
from activities import TASK_QUEUE_NAME
from workflow import YellowTaxiTrainWorkflow

async def main():
    client = await Client.connect("localhost:7233")
    result = await client.execute_workflow(
        YellowTaxiTrainWorkflow.run,
        id="nyc-yellow-taxi-train-workflow",
        task_queue=TASK_QUEUE_NAME
    )
    print(f"Workflow complete. Run ID: {result}")

asyncio.run(main())
