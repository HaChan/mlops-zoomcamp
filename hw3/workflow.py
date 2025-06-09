from datetime import timedelta
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities import (download_file, preprocess_dataframe,
        train_model, register_model)


@workflow.defn
class YellowTaxiTrainWorkflow:
    @workflow.run
    async def run(self):
        file_name = await workflow.execute_activity(
            download_file,
            start_to_close_timeout=timedelta(minutes=1)
        )
        file_name = await workflow.execute_activity(
            preprocess_dataframe,
            file_name,
            start_to_close_timeout=timedelta(minutes=1)
        )
        await workflow.execute_activity(
            train_model,
            file_name,
            start_to_close_timeout=timedelta(minutes=10)
        )
        run_id = await workflow.execute_activity(
            register_model,
            start_to_close_timeout=timedelta(minutes=3)
        )
        return run_id
