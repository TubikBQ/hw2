from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, get_current_context, task


@dag(
    schedule="0 0 * * 1",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["olist"],
)
def olist_training():
    @task(retries=2, retry_delay=timedelta(seconds=30), execution_timeout=timedelta(minutes=5))
    def extract():
        from olist_pipeline import extract as extract_orders

        return extract_orders(get_current_context()["run_id"])

    @task(retries=0, execution_timeout=timedelta(minutes=2))
    def validate(snapshot):
        from olist_pipeline import validate as validate_snapshot

        return validate_snapshot(snapshot)

    @task(execution_timeout=timedelta(minutes=5))
    def prepare(snapshot):
        from olist_pipeline import prepare as prepare_splits

        return prepare_splits(snapshot)

    @task(execution_timeout=timedelta(minutes=20))
    def train(prepared):
        from olist_pipeline import train as train_model

        return train_model(prepared, dag_run_id=get_current_context()["run_id"])

    train(prepare(validate(extract())))


olist_training()
