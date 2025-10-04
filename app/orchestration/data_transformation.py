from prefect import flow, task
from app.scripts.flows.initial_data_trainsformation.collect_data import collect_data
from app.scripts.flows.initial_data_trainsformation.transform_data import transform_data


@task
def collect_data_task() -> None:
    collect_data()


@task
def transform_data_task() -> None:
    transform_data()


@flow
def chembl_pipeline() -> None:
    collect_data_task()
    transform_data_task()


if __name__ == "__main__":
    chembl_pipeline()
