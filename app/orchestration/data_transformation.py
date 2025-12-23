from prefect import flow, task
from app.scripts.flows.initial_data_transformation.collect_data import collect_data
from app.scripts.flows.initial_data_transformation.transform_data import transform_data
from app.scripts.flows.llm_finetuning_data.build_finetune_dataset import (
    create_finetuning_dataset,
)
from app.scripts.flows.finetuning.finetuning import gemma3_chembl_toon_finetune_flow


@task
def collect_data_task() -> None:
    collect_data()


@task
def transform_data_task() -> None:
    transform_data()


@task
def create_finetune_dataset_task() -> None:
    create_finetuning_dataset()


@task
def finetune_llm_task() -> None:
    gemma3_chembl_toon_finetune_flow()


@flow
def chembl_pipeline() -> None:
    collect_data_task()
    transform_data_task()
    create_finetune_dataset_task()
    gemma3_chembl_toon_finetune_flow()


if __name__ == "__main__":
    chembl_pipeline()
