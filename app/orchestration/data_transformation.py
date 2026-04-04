from prefect import flow, task
from app.scripts.flows.initial_data_transformation.collect_data import collect_data
from app.scripts.flows.initial_data_transformation.transform_data import transform_data
from app.scripts.flows.llm_finetuning_data.build_finetune_dataset import (
    create_finetuning_dataset,
)
from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
    build_drug_interaction_dataset,
)
from app.scripts.flows.finetuning.finetuning import gemma3_chembl_toon_finetune_flow


@task
def collect_data_task(chembl_version: str = "36") -> None:
    collect_data(chembl_version)


@task
def transform_data_task(chembl_version: str = "36") -> None:
    transform_data(chembl_version)


@task
def create_finetune_dataset_task() -> None:
    create_finetuning_dataset()


@task
def build_drug_interaction_dataset_task() -> None:
    build_drug_interaction_dataset()


@task
def finetune_llm_task() -> None:
    gemma3_chembl_toon_finetune_flow()


@flow
def chembl_pipeline(chembl_version: str = "36") -> None:
    f1 = collect_data_task.submit(chembl_version)
    f2 = transform_data_task.submit(chembl_version, wait_for=[f1])
    # Build both the raw activity parquet and the QA JSONL in parallel
    f3a = create_finetune_dataset_task.submit(wait_for=[f2])
    f3b = build_drug_interaction_dataset_task.submit(wait_for=[f2])
    finetune_llm_task.submit(wait_for=[f3a, f3b])


if __name__ == "__main__":
    chembl_pipeline()
