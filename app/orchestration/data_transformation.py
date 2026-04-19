from dagster import Config, Definitions, In, Nothing, Out, ScheduleDefinition, graph, op

from app.scripts.flows.finetuning.export_to_ollama import (
    ARTIFACTS_DIR,
    export_to_ollama,
    latest_run_dir,
)
from app.scripts.flows.finetuning.finetuning import gemma3_chembl_toon_finetune_flow
from app.scripts.flows.initial_data_transformation.collect_data import collect_data
from app.scripts.flows.initial_data_transformation.transform_data import transform_data
from app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset import (
    build_drug_interaction_dataset,
)
from app.scripts.flows.llm_finetuning_data.build_finetune_dataset import (
    create_finetuning_dataset,
)


class ChemblConfig(Config):
    chembl_version: str = "36"


@op(out=Out(Nothing))
def collect_data_op(config: ChemblConfig) -> None:
    collect_data(config.chembl_version)


@op(ins={"start": In(Nothing)}, out=Out(Nothing))
def transform_data_op(config: ChemblConfig) -> None:
    transform_data(config.chembl_version)


@op(ins={"start": In(Nothing)}, out=Out(Nothing))
def create_finetune_dataset_op() -> None:
    create_finetuning_dataset()


@op(ins={"start": In(Nothing)}, out=Out(Nothing))
def build_drug_interaction_dataset_op() -> None:
    build_drug_interaction_dataset()


# Both f3a/f3b must complete before finetuning begins (fan-in via Nothing inputs)
@op(ins={"start_a": In(Nothing), "start_b": In(Nothing)}, out=Out(Nothing))
def finetune_llm_op() -> None:
    gemma3_chembl_toon_finetune_flow()


@op(ins={"start": In(Nothing)})
def export_to_ollama_op() -> None:
    export_to_ollama(run_dir=latest_run_dir(ARTIFACTS_DIR), force=True)


@graph
def chembl_pipeline_graph() -> None:
    collected = collect_data_op()
    transformed = transform_data_op(start=collected)
    # Build raw activity parquet and QA JSONL in parallel, then fan-in
    finetune_data = create_finetune_dataset_op(start=transformed)
    drug_data = build_drug_interaction_dataset_op(start=transformed)
    finetuned = finetune_llm_op(start_a=finetune_data, start_b=drug_data)
    export_to_ollama_op(start=finetuned)


chembl_pipeline = chembl_pipeline_graph.to_job(name="chembl_pipeline")

daily_schedule = ScheduleDefinition(
    job=chembl_pipeline,
    cron_schedule="0 0 * * *",
    execution_timezone="UTC",
)

defs = Definitions(
    jobs=[chembl_pipeline],
    schedules=[daily_schedule],
)

if __name__ == "__main__":
    chembl_pipeline.execute_in_process()
