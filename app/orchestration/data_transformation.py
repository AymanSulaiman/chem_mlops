from dagster import Config, Definitions, In, Nothing, Out, ScheduleDefinition, graph, op

from app.scripts.flows.eval.eval_model import eval_flow
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
from app.scripts.flows.llm_finetuning_data.download_twosides import download_twosides
from app.scripts.flows.vector_store.ingest_to_lancedb import (
    ingest_compounds_to_lancedb,
)
from app.scripts.flows.vector_store.ingest_twosides_to_lancedb import (
    ingest_twosides_to_lancedb,
)


class ChemblConfig(Config):
    chembl_version: str = "37"


@op(out=Out(Nothing))
def collect_chembl_op(config: ChemblConfig) -> None:
    collect_data(config.chembl_version)


@op(ins={"start": In(Nothing)}, out=Out(Nothing))
def transform_chembl_op(config: ChemblConfig) -> None:
    transform_data(config.chembl_version)


@op(ins={"start": In(Nothing)}, out=Out(Nothing))
def create_chembl_finetune_dataset_op() -> None:
    create_finetuning_dataset()


@op(out=Out(Nothing))
def download_twosides_op() -> None:
    download_twosides()


# Fan-in from ChEMBL transform + TWOSIDES download so QA generation has both sources ready.
@op(ins={"start_chembl": In(Nothing), "start_twosides": In(Nothing)}, out=Out(Nothing))
def build_drug_interaction_dataset_op() -> None:
    # workers=1: Dagster uses an in-process executor, so spawning a ProcessPoolExecutor
    # inside it leaks semaphores and can interfere with subsequent ops. Sequential
    # mode is safe here; run standalone for parallel speed.
    build_drug_interaction_dataset(workers=1)


# Both finetune-data ops must complete before finetuning begins (fan-in via Nothing inputs).
@op(ins={"start_a": In(Nothing), "start_b": In(Nothing)}, out=Out(Nothing))
def finetune_llm_op() -> None:
    gemma3_chembl_toon_finetune_flow()


@op(ins={"start": In(Nothing)}, out=Out(Nothing))
def ingest_chembl_to_lancedb_op() -> None:
    ingest_compounds_to_lancedb()


# Fan-in from compounds ingestion + TWOSIDES download; runs in parallel with finetuning.
@op(ins={"start_lancedb": In(Nothing), "start_twosides": In(Nothing)}, out=Out(Nothing))
def ingest_twosides_to_lancedb_op() -> None:
    ingest_twosides_to_lancedb()


@op(ins={"start": In(Nothing)}, out=Out(Nothing))
def eval_model_op() -> None:
    eval_flow(run_dir=latest_run_dir(ARTIFACTS_DIR))


@op(ins={"start": In(Nothing)})
def export_to_ollama_op() -> None:
    export_to_ollama(run_dir=latest_run_dir(ARTIFACTS_DIR), force=True)


@graph
def chembl_pipeline_graph() -> None:
    chembl_collected = collect_chembl_op()
    # TWOSIDES download is independent — starts immediately in parallel with ChEMBL collection.
    twosides = download_twosides_op()
    chembl_transformed = transform_chembl_op(start=chembl_collected)
    # Three parallel workloads after ChEMBL transform:
    chembl_finetune_data = create_chembl_finetune_dataset_op(start=chembl_transformed)
    drug_data = build_drug_interaction_dataset_op(start_chembl=chembl_transformed, start_twosides=twosides)
    chembl_lancedb_done = ingest_chembl_to_lancedb_op(start=chembl_transformed)
    # TWOSIDES LanceDB table: needs ChEMBL compounds DB to exist + TWOSIDES file ready.
    # Runs in parallel with finetuning.
    ingest_twosides_to_lancedb_op(start_lancedb=chembl_lancedb_done, start_twosides=twosides)
    finetuned = finetune_llm_op(start_a=chembl_finetune_data, start_b=drug_data)
    evaled = eval_model_op(start=finetuned)
    export_to_ollama_op(start=evaled)


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
