from pathlib import Path

from dagster import Config, Definitions, In, Nothing, Out, ScheduleDefinition, graph, op

from app.scripts.flows.eval.eval_finetuned_model import eval_flow
from app.scripts.flows.finetuning.continue_tool_training import continue_tool_training
from app.scripts.flows.finetuning.export_to_ollama import export_to_ollama
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
@op(ins={"start_a": In(Nothing), "start_b": In(Nothing)}, out=Out(str))
def finetune_llm_op() -> str:
    # export=False: the model that gets published is the tool-trained one, and
    # only after the gate. Exporting here would put an unevaluated model in
    # Ollama and leave it there if the gate later failed.
    return str(gemma3_chembl_toon_finetune_flow(export=False))


@op(ins={"start": In(Nothing)}, out=Out(Nothing))
def ingest_chembl_to_lancedb_op() -> None:
    ingest_compounds_to_lancedb()


# Fan-in from compounds ingestion + TWOSIDES download; runs in parallel with finetuning.
@op(ins={"start_lancedb": In(Nothing), "start_twosides": In(Nothing)}, out=Out(Nothing))
def ingest_twosides_to_lancedb_op() -> None:
    ingest_twosides_to_lancedb()


# Measured, not shipped. Two reasons to spend ~15 minutes here:
#   - Attribution. Without it, a bad number after continued training has two
#     suspects — the base run or the continuation — and no way to tell them
#     apart once artifacts/ has been cleaned.
#   - It answers whether continued training earns its place, on every run,
#     instead of as a special investigation.
# Both thresholds are 0, which disables those gates: this adapter has tool
# records at ~6% of its mix and is not expected to clear a tool-call or a
# lookup-dependent bar, and blocking the pipeline on a model nobody ships is
# the mistake the golden gate used to make in the first place. The perplexity
# gate still applies — continuing from a regressed adapter is pointless, and
# better to find out before spending the continuation.
@op(out=Out(str))
def eval_base_model_op(run_dir: str) -> str:
    eval_flow(run_dir=Path(run_dir), tool_call_threshold=0.0, pass_threshold=0.0)
    return run_dir


# Continues the run above on a tool-heavy mix, into its own <stamp>_tools run
# directory. The full run's 60 K tool-call records compete with ~900 K prose ones
# answering the same question shapes from memory; this trains on a mix where tool
# calls are a third of what the model sees, in minutes rather than hours. The
# model that reaches the export is therefore the tool-trained one.
@op(out=Out(str))
def continue_tool_training_op(run_dir: str) -> str:
    return str(continue_tool_training(from_run=Path(run_dir)))


# Both of these take the run directory explicitly rather than calling
# latest_run_dir(). They used to guess, which meant anything else writing to
# artifacts/ could silently send the gate and the export at different models —
# or at a model this pipeline never trained.
@op(out=Out(str))
def eval_finetuned_model_op(run_dir: str) -> str:
    eval_flow(run_dir=Path(run_dir))
    return run_dir


@op
def export_to_ollama_op(run_dir: str) -> None:
    export_to_ollama(run_dir=Path(run_dir), force=True)


@graph
def chembl_pipeline_graph() -> None:
    raw_chembl = collect_chembl_op()
    raw_twosides = download_twosides_op()
    chembl_parquet = transform_chembl_op(start=raw_chembl)
    chembl_finetune_dataset = create_chembl_finetune_dataset_op(start=chembl_parquet)
    drug_interaction_dataset = build_drug_interaction_dataset_op(
        start_chembl=chembl_parquet, start_twosides=raw_twosides
    )
    compounds_vector_store = ingest_chembl_to_lancedb_op(start=chembl_parquet)
    # Terminal op: the vector store serves the web app's agent tools at query
    # time, so nothing downstream in this pipeline depends on it.
    ingest_twosides_to_lancedb_op(start_lancedb=compounds_vector_store, start_twosides=raw_twosides)
    finetuned_model = finetune_llm_op(
        start_a=chembl_finetune_dataset, start_b=drug_interaction_dataset
    )
    base_model_measured = eval_base_model_op(finetuned_model)
    tool_trained_model = continue_tool_training_op(base_model_measured)
    export_to_ollama_op(eval_finetuned_model_op(tool_trained_model))


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
