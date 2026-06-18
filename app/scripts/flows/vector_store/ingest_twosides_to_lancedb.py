"""
Ingest the TWOSIDES polypharmacy side-effect dataset into LanceDB.

Creates a `polypharmacy` table (separate from the `compounds` vector table) keyed
by drug name pairs. At RAG inference time, this table can be queried by drug name
to retrieve known polypharmacy side-effect signals for prompt augmentation.

The table has no vector column — it is a scalar-indexed lookup store.

Run download_twosides.py first:
    uv run python -m app.scripts.flows.llm_finetuning_data.download_twosides
"""

from pathlib import Path

import lancedb
import polars as pl
from tqdm import tqdm

TWOSIDES_PATH = Path("data/twosides/TWOSIDES.parquet")
LANCEDB_DIR = Path("data/lancedb")
POLYPHARMACY_TABLE = "polypharmacy"

MIN_PRR = 3.0
MIN_CASES = 5
BATCH_SIZE = 50_000


def _load_twosides(path: Path) -> pl.DataFrame:
    """Load, filter, and aggregate TWOSIDES into one row per drug pair."""
    print(f"Loading TWOSIDES from {path} ...")

    df = (
        pl.scan_parquet(path)
        .with_columns([
            pl.col("PRR").cast(pl.Float32, strict=False),
            pl.col("A").cast(pl.Int32, strict=False),
            pl.col("mean_reporting_frequency").cast(pl.Float32, strict=False),
        ])
        .filter(
            (pl.col("PRR") >= MIN_PRR)
            & (pl.col("A") >= MIN_CASES)
        )
        .select([
            pl.col("drug_1_rxnorn_id").cast(pl.Int64, strict=False).alias("drug_1_rxnorm_id"),
            pl.col("drug_1_concept_name").str.to_titlecase().alias("drug_1_name"),
            pl.col("drug_2_rxnorm_id").cast(pl.Int64, strict=False),
            pl.col("drug_2_concept_name").str.to_titlecase().alias("drug_2_name"),
            pl.col("condition_concept_name").alias("side_effect"),
            pl.col("PRR").alias("prr"),
            pl.col("A").alias("cases"),
            pl.col("mean_reporting_frequency"),
        ])
        .collect()
    )
    print(f"  Rows after filtering (PRR >= {MIN_PRR}, cases >= {MIN_CASES}): {len(df):,}")

    aggregated = (
        df.sort("prr", descending=True)
        .group_by(["drug_1_rxnorm_id", "drug_1_name", "drug_2_rxnorm_id", "drug_2_name"])
        .agg([
            pl.col("side_effect").str.join("; ").alias("side_effects"),
            pl.col("prr").max().alias("max_prr"),
            pl.col("prr").mean().round(2).alias("mean_prr"),
            pl.col("cases").sum().alias("total_cases"),
            pl.len().alias("n_side_effects"),
            pl.col("mean_reporting_frequency").mean().round(4).alias("mean_reporting_freq"),
        ])
        # Canonical ordering: drug_1 < drug_2 alphabetically eliminates reverse duplicates
        .with_columns([
            pl.concat_str(
                [pl.col("drug_1_name"), pl.lit("|"), pl.col("drug_2_name")]
            ).alias("pair_key"),
        ])
        .sort("max_prr", descending=True)
    )

    print(f"  Unique drug pairs after aggregation: {len(aggregated):,}")
    return aggregated


def _resolve_chembl_version(lancedb_dir: Path) -> str:
    """Find the ChEMBL-versioned LanceDB database directory."""
    dirs = sorted(lancedb_dir.glob("chembl_*"), reverse=True)
    if not dirs:
        raise FileNotFoundError(
            f"No ChEMBL LanceDB database found in {lancedb_dir}. "
            "Run ingest_to_lancedb.py first."
        )
    return dirs[0].name


def ingest_twosides_to_lancedb(
    twosides_path: Path = TWOSIDES_PATH,
    lancedb_dir: Path = LANCEDB_DIR,
) -> None:
    """Build the polypharmacy LanceDB table from TWOSIDES data.

    Connects to the existing ChEMBL LanceDB database and creates (or overwrites)
    a `polypharmacy` table with scalar indexes on drug names for fast lookup.
    """
    if not twosides_path.exists():
        raise FileNotFoundError(
            f"TWOSIDES not found at {twosides_path}. "
            "Run: uv run python -m app.scripts.flows.llm_finetuning_data.download_twosides"
        )

    df = _load_twosides(twosides_path)

    db_name = _resolve_chembl_version(lancedb_dir)
    db = lancedb.connect(str(lancedb_dir / db_name))
    print(f"Connected to LanceDB: {lancedb_dir / db_name}")

    if POLYPHARMACY_TABLE in db.list_tables().tables:
        print(f"'{POLYPHARMACY_TABLE}' table exists — overwriting")

    total = len(df)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    table = None

    for i in tqdm(range(n_batches), desc="Writing polypharmacy batches"):
        batch = df.slice(i * BATCH_SIZE, BATCH_SIZE).to_dicts()
        if table is None:
            table = db.create_table(POLYPHARMACY_TABLE, data=batch, mode="overwrite")
        else:
            table.add(batch)

    if table is not None:
        table.create_scalar_index("drug_1_name")
        table.create_scalar_index("drug_2_name")
        table.create_scalar_index("pair_key")
        print(f"Indexed on drug_1_name, drug_2_name, pair_key")

    print(f"Done. {total:,} drug pairs written to '{POLYPHARMACY_TABLE}' table.")


if __name__ == "__main__":
    ingest_twosides_to_lancedb()
