# ChEMBL → LanceDB RAG Vector Store
# Ingests all ChEMBL compound data into a single flat LanceDB table.
# Each row = one compound with a Morgan fingerprint `vector` column for
# similarity search, plus all metadata columns for scalar filtering.

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import lancedb
import math
import numpy as np
import polars as pl
from lancedb.db import DBConnection
from lancedb.table import Table
from lancedb._lancedb import AddResult
from polars.dataframe.frame import DataFrame
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdchem import Mol
from rdkit.rdBase import BlockLogs
from tqdm import tqdm


DATA_DIR: str = "data/chembl_transform"
LANCEDB_DIR: str = "data/lancedb"
COMPOUNDS_TABLE: str = "compounds"
MORGAN_RADIUS: int = 2
MORGAN_BITS: int = 2048
BATCH_SIZE: int = 10_000

# Generator created once per process — reused across all _smiles_to_fp calls.
# ~10× faster than the deprecated GetMorganFingerprintAsBitVect and returns a
# numpy array directly, skipping an extra conversion step.
_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)


# ── Private helpers ───────────────────────────────────────────────────────────


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Collect a LazyFrame, asserting the result is a DataFrame for type checkers."""
    result = lf.collect()
    assert isinstance(result, pl.DataFrame)
    return result


def _smiles_to_fp(smiles: str | None) -> np.ndarray | None:
    """Convert a SMILES string to a 2048-bit Morgan fingerprint, or None if invalid."""
    if smiles is None:
        return None
    with BlockLogs():
        mol: Mol | None = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)


def _resolve_chembl_version(parquet_dir: str) -> str:
    """Read the latest ChEMBL release identifier from chembl_release.parquet."""
    return _collect(
        pl.scan_parquet(f"{parquet_dir}/chembl_release.parquet")
        .sort("chembl_release_id", descending=True)
        .select("chembl_release")
        .limit(1)
    ).item()


def _build_flat_df(parquet_dir: str) -> pl.DataFrame:
    """Join all relevant ChEMBL tables into one flat compound DataFrame.

    Anchor: molecule_dictionary filtered to structure_type == 'MOL'.
    Rows with no canonical SMILES are dropped (no fingerprint possible).
    """
    # ── Base + 1:1 joins ──────────────────────────────────────────────────
    base: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/molecule_dictionary.parquet").filter(
            pl.col("structure_type") == "MOL"
        )
    )

    cs: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/compound_structures.parquet").select(
            ["molregno", "canonical_smiles", "standard_inchi_key"]
        )
    )
    cp: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/compound_properties.parquet").select(
            [
                "molregno",
                "mw_freebase",
                "alogp",
                "hba",
                "hbd",
                "psa",
                "qed_weighted",
                "full_molformula",
                "num_ro5_violations",
                "heavy_atoms",
            ]
        )
    )
    mh: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/molecule_hierarchy.parquet").select(
            ["molregno", "parent_molregno", "active_molregno"]
        )
    )
    usan: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/usan_stems.parquet").select(
            [
                "stem",
                pl.col("annotation").alias("usan_stem_annotation"),
                pl.col("stem_class").alias("usan_stem_class"),
            ]
        )
    )

    base = (
        base.join(cs, on="molregno", how="left")
        .join(cp, on="molregno", how="left")
        .join(mh, on="molregno", how="left")
        .join(usan, left_on="usan_stem", right_on="stem", how="left")
        .filter(pl.col("canonical_smiles").is_not_null())
    )

    # ── 1:many aggregation joins ──────────────────────────────────────────

    _target_dict: pl.LazyFrame = pl.scan_parquet(f"{parquet_dir}/target_dictionary.parquet").select(
        ["tid", pl.col("pref_name").alias("target_name")]
    )
    mech_agg: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/drug_mechanism.parquet")
        .join(_target_dict, on="tid", how="left")
        .group_by("molregno")
        .agg(
            [
                pl.col("mechanism_of_action")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("mechanisms"),
                pl.col("action_type").drop_nulls().unique().str.join("; ").alias("action_types"),
                pl.col("target_name")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("mechanism_targets"),
            ]
        )
    )

    ind_agg: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/drug_indication.parquet")
        .group_by("molregno")
        .agg(
            [
                pl.col("mesh_heading").drop_nulls().unique().str.join("; ").alias("indications"),
                pl.col("efo_term")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("indication_efo_terms"),
                pl.col("max_phase_for_ind").max().alias("max_indication_phase"),
            ]
        )
    )

    warn_agg: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/drug_warning.parquet")
        .group_by("molregno")
        .agg(
            [
                pl.col("warning_type").drop_nulls().unique().str.join("; ").alias("warning_types"),
                pl.col("warning_class")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("warning_classes"),
                pl.col("warning_description")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("warning_descriptions"),
                pl.col("warning_country")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("warning_countries"),
                pl.col("warning_year").min().alias("first_warning_year"),
            ]
        )
    )

    syn_agg: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/molecule_synonyms.parquet")
        .group_by("molregno")
        .agg(
            [
                pl.col("synonyms").drop_nulls().unique().str.join("; ").alias("synonyms"),
            ]
        )
    )

    _atc_class: pl.LazyFrame = pl.scan_parquet(f"{parquet_dir}/atc_classification.parquet")
    atc_agg: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/molecule_atc_classification.parquet")
        .join(_atc_class, on="level5", how="left")
        .group_by("molregno")
        .agg(
            [
                pl.col("level5").drop_nulls().unique().str.join("; ").alias("atc_codes"),
                pl.col("who_name").drop_nulls().unique().str.join("; ").alias("atc_who_names"),
                pl.col("level1_description")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("atc_level1"),
                pl.col("level2_description")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("atc_level2"),
                pl.col("level3_description")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("atc_level3"),
                pl.col("level4_description")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("atc_level4"),
            ]
        )
    )

    _cr: pl.LazyFrame = pl.scan_parquet(f"{parquet_dir}/compound_records.parquet").select(
        ["record_id", "molregno"]
    )
    met_agg: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/metabolism.parquet")
        .filter(pl.col("organism") == "Homo sapiens")
        .join(_cr, left_on="substrate_record_id", right_on="record_id", how="left")
        .group_by("molregno")
        .agg(
            [
                pl.col("enzyme_name")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("metabolic_enzymes"),
                pl.col("met_conversion")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("metabolic_conversions"),
            ]
        )
    )

    _products: pl.LazyFrame = pl.scan_parquet(f"{parquet_dir}/products.parquet").select(
        ["product_id", "trade_name", "route", "dosage_form"]
    )
    form_agg: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/formulations.parquet")
        .select(["product_id", "molregno"])
        .join(_products, on="product_id", how="left")
        .group_by("molregno")
        .agg(
            [
                pl.col("trade_name").drop_nulls().unique().str.join("; ").alias("trade_names"),
                pl.col("route").drop_nulls().unique().str.join("; ").alias("routes"),
                pl.col("dosage_form").drop_nulls().unique().str.join("; ").alias("dosage_forms"),
            ]
        )
    )

    _sa: pl.LazyFrame = pl.scan_parquet(f"{parquet_dir}/structural_alerts.parquet").select(
        ["alert_id", "alert_set_id", "alert_name"]
    )
    _sas: pl.LazyFrame = pl.scan_parquet(f"{parquet_dir}/structural_alert_sets.parquet").select(
        ["alert_set_id", "set_name"]
    )
    alert_agg: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/compound_structural_alerts.parquet")
        .join(_sa, on="alert_id", how="left")
        .join(_sas, on="alert_set_id", how="left")
        .group_by("molregno")
        .agg(
            [
                pl.col("alert_name")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("structural_alerts"),
                pl.col("set_name")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("structural_alert_sets"),
            ]
        )
    )

    _ddd: pl.LazyFrame = pl.scan_parquet(f"{parquet_dir}/defined_daily_dose.parquet")
    ddd_agg: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/molecule_atc_classification.parquet")
        .join(_ddd, left_on="level5", right_on="atc_code", how="left")
        .with_columns(
            pl.concat_str(
                [pl.col("ddd_value").cast(pl.String), pl.col("ddd_units")],
                separator=" ",
                ignore_nulls=True,
            ).alias("ddd_full")
        )
        .group_by("molregno")
        .agg(
            [
                pl.col("ddd_full")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("defined_daily_doses"),
                pl.col("ddd_admr").drop_nulls().unique().str.join("; ").alias("ddd_routes"),
            ]
        )
    )

    _assays: pl.LazyFrame = pl.scan_parquet(f"{parquet_dir}/assays.parquet").select(
        ["assay_id", "tid"]
    )
    _act_targets: pl.LazyFrame = pl.scan_parquet(f"{parquet_dir}/target_dictionary.parquet").select(
        ["tid", pl.col("pref_name").alias("target_name")]
    )
    acts_agg: pl.DataFrame = _collect(
        pl.scan_parquet(f"{parquet_dir}/activities.parquet")
        .filter((pl.col("standard_flag") == 1) & pl.col("pchembl_value").is_not_null())
        .join(_assays, on="assay_id", how="left")
        .join(_act_targets, on="tid", how="left")
        .group_by("molregno")
        .agg(
            [
                pl.col("pchembl_value").max().alias("max_pchembl"),
                pl.col("pchembl_value").mean().round(2).alias("mean_pchembl"),
                pl.len().alias("activity_count"),
                pl.col("standard_type")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("activity_types"),
                pl.col("target_name")
                .drop_nulls()
                .unique()
                .str.join("; ")
                .alias("activity_targets"),
            ]
        )
    )

    return (
        base.join(mech_agg, on="molregno", how="left")
        .join(ind_agg, on="molregno", how="left")
        .join(warn_agg, on="molregno", how="left")
        .join(syn_agg, on="molregno", how="left")
        .join(atc_agg, on="molregno", how="left")
        .join(met_agg, on="molregno", how="left")
        .join(form_agg, on="molregno", how="left")
        .join(alert_agg, on="molregno", how="left")
        .join(ddd_agg, on="molregno", how="left")
        .join(acts_agg, on="molregno", how="left")
    )


def _write_to_lancedb(
    base: pl.DataFrame,
    db: DBConnection,
    cpu_count: int = 0,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Fingerprint rows and stream them into the LanceDB compounds table.

    Fingerprinting uses the MorganGenerator API (~10× faster than the deprecated
    GetMorganFingerprintAsBitVect), making it fast enough in a single thread that
    a ProcessPoolExecutor adds only overhead. A ThreadPoolExecutor pipeline
    overlaps the LanceDB write of batch N with the fingerprinting of batch N+1.

    Args:
        base: Flat compound DataFrame (must contain 'canonical_smiles').
        db: Open LanceDB connection.
        cpu_count: Unused — kept for API compatibility.
        overwrite: If True, replace the table on first batch (idempotent re-runs).

    Returns:
        Tuple of (rows written, rows skipped due to invalid SMILES).
    """
    total_batches: int = math.ceil(len(base) / BATCH_SIZE)
    table: Table | None = None
    written: int = 0
    skipped: int = 0

    with ThreadPoolExecutor(max_workers=1) as write_exec:
        # pending_write holds the in-flight LanceDB write for the previous batch.
        # Fingerprinting of batch N+1 overlaps with the I/O of batch N.
        pending_write: Future[AddResult] | None = None
        pending_count: int = 0

        batch_iter: tqdm[pl.DataFrame] = tqdm(
            base.iter_slices(BATCH_SIZE),
            total=total_batches,
            desc="Writing batches",
        )
        for slice_df in batch_iter:
            # ── Step 1: fingerprint current batch ────────────────────────────
            smiles: list[str | None] = slice_df["canonical_smiles"].to_list()
            fps: list[np.ndarray | None] = list(map(_smiles_to_fp, smiles))
            batch_records: list[dict[str, Any]] = [
                {**row, "vector": fp.tolist()}
                for row, fp in zip(slice_df.to_dicts(), fps)
                if fp is not None
            ]
            skipped += slice_df.height - len(batch_records)

            # ── Step 2: wait for previous batch's write to finish ─────────────
            if pending_write is not None:
                pending_write.result()
                written += pending_count
                pending_write = None
                pending_count = 0
                batch_iter.set_postfix(written=f"{written:,}", skipped=skipped)

            if not batch_records:
                continue

            # ── Step 3: submit this batch's write to the background thread ────
            if table is None:
                mode = "overwrite" if overwrite else "create"
                table = db.create_table(COMPOUNDS_TABLE, data=batch_records, mode=mode)
                written += len(batch_records)
                batch_iter.set_postfix(written=f"{written:,}", skipped=skipped)
            else:
                pending_count = len(batch_records)
                pending_write = write_exec.submit(table.add, batch_records)

        # ── Flush final pending write ─────────────────────────────────────────
        if pending_write is not None:
            pending_write.result()
            written += pending_count

    if table is not None:
        table.create_scalar_index("chembl_id")
        table.create_scalar_index("standard_inchi_key")

    return written, skipped


# ── Public API ────────────────────────────────────────────────────────────────


def ingest_compounds_to_lancedb(
    parquet_dir: str = DATA_DIR,
    lancedb_dir: str = LANCEDB_DIR,
) -> None:
    """Build a LanceDB compounds table from ChEMBL parquet files.

    Joins molecule_dictionary (filtered to structure_type == 'MOL') against all
    relevant ChEMBL tables, computes Morgan fingerprint vectors in parallel, and
    streams the result to LanceDB in batches.
    """
    chembl_version: str = _resolve_chembl_version(parquet_dir)
    print(f"[ingest] ChEMBL version: {chembl_version}")

    base: pl.DataFrame = _build_flat_df(parquet_dir)
    print(f"[ingest] Flat DataFrame: {len(base.columns)} columns, {len(base):,} rows")

    db: DBConnection = lancedb.connect(f"{lancedb_dir}/chembl_{chembl_version}")
    if COMPOUNDS_TABLE in db.list_tables().tables:
        print(f"[ingest] '{COMPOUNDS_TABLE}' table exists — overwriting")

    written, skipped = _write_to_lancedb(base, db, overwrite=True)

    if written == 0:
        print("[ingest] Warning: no rows were written — check your parquet data.")
        return

    print(f"[ingest] Done. {written:,} compounds written ({skipped} skipped — invalid SMILES).")


if __name__ == "__main__":
    ingest_compounds_to_lancedb()
