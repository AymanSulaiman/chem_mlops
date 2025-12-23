import json
import polars as pl
from pathlib import Path
from app.scripts.load_data.load_data import ChemblDataLoader
from toon import encode
from typing import Iterable, Any

OUT_DIR = Path("data/llm_finetune")

MAX_MOLECULES = 10000

MAX_ACTIVITIES_PER_MOLECULE = 25

TRAIN_FRACTION = 0.9


def load_tables() -> tuple[pl.DataFrame]:
    """
    Loads the tables with the ChemblDataLoader library
    """
    data_dir: Path = Path("data/chembl_transform")

    loader: ChemblDataLoader = ChemblDataLoader(data_dir=data_dir)

    compound_structures = loader.load_table("compound_structures")
    activities = loader.load_table("activities")
    molecule_dict = loader.load_table("molecule_dictionary")

    # Remove target_dict - it's not used anywhere
    return compound_structures, activities, molecule_dict


def join_tables(
    compound_structures: pl.DataFrame,
    activities: pl.DataFrame,
    molecule_dict: pl.DataFrame,
) -> pl.DataFrame:
    """
    Join activities with molecule + structure info on molregno.
    """
    print("Joining tables on molregno...")

    mol_cols = [
        "molregno",
        "pref_name",
        "chembl_id",
        "max_phase",
        "therapeutic_flag",
        "molecule_type",
        "structure_type",
        "natural_product",
        "first_in_class",
        "black_box_warning",
    ]

    struct_cols = [
        "molregno",
        "canonical_smiles",
        "standard_inchi",
        "standard_inchi_key",
    ]

    joined = activities.join(
        molecule_dict.select(mol_cols), on="molregno", how="inner"
    ).join(compound_structures.select(struct_cols), on="molregno", how="left")

    print("Joined dataframe:", joined.shape)
    return joined


def filter_activities(activities: pl.DataFrame) -> pl.DataFrame:
    """
    Basic cleaning: keep rows with a numeric standard_value and/or pchembl_value,
    and drop obviously bad / incomplete records.
    """
    print("Filtering activities...")
    act = (
        activities.filter(
            pl.any_horizontal(
                pl.col("pchembl_value").is_not_null(),
                pl.col("standard_value").is_not_null(),
            )
        )
        # keep only a few columns we care about
        .select(
            [
                "activity_id",
                "assay_id",
                "doc_id",
                "record_id",
                "molregno",
                "standard_relation",
                "standard_value",
                "standard_units",
                "standard_type",
                "pchembl_value",
                "data_validity_comment",
                "activity_comment",
            ]
        )
    )
    print("Filtered activities:", act.shape)
    return act


def build_activity_list(group_df: pl.DataFrame) -> list[dict[str, Any]]:
    """
    From a per-molecule group, build a list of activity dicts (capped).
    """
    # Sort by pchembl_value descending (most potent first) when available
    if "pchembl_value" in group_df.columns:
        group_df = group_df.sort("pchembl_value", descending=True, nulls_last=True)

    # Limit activities per molecule
    group_df = group_df.head(MAX_ACTIVITIES_PER_MOLECULE)

    activities: list[dict[str, Any]] = []
    for row in group_df.iter_rows(named=True):
        activities.append(
            {
                "activity_id": row["activity_id"],
                "assay_id": row["assay_id"],
                "doc_id": row["doc_id"],
                "record_id": row["record_id"],
                "standard_type": row["standard_type"],
                "standard_relation": row["standard_relation"],
                "standard_value": float(row["standard_value"])
                if row["standard_value"] is not None
                else None,
                "standard_units": row["standard_units"],
                "pchembl_value": float(row["pchembl_value"])
                if row["pchembl_value"] is not None
                else None,
                "data_validity_comment": row["data_validity_comment"],
                "activity_comment": row["activity_comment"],
            }
        )
    return activities


def build_example_obj(group_df: pl.DataFrame) -> dict[str, Any]:
    """
    Build the Python object that will be encoded to TOON for a single molecule.
    """
    # Use first row for molecule-level info (all rows share same molregno/chembl_id/etc.)
    first = group_df.row(0, named=True)

    molecule = {
        "molregno": int(first["molregno"]),
        "chembl_id": first["chembl_id"],
        "name": first["pref_name"],
        "canonical_smiles": first["canonical_smiles"],
        "standard_inchi": first["standard_inchi"],
        "standard_inchi_key": first["standard_inchi_key"],
        "molecule_type": first["molecule_type"],
        "structure_type": first["structure_type"],
        "max_phase": float(first["max_phase"])
        if first["max_phase"] is not None
        else None,
        "therapeutic_flag": int(first["therapeutic_flag"])
        if first["therapeutic_flag"] is not None
        else None,
        "natural_product": int(first["natural_product"])
        if first["natural_product"] is not None
        else None,
        "first_in_class": int(first["first_in_class"])
        if first["first_in_class"] is not None
        else None,
        "black_box_warning": int(first["black_box_warning"])
        if first["black_box_warning"] is not None
        else None,
    }

    # Simple activity summary
    n_activities = group_df.height
    mean_pchembl = group_df["pchembl_value"].drop_nulls().mean()

    activity_summary = {
        "n_activities": int(n_activities),
        "mean_pchembl": float(mean_pchembl) if mean_pchembl is not None else None,
    }

    activities = build_activity_list(group_df)

    example = {
        "molecule": molecule,
        "activity_summary": activity_summary,
        "activities": activities,
    }
    return example


def build_prompt(example_obj: dict[str, Any]) -> str:
    """
    Build the natural-language prompt that will precede the TOON completion.
    """
    mol = example_obj["molecule"]
    summary = example_obj["activity_summary"]

    prompt = f"""You are a medicinal chemistry assistant.
Given the following ChEMBL molecule information, output a TOON representation of the data.

ChEMBL ID: {mol["chembl_id"]}
Name: {mol["name"] or "N/A"}
SMILES: {mol["canonical_smiles"]}
Standard InChI Key: {mol["standard_inchi_key"]}

Number of activities: {summary["n_activities"]}
Mean pChEMBL: {summary["mean_pchembl"]}

Respond ONLY with the TOON representation of the molecule and its activities, no extra text.
TOON:
"""
    return prompt


def iter_examples(joined: pl.DataFrame) -> Iterable[tuple[str, str]]:
    """
    Yield (prompt, completion) pairs where completion is TOON text.
    """
    # Sample a subset of molecules for training
    unique_mols = joined.select("molregno").unique()
    n_mols = unique_mols.height
    n_sample = min(MAX_MOLECULES, n_mols)

    print(f"Total unique molregno: {n_mols}, sampling {n_sample} molecules for dataset")

    sampled_mol_ids = unique_mols.sample(n=n_sample, shuffle=True, seed=42).sort(
        "molregno"
    )

    sampled = joined.join(sampled_mol_ids, on="molregno", how="inner")

    # Group by molregno
    grouped = sampled.group_by("molregno", maintain_order=False)

    for molregno, group_df in grouped:
        example_obj = build_example_obj(group_df)
        prompt = build_prompt(example_obj)

        # Encode to TOON using python-toon
        toon_text = encode(example_obj)  # returns a TOON string

        # Start completion with newline to separate clearly from prompt
        completion = "\n" + toon_text.strip() + "\n"

        yield prompt, completion


def write_jsonl_splits(
    examples: Iterable[tuple[str, str]],
    train_path: Path,
    valid_path: Path,
    train_fraction: float = TRAIN_FRACTION,
) -> None:
    """
    Write examples into train/valid JSONL files with a single 'text' field.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = list(examples)
    n_total = len(examples)
    n_train = int(n_total * train_fraction)

    print(f"Writing {n_train} train and {n_total - n_train} valid examples")

    train_ex = examples[:n_train]
    valid_ex = examples[n_train:]

    with train_path.open("w") as f_train:
        for prompt, completion in train_ex:
            # Combine prompt and completion into single text field
            text = prompt + completion
            f_train.write(json.dumps({"text": text}) + "\n")

    with valid_path.open("w") as f_valid:
        for prompt, completion in valid_ex:
            text = prompt + completion
            f_valid.write(json.dumps({"text": text}) + "\n")


def create_finetuning_dataset() -> None:
    compound_structures, activities, molecule_dict = load_tables()  # Remove target_dict

    activities_clean = filter_activities(activities)
    joined = join_tables(compound_structures, activities_clean, molecule_dict)

    examples = list(iter_examples(joined))

    train_path = OUT_DIR / "train.jsonl"
    valid_path = OUT_DIR / "valid.jsonl"

    write_jsonl_splits(examples, train_path, valid_path)


if __name__ == "__main__":
    create_finetuning_dataset()
