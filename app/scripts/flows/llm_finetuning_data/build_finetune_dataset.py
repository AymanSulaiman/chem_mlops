from pathlib import Path

import polars as pl

from app.scripts.load_data.load_data import ChemblDataLoader


def load_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Loads the tables with the ChemblDataLoader library
    """
    data_dir: Path = Path("data/chembl_transform")

    loader: ChemblDataLoader = ChemblDataLoader(data_dir=data_dir)

    compound_structures = loader.load_table("compound_structures")
    activities = loader.load_table("activities")
    molecule_dict = loader.load_table("molecule_dictionary")

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


OUTPUT_PATH = Path("data/chembl_activity_dataset.parquet")


def create_finetuning_dataset() -> pl.DataFrame:
    compound_structures, activities, molecule_dict = load_tables()

    activities_clean = filter_activities(activities)
    joined = join_tables(compound_structures, activities_clean, molecule_dict)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joined.write_parquet(OUTPUT_PATH)
    print(f"Saved finetuning dataset to {OUTPUT_PATH} ({joined.shape[0]} rows)")

    return joined


if __name__ == "__main__":
    create_finetuning_dataset()
