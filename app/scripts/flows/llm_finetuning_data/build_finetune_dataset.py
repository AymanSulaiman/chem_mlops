import json
import polars as pl
from pathlib import Path
from app.scripts.load_data.load_data import ChemblDataLoader


def main() -> None:
    data_dir: Path = Path("data/chembl_transform")

    loader: ChemblDataLoader = ChemblDataLoader(data_dir=data_dir)
    # Load molecular structure and bioactivity data for oncology and cardiovascular compounds
    print("Loading molecular and bioactivity data...")

    # Load compound structures (contains SMILES)
    compound_structures = loader.load_table("compound_structures")
    print(f"Compound structures: {compound_structures.shape}")
    print(f"Columns: {list(compound_structures.columns)}")

    # Load activities (bioactivity measurements)
    activities = loader.load_table("activities")
    print(f"\nActivities: {activities.shape}")
    print(f"Columns: {list(activities.columns[:15])}...")  # Show first 15 columns

    # Load target dictionary
    target_dict = loader.load_table("target_dictionary")
    print(f"\nTarget dictionary: {target_dict.shape}")
    print(f"Columns: {list(target_dict.columns)}")

    # Load molecule dictionary for compound names
    molecule_dict = loader.load_table("molecule_dictionary")
    print(f"\nMolecule dictionary: {molecule_dict.shape}")
    print(f"Columns: {list(molecule_dict.columns)}")


if __name__ == '__main__':
    main()