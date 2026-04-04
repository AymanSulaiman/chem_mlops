import re
import duckdb
from duckdb import DuckDBPyConnection
import polars as pl
import os
import shutil
import sys


def transform_data(chembl_version: str = "36") -> None:
    if not re.match(r"^\d+$", chembl_version):
        raise ValueError(
            f"Invalid chembl_version '{chembl_version}': must contain only digits"
        )

    conn: DuckDBPyConnection = duckdb.connect()
    alias = f"chembl{chembl_version}"
    db_path = f"data/chembl_{chembl_version}/chembl_{chembl_version}_sqlite/chembl_{chembl_version}.db"
    try:
        conn.execute("INSTALL sqlite;")
        conn.execute("LOAD sqlite;")
        print("Loading data")
        conn.execute("SET arrow_large_buffer_size=true;")
        conn.execute(
            f"ATTACH '{db_path}' AS {alias} (TYPE sqlite);"
        )
        tables: pl.DataFrame = conn.execute(f"SHOW TABLES FROM {alias};").pl()
        output_dir = os.path.join("data", "chembl_transform")
        os.makedirs(output_dir, exist_ok=True)
        for table in tables.select(pl.col("name")).to_series().to_list():
            print(f"Processing table: {table}")
            result: pl.DataFrame = conn.execute(f"SELECT * FROM {alias}.{table}").pl()
            if result.height > 0:
                result.write_parquet(os.path.join(output_dir, f"{table}.parquet"))
                print(f"Saved {table}.parquet")
            else:
                print(f"Table {table} is empty, skipping.")
    finally:
        conn.close()

    chembl_dir = os.path.join("data", f"chembl_{chembl_version}")
    if os.path.exists(chembl_dir):
        try:
            shutil.rmtree(chembl_dir)
            print(f"Removed directory: {chembl_dir}")
        except Exception as e:
            print(f"Failed to remove {chembl_dir}: {e}", file=sys.stderr)
    else:
        print(f"{chembl_dir} does not exist, nothing to remove.")
