import duckdb
from duckdb import DuckDBPyConnection
import polars as pl
import os


def transform_data() -> None:
    conn: DuckDBPyConnection = duckdb.connect()
    conn.execute("INSTALL sqlite;")
    conn.execute("LOAD sqlite;")
    print("Loading data")
    conn.execute("SET arrow_large_buffer_size=true;")
    conn.execute(
        "ATTACH 'data/chembl_36/chembl_36_sqlite/chembl_36.db' AS chembl36 (TYPE sqlite);"
    )
    tables: pl.DataFrame = conn.execute("SHOW TABLES FROM chembl36;").pl()
    output_dir = os.path.join("data", "chembl_transform")
    os.makedirs(output_dir, exist_ok=True)
    for table in tables.select(pl.col("name")).to_series().to_list():
        print(f"Processing table: {table}")
        result: pl.DataFrame = conn.execute(f"SELECT * FROM chembl36.{table}").pl()
        if result.height > 0:
            result.write_parquet(os.path.join(output_dir, f"{table}.parquet"))
            print(f"Saved {table}.parquet")
        else:
            print(f"Table {table} is empty, skipping.")
