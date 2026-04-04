import re
import duckdb
from duckdb import DuckDBPyConnection
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
    output_dir = os.path.join("data", "chembl_transform")

    try:
        conn.execute("INSTALL sqlite;")
        conn.execute("LOAD sqlite;")
        conn.execute("SET arrow_large_buffer_size=true;")
        conn.execute(f"ATTACH '{db_path}' AS {alias} (TYPE sqlite);")

        tables: list[str] = [
            row[0] for row in conn.execute(f"SHOW TABLES FROM {alias};").fetchall()
        ]
        os.makedirs(output_dir, exist_ok=True)

        for table in tables:
            print(f"Processing table: {table}")
            count: int = conn.execute(
                f"SELECT COUNT(*) FROM {alias}.{table}"
            ).fetchone()[0]

            if count == 0:
                print(f"  Table {table} is empty, skipping.")
                continue

            parquet_path = os.path.join(output_dir, f"{table}.parquet")
            # Stream directly from SQLite to Parquet inside DuckDB —
            # avoids loading any table into Python/Polars memory.
            conn.execute(
                f"COPY (SELECT * FROM {alias}.{table}) TO '{parquet_path}' "
                f"(FORMAT PARQUET);"
            )
            print(f"  Saved {table}.parquet ({count:,} rows)")
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

if __name__ == "__main__":
    transform_data()