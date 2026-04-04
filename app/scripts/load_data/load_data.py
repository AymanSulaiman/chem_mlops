import re
from pathlib import Path

import polars as pl


class ChemblDataLoader:
    """Easy-to-use data loader for ChEMBL parquet files using Polars."""

    def __init__(self, data_dir: str | Path = "data/chembl_transform"):
        """Initialize the data loader.

        Args:
            data_dir: Path to the directory containing parquet files
        """
        self.data_dir = Path(data_dir)
        self._cache: dict[str, pl.DataFrame] = {}
        self._discover_tables()

    def _discover_tables(self) -> None:
        """Discover available parquet files."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")

        self.available_tables = {}
        for file_path in self.data_dir.glob("*.parquet"):
            table_name = file_path.stem
            self.available_tables[table_name] = file_path

    def list_tables(self) -> list[str]:
        """Get list of available table names."""
        return list(self.available_tables.keys())

    def get_table_path(self, table_name: str) -> Path:
        """Get the file path for a specific table."""
        if table_name not in self.available_tables:
            raise ValueError(
                f"Table '{table_name}' not found. Available tables: {self.list_tables()}"
            )
        return self.available_tables[table_name]

    def load_table(self, table_name: str, use_cache: bool = True) -> pl.DataFrame:
        """Load a specific table.

        Args:
            table_name: Name of the table to load
            use_cache: Whether to use cached data if available

        Returns:
            Polars DataFrame containing the table data
        """
        if use_cache and table_name in self._cache:
            return self._cache[table_name]

        file_path = self.get_table_path(table_name)
        df = pl.read_parquet(file_path)

        if use_cache:
            self._cache[table_name] = df

        return df

    def load_all(self, use_cache: bool = True) -> dict[str, pl.DataFrame]:
        """Load all available tables.

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary mapping table names to Polars DataFrames
        """
        tables = {}
        for table_name in self.available_tables:
            tables[table_name] = self.load_table(table_name, use_cache)
        return tables

    def load_tables(
        self, table_names: list[str], use_cache: bool = True
    ) -> dict[str, pl.DataFrame]:
        """Load specific tables by name.

        Args:
            table_names: List of table names to load
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary mapping table names to Polars DataFrames
        """
        return {name: self.load_table(name, use_cache) for name in table_names}

    def get_table_info(self, table_name: str) -> dict:
        """Get information about a specific table."""
        df = self.load_table(table_name, use_cache=True)
        return {
            "name": table_name,
            "shape": df.shape,
            "columns": df.columns,
            "dtypes": dict(zip(df.columns, [str(dtype) for dtype in df.dtypes])),
            "file_size_mb": round(
                self.get_table_path(table_name).stat().st_size / (1024 * 1024), 2
            ),
            "memory_usage_mb": round(df.estimated_size("mb"), 2),
        }

    def get_all_table_info(self) -> dict[str, dict]:
        """Get information about all tables."""
        return {table: self.get_table_info(table) for table in self.available_tables}

    def get_summary(self) -> pl.DataFrame:
        """Get a summary of all tables as a Polars DataFrame."""
        summaries = []
        for table_name in self.available_tables:
            file_path = self.get_table_path(table_name)
            file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)

            # Get basic info without loading full table (lazy)
            lazy_df = pl.scan_parquet(file_path)
            columns = lazy_df.columns

            summaries.append(
                {
                    "table_name": table_name,
                    "file_size_mb": file_size_mb,
                    "num_columns": len(columns),
                    "file_path": str(file_path),
                }
            )

        return pl.DataFrame(summaries)

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()

    def search_tables(self, pattern: str) -> list[str]:
        """Search for tables matching a pattern."""
        pattern = pattern.lower()
        return [
            table
            for table in self.available_tables
            if re.search(pattern, table.lower())
        ]

    def lazy_load_table(self, table_name: str) -> pl.LazyFrame:
        """Load a table as a lazy frame for memory-efficient operations.

        Args:
            table_name: Name of the table to load lazily

        Returns:
            Polars LazyFrame for the table
        """
        file_path = self.get_table_path(table_name)
        return pl.scan_parquet(file_path)

    def sample_table(self, table_name: str, n: int = 1000) -> pl.DataFrame:
        """Get a sample of rows from a table without loading the entire dataset.

        Args:
            table_name: Name of the table to sample
            n: Number of rows to sample

        Returns:
            Polars DataFrame with sampled rows
        """
        return pl.DataFrame(self.lazy_load_table(table_name).limit(n).collect())

    def __repr__(self) -> str:
        return f"ChemblDataLoader(data_dir='{self.data_dir}', tables={len(self.available_tables)})"
