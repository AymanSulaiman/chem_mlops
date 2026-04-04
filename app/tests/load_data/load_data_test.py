import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from app.scripts.load_data.load_data import ChemblDataLoader


class TestChemblDataLoader:
    """Test suite for ChemblDataLoader class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory with sample parquet files for testing."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "test_chembl_transform"
        data_dir.mkdir(parents=True)

        # Create sample dataframes
        df1 = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["compound_a", "compound_b", "compound_c"],
                "value": [10.5, 20.3, 15.7],
            }
        )

        df2 = pl.DataFrame(
            {
                "assay_id": [101, 102, 103],
                "type": ["binding", "functional", "admet"],
                "result": [1.2, 2.4, 0.8],
            }
        )

        df3 = pl.DataFrame(
            {"mol_id": [1001, 1002], "smiles": ["CCO", "CCN"], "mw": [46.07, 45.08]}
        )

        # Write test parquet files
        df1.write_parquet(data_dir / "activities.parquet")
        df2.write_parquet(data_dir / "assays.parquet")
        df3.write_parquet(data_dir / "molecule_dictionary.parquet")

        yield str(data_dir)

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def empty_data_dir(self):
        """Create an empty temporary directory."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "empty_chembl"
        data_dir.mkdir(parents=True)

        yield str(data_dir)

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def loader(self, temp_data_dir):
        """Create a ChemblDataLoader instance with test data."""
        return ChemblDataLoader(data_dir=temp_data_dir)

    def test_init_with_valid_directory(self, temp_data_dir):
        """Test initialization with a valid data directory."""
        loader = ChemblDataLoader(data_dir=temp_data_dir)
        assert loader.data_dir == Path(temp_data_dir)
        assert len(loader.available_tables) == 3
        assert "activities" in loader.available_tables
        assert "assays" in loader.available_tables
        assert "molecule_dictionary" in loader.available_tables

    def test_init_with_nonexistent_directory(self):
        """Test initialization with a non-existent directory."""
        with pytest.raises(FileNotFoundError, match="Data directory .* not found"):
            ChemblDataLoader(data_dir="/nonexistent/path")

    def test_init_with_empty_directory(self, empty_data_dir):
        """Test initialization with an empty directory."""
        loader = ChemblDataLoader(data_dir=empty_data_dir)
        assert len(loader.available_tables) == 0

    def test_list_tables(self, loader):
        """Test listing available tables."""
        tables = loader.list_tables()
        assert isinstance(tables, list)
        assert len(tables) == 3
        assert set(tables) == {"activities", "assays", "molecule_dictionary"}

    def test_get_table_path_valid(self, loader):
        """Test getting path for a valid table."""
        path = loader.get_table_path("activities")
        assert isinstance(path, Path)
        assert path.name == "activities.parquet"

    def test_get_table_path_invalid(self, loader):
        """Test getting path for an invalid table."""
        with pytest.raises(ValueError, match="Table 'nonexistent' not found"):
            loader.get_table_path("nonexistent")

    def test_load_table_valid(self, loader):
        """Test loading a valid table."""
        df = loader.load_table("activities")
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 3)
        assert df.columns == ["id", "name", "value"]

    def test_load_table_caching(self, loader):
        """Test that caching works correctly."""
        # First load should cache the data
        df1 = loader.load_table("activities", use_cache=True)

        # Second load should return cached data
        df2 = loader.load_table("activities", use_cache=True)

        # Should be the same object in memory
        assert df1 is df2
        assert "activities" in loader._cache

    def test_load_table_no_caching(self, loader):
        """Test loading without caching."""
        df1 = loader.load_table("activities", use_cache=False)
        df2 = loader.load_table("activities", use_cache=False)

        # Should be different objects in memory
        assert df1 is not df2
        # But should have same content
        assert df1.equals(df2)

    def test_load_table_invalid(self, loader):
        """Test loading an invalid table."""
        with pytest.raises(ValueError, match="Table 'nonexistent' not found"):
            loader.load_table("nonexistent")

    def test_load_all(self, loader):
        """Test loading all tables."""
        tables = loader.load_all()
        assert isinstance(tables, dict)
        assert len(tables) == 3
        assert set(tables.keys()) == {"activities", "assays", "molecule_dictionary"}

        for table_name, df in tables.items():
            assert isinstance(df, pl.DataFrame)
            assert df.height > 0

    def test_load_tables_specific(self, loader):
        """Test loading specific tables."""
        table_names = ["activities", "assays"]
        tables = loader.load_tables(table_names)

        assert isinstance(tables, dict)
        assert len(tables) == 2
        assert set(tables.keys()) == set(table_names)

    def test_load_tables_with_invalid(self, loader):
        """Test loading tables with invalid table name."""
        table_names = ["activities", "nonexistent"]
        with pytest.raises(ValueError, match="Table 'nonexistent' not found"):
            loader.load_tables(table_names)

    def test_get_table_info(self, loader):
        """Test getting table information."""
        info = loader.get_table_info("activities")

        assert isinstance(info, dict)
        assert info["name"] == "activities"
        assert info["shape"] == (3, 3)
        assert info["columns"] == ["id", "name", "value"]
        assert "dtypes" in info
        assert "file_size_mb" in info
        assert "memory_usage_mb" in info

    def test_get_all_table_info(self, loader):
        """Test getting information for all tables."""
        all_info = loader.get_all_table_info()

        assert isinstance(all_info, dict)
        assert len(all_info) == 3
        assert set(all_info.keys()) == {"activities", "assays", "molecule_dictionary"}

        for table_name, info in all_info.items():
            assert info["name"] == table_name
            assert "shape" in info
            assert "columns" in info

    def test_get_summary(self, loader):
        """Test getting summary of all tables."""
        summary = loader.get_summary()

        assert isinstance(summary, pl.DataFrame)
        assert summary.shape[0] == 3  # 3 tables
        assert "table_name" in summary.columns
        assert "file_size_mb" in summary.columns
        assert "num_columns" in summary.columns
        assert "file_path" in summary.columns

    def test_clear_cache(self, loader):
        """Test clearing the cache."""
        # Load some data to populate cache
        loader.load_table("activities", use_cache=True)
        assert "activities" in loader._cache

        # Clear cache
        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_search_tables(self, loader):
        """Test searching for tables."""
        # Test exact match
        results = loader.search_tables("activities")
        assert results == ["activities"]

        # Test partial match
        results = loader.search_tables("assay")
        assert "assays" in results

        # Test case insensitive
        results = loader.search_tables("MOLECULE")
        assert "molecule_dictionary" in results

        # Test no match
        results = loader.search_tables("nonexistent")
        assert results == []

    def test_lazy_load_table(self, loader):
        """Test lazy loading a table."""
        lazy_df = loader.lazy_load_table("activities")
        assert isinstance(lazy_df, pl.LazyFrame)

        # Collect to verify it works
        df = lazy_df.collect()
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 3)

    def test_sample_table(self, loader):
        """Test sampling from a table."""
        # Test with default sample size
        sample = loader.sample_table("activities")
        assert isinstance(sample, pl.DataFrame)
        assert sample.shape[0] <= 1000  # Should be <= default limit

        # Test with custom sample size
        sample = loader.sample_table("activities", n=2)
        assert sample.shape[0] <= 2

    def test_repr(self, loader):
        """Test string representation."""
        repr_str = repr(loader)
        assert "ChemblDataLoader" in repr_str
        assert "tables=3" in repr_str
        assert str(loader.data_dir) in repr_str

    @patch("polars.read_parquet")
    def test_load_table_polars_error(self, mock_read_parquet, loader):
        """Test handling of Polars read errors."""
        mock_read_parquet.side_effect = Exception("Parquet read error")

        with pytest.raises(Exception, match="Parquet read error"):
            loader.load_table("activities")

    def test_concurrent_access(self, loader):
        """Test that multiple table accesses work correctly."""
        # Load multiple tables concurrently
        df1 = loader.load_table("activities")
        df2 = loader.load_table("assays")
        df3 = loader.load_table("molecule_dictionary")

        assert df1.shape == (3, 3)
        assert df2.shape == (3, 3)
        assert df3.shape == (2, 3)

        # Verify cache contains all tables
        assert len(loader._cache) == 3

    def test_file_permissions_error(self, temp_data_dir):
        """Test handling of file permission errors."""
        # Create loader first
        loader = ChemblDataLoader(data_dir=temp_data_dir)

        # Make file unreadable (on Unix systems)
        activities_path = loader.get_table_path("activities")
        try:
            activities_path.chmod(0o000)

            with pytest.raises(PermissionError):
                loader.load_table("activities")
        finally:
            # Restore permissions for cleanup
            activities_path.chmod(0o644)

    def test_empty_parquet_file(self, temp_data_dir):
        """Test handling of empty parquet files."""
        # Create an empty dataframe and save it
        empty_df = pl.DataFrame()
        empty_file_path = Path(temp_data_dir) / "empty_table.parquet"
        empty_df.write_parquet(empty_file_path)

        # Recreate loader to pick up the new file
        loader = ChemblDataLoader(data_dir=temp_data_dir)

        # Should be able to load empty table
        df = loader.load_table("empty_table")
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (0, 0)
