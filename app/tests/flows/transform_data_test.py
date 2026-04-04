import pytest
import os
import tempfile
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import polars as pl
import duckdb

from app.scripts.flows.initial_data_transformation.transform_data import transform_data


class TestTransformData:
    """Test suite for transform_data function."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory structure for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        # Create the expected directory structure
        data_dir = Path("data")
        chembl_dir = data_dir / "chembl_36" / "chembl_36_sqlite"
        chembl_dir.mkdir(parents=True)

        yield temp_dir

        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_sqlite_db(self, temp_data_dir):
        """Create a mock SQLite database with sample data."""
        db_path = Path("data/chembl_36/chembl_36_sqlite/chembl_36.db")

        # Create SQLite database with sample tables
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create sample tables
        cursor.execute("""
            CREATE TABLE activities (
                activity_id INTEGER PRIMARY KEY,
                assay_id INTEGER,
                molregno INTEGER,
                standard_value REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE assays (
                assay_id INTEGER PRIMARY KEY,
                assay_type TEXT,
                description TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE empty_table (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """)

        # Insert sample data
        cursor.executemany(
            "INSERT INTO activities VALUES (?, ?, ?, ?)",
            [(1, 101, 1001, 10.5), (2, 102, 1002, 20.3), (3, 103, 1003, 15.7)],
        )

        cursor.executemany(
            "INSERT INTO assays VALUES (?, ?, ?)",
            [
                (101, "binding", "Test binding assay"),
                (102, "functional", "Test functional assay"),
            ],
        )

        # empty_table remains empty

        conn.commit()
        conn.close()

        return str(db_path)

    @pytest.fixture
    def mock_duckdb_connection(self):
        """Create a mock DuckDB connection."""
        mock_conn = MagicMock()

        # Mock the tables DataFrame
        tables_df = pl.DataFrame({"name": ["activities", "assays", "empty_table"]})
        mock_conn.execute.return_value.pl.return_value = tables_df

        return mock_conn

    @patch("duckdb.connect")
    @patch("os.makedirs")
    @patch("shutil.rmtree")
    def test_transform_data_success(
        self,
        mock_rmtree,
        mock_makedirs,
        mock_duckdb_connect,
        temp_data_dir,
        mock_sqlite_db,
    ):
        """Test successful data transformation."""
        # Setup mock connection
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        # Mock SHOW TABLES result
        tables_df = pl.DataFrame({"name": ["activities", "assays"]})
        mock_conn.execute.return_value.pl.return_value = tables_df

        # Mock SELECT results
        activities_df = pl.DataFrame(
            {
                "activity_id": [1, 2, 3],
                "assay_id": [101, 102, 103],
                "molregno": [1001, 1002, 1003],
                "standard_value": [10.5, 20.3, 15.7],
            }
        )

        assays_df = pl.DataFrame(
            {
                "assay_id": [101, 102],
                "assay_type": ["binding", "functional"],
                "description": ["Test binding", "Test functional"],
            }
        )

        # Configure side effects for different SELECT queries
        def mock_execute_side_effect(query):
            mock_result = MagicMock()
            if "SELECT * FROM chembl36.activities" in query:
                mock_result.pl.return_value = activities_df
            elif "SELECT * FROM chembl36.assays" in query:
                mock_result.pl.return_value = assays_df
            elif "SHOW TABLES FROM chembl36" in query:
                mock_result.pl.return_value = tables_df
            return mock_result

        mock_conn.execute.side_effect = mock_execute_side_effect

        # Mock polars write_parquet
        with patch.object(pl.DataFrame, "write_parquet") as mock_write_parquet:
            transform_data()

        # Verify DuckDB operations
        mock_conn.execute.assert_any_call("INSTALL sqlite;")
        mock_conn.execute.assert_any_call("LOAD sqlite;")
        mock_conn.execute.assert_any_call("SET arrow_large_buffer_size=true;")
        mock_conn.execute.assert_any_call(
            "ATTACH 'data/chembl_36/chembl_36_sqlite/chembl_36.db' AS chembl36 (TYPE sqlite);"
        )
        mock_conn.execute.assert_any_call("SHOW TABLES FROM chembl36;")

        # Verify directory creation
        mock_makedirs.assert_called_once_with("data/chembl_transform", exist_ok=True)

        # Verify parquet files were written
        assert mock_write_parquet.call_count == 2  # activities and assays

        # Verify cleanup
        mock_rmtree.assert_called_once_with("data/chembl_36")
        mock_conn.close.assert_called_once()

    @patch("duckdb.connect")
    def test_duckdb_connection_failure(self, mock_duckdb_connect, temp_data_dir):
        """Test handling of DuckDB connection failures."""
        mock_duckdb_connect.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            transform_data()

    @patch("duckdb.connect")
    def test_sqlite_installation_failure(self, mock_duckdb_connect, temp_data_dir):
        """Test handling of SQLite extension installation failures."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn
        mock_conn.execute.side_effect = [
            Exception("Failed to install sqlite"),  # INSTALL sqlite fails
            None,
            None,
            None,  # Other calls succeed
        ]

        with pytest.raises(Exception, match="Failed to install sqlite"):
            transform_data()

    @patch("duckdb.connect")
    def test_database_attach_failure(self, mock_duckdb_connect, temp_data_dir):
        """Test handling of database attachment failures."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        def mock_execute_side_effect(query):
            if "ATTACH" in query:
                raise Exception("Database not found")
            return MagicMock()

        mock_conn.execute.side_effect = mock_execute_side_effect

        with pytest.raises(Exception, match="Database not found"):
            transform_data()

    @patch("duckdb.connect")
    @patch("os.makedirs")
    def test_directory_creation_failure(
        self, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        """Test handling of directory creation failures."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        # Mock successful connection setup
        tables_df = pl.DataFrame({"name": ["activities"]})
        mock_conn.execute.return_value.pl.return_value = tables_df

        # Mock directory creation failure
        mock_makedirs.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError, match="Permission denied"):
            transform_data()

    @patch("duckdb.connect")
    @patch("os.makedirs")
    def test_empty_tables_handling(
        self, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        """Test handling of empty tables."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        # Mock tables list including an empty table
        tables_df = pl.DataFrame({"name": ["activities", "empty_table"]})

        # Mock empty DataFrame for empty_table
        empty_df = pl.DataFrame()  # height = 0
        activities_df = pl.DataFrame({"id": [1, 2], "value": [10, 20]})  # height = 2

        def mock_execute_side_effect(query):
            mock_result = MagicMock()
            if "SHOW TABLES" in query:
                mock_result.pl.return_value = tables_df
            elif "SELECT * FROM chembl36.empty_table" in query:
                mock_result.pl.return_value = empty_df
            elif "SELECT * FROM chembl36.activities" in query:
                mock_result.pl.return_value = activities_df
            return mock_result

        mock_conn.execute.side_effect = mock_execute_side_effect

        with patch.object(pl.DataFrame, "write_parquet") as mock_write_parquet:
            with patch("builtins.print") as mock_print:
                transform_data()

        # Verify only non-empty table was written
        mock_write_parquet.assert_called_once()

        # Verify appropriate messages were printed
        mock_print.assert_any_call("Table empty_table is empty, skipping.")
        mock_print.assert_any_call("Processing table: empty_table")
        mock_print.assert_any_call("Processing table: activities")

    @patch("duckdb.connect")
    @patch("os.makedirs")
    def test_parquet_write_failure(
        self, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        """Test handling of parquet file write failures."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        tables_df = pl.DataFrame({"name": ["activities"]})
        activities_df = pl.DataFrame({"id": [1, 2], "value": [10, 20]})

        def mock_execute_side_effect(query):
            mock_result = MagicMock()
            if "SHOW TABLES" in query:
                mock_result.pl.return_value = tables_df
            elif "SELECT * FROM chembl36.activities" in query:
                mock_result.pl.return_value = activities_df
            return mock_result

        mock_conn.execute.side_effect = mock_execute_side_effect

        with patch.object(pl.DataFrame, "write_parquet") as mock_write_parquet:
            mock_write_parquet.side_effect = IOError("Disk full")

            with pytest.raises(IOError, match="Disk full"):
                transform_data()

    @patch("duckdb.connect")
    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    def test_cleanup_success(
        self,
        mock_rmtree,
        mock_exists,
        mock_makedirs,
        mock_duckdb_connect,
        temp_data_dir,
    ):
        """Test successful cleanup of chembl directory."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        # Mock empty tables result
        tables_df = pl.DataFrame({"name": []})
        mock_conn.execute.return_value.pl.return_value = tables_df

        # Mock directory exists
        mock_exists.return_value = True

        transform_data()

        # Verify cleanup was attempted
        mock_exists.assert_called_once_with("data/chembl_36")
        mock_rmtree.assert_called_once_with("data/chembl_36")

    @patch("duckdb.connect")
    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    def test_cleanup_failure(
        self,
        mock_rmtree,
        mock_exists,
        mock_makedirs,
        mock_duckdb_connect,
        temp_data_dir,
    ):
        """Test handling of cleanup failures."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        tables_df = pl.DataFrame({"name": []})
        mock_conn.execute.return_value.pl.return_value = tables_df

        mock_exists.return_value = True
        mock_rmtree.side_effect = PermissionError("Permission denied")

        with patch("builtins.print") as mock_print:
            transform_data()  # Should not raise exception

        # Verify error was printed to stderr
        mock_print.assert_any_call(
            "Failed to remove data/chembl_36: Permission denied",
            file=pytest.importorskip("sys").stderr,
        )

    @patch("duckdb.connect")
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_cleanup_directory_not_exists(
        self, mock_exists, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        """Test cleanup when directory doesn't exist."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        tables_df = pl.DataFrame({"name": []})
        mock_conn.execute.return_value.pl.return_value = tables_df

        mock_exists.return_value = False

        with patch("builtins.print") as mock_print:
            transform_data()

        mock_print.assert_any_call("data/chembl_36 does not exist, nothing to remove.")

    @patch("duckdb.connect")
    @patch("os.makedirs")
    def test_sql_query_failure(self, mock_makedirs, mock_duckdb_connect, temp_data_dir):
        """Test handling of SQL query failures."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        tables_df = pl.DataFrame({"name": ["activities"]})

        def mock_execute_side_effect(query):
            mock_result = MagicMock()
            if "SHOW TABLES" in query:
                mock_result.pl.return_value = tables_df
            elif "SELECT * FROM chembl36.activities" in query:
                raise Exception("SQL execution failed")
            return mock_result

        mock_conn.execute.side_effect = mock_execute_side_effect

        with pytest.raises(Exception, match="SQL execution failed"):
            transform_data()

    @patch("duckdb.connect")
    @patch("os.makedirs")
    def test_large_table_processing(
        self, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        """Test processing of large tables."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        tables_df = pl.DataFrame({"name": ["large_table"]})

        # Create a "large" DataFrame (simulate memory usage)
        large_df = pl.DataFrame(
            {"id": list(range(10000)), "data": [f"value_{i}" for i in range(10000)]}
        )

        def mock_execute_side_effect(query):
            mock_result = MagicMock()
            if "SHOW TABLES" in query:
                mock_result.pl.return_value = tables_df
            elif "SELECT * FROM chembl36.large_table" in query:
                mock_result.pl.return_value = large_df
            return mock_result

        mock_conn.execute.side_effect = mock_execute_side_effect

        with patch.object(pl.DataFrame, "write_parquet") as mock_write_parquet:
            transform_data()

        # Verify large table was processed
        mock_write_parquet.assert_called_once()
        args, kwargs = mock_write_parquet.call_args
        assert "large_table.parquet" in args[0]

    @patch("duckdb.connect")
    @patch("os.makedirs")
    def test_multiple_tables_processing(
        self, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        """Test processing multiple tables."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        tables_df = pl.DataFrame(
            {"name": ["activities", "assays", "compounds", "targets"]}
        )

        # Create sample DataFrames for each table
        sample_dfs = {
            "activities": pl.DataFrame({"activity_id": [1, 2], "value": [10, 20]}),
            "assays": pl.DataFrame({"assay_id": [101, 102], "type": ["A", "B"]}),
            "compounds": pl.DataFrame({"compound_id": [1001], "smiles": ["CCO"]}),
            "targets": pl.DataFrame({"target_id": [2001], "name": ["Target1"]}),
        }

        def mock_execute_side_effect(query):
            mock_result = MagicMock()
            if "SHOW TABLES" in query:
                mock_result.pl.return_value = tables_df
            else:
                for table_name, df in sample_dfs.items():
                    if f"SELECT * FROM chembl36.{table_name}" in query:
                        mock_result.pl.return_value = df
                        break
            return mock_result

        mock_conn.execute.side_effect = mock_execute_side_effect

        with patch.object(pl.DataFrame, "write_parquet") as mock_write_parquet:
            transform_data()

        # Verify all tables were processed
        assert mock_write_parquet.call_count == 4

        # Verify correct file paths
        call_args = [call[0][0] for call in mock_write_parquet.call_args_list]
        expected_files = [
            "data/chembl_transform/activities.parquet",
            "data/chembl_transform/assays.parquet",
            "data/chembl_transform/compounds.parquet",
            "data/chembl_transform/targets.parquet",
        ]

        for expected_file in expected_files:
            assert any(expected_file in arg for arg in call_args)

    @patch("duckdb.connect")
    def test_connection_cleanup_on_exception(self, mock_duckdb_connect, temp_data_dir):
        """Test that connection is properly closed even when exceptions occur."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        # Make execute fail after connection is established
        mock_conn.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            transform_data()

        # Connection should still be closed
        mock_conn.close.assert_called_once()

    @patch("duckdb.connect")
    @patch("os.makedirs")
    def test_polars_dataframe_operations(
        self, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        """Test that Polars DataFrame operations work correctly."""
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        # Create a realistic tables DataFrame
        tables_df = pl.DataFrame({"name": ["test_table", "another_table"]})

        # Create test data
        test_df = pl.DataFrame(
            {"id": [1, 2, 3], "name": ["A", "B", "C"], "value": [1.1, 2.2, 3.3]}
        )

        def mock_execute_side_effect(query):
            mock_result = MagicMock()
            if "SHOW TABLES" in query:
                mock_result.pl.return_value = tables_df
            else:
                mock_result.pl.return_value = test_df
            return mock_result

        mock_conn.execute.side_effect = mock_execute_side_effect

        with patch.object(pl.DataFrame, "write_parquet") as mock_write_parquet:
            transform_data()

        # Verify that to_series().to_list() was called correctly
        # This is implicitly tested by the successful iteration over table names
        assert mock_write_parquet.call_count == 2
