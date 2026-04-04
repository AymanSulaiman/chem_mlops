import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.scripts.flows.initial_data_transformation.transform_data import transform_data


class TestTransformData:
    """Test suite for transform_data function (DuckDB COPY-based implementation)."""

    @pytest.fixture
    def temp_data_dir(self):
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        data_dir = Path("data")
        chembl_dir = data_dir / "chembl_36" / "chembl_36_sqlite"
        chembl_dir.mkdir(parents=True)

        yield temp_dir

        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)

    # ------------------------------------------------------------------
    # Helpers to build consistent mock connections
    # ------------------------------------------------------------------

    def _make_conn(self, tables: list[str], row_counts: dict[str, int] | None = None):
        """Return a MagicMock DuckDB connection with predictable execute behaviour."""
        if row_counts is None:
            row_counts = {t: 1 for t in tables}

        mock_conn = MagicMock()

        def side_effect(query):
            result = MagicMock()
            if "SHOW TABLES" in query:
                result.fetchall.return_value = [(t,) for t in tables]
            elif "SELECT COUNT(*)" in query:
                # Extract table name from "SELECT COUNT(*) FROM chembl36.<table>"
                for t in tables:
                    if t in query:
                        result.fetchone.return_value = (row_counts.get(t, 1),)
                        break
                else:
                    result.fetchone.return_value = (0,)
            # COPY queries return nothing meaningful
            return result

        mock_conn.execute.side_effect = side_effect
        return mock_conn

    # ------------------------------------------------------------------
    # Success path
    # ------------------------------------------------------------------

    @patch("duckdb.connect")
    @patch("os.makedirs")
    @patch("shutil.rmtree")
    def test_transform_data_success(
        self, mock_rmtree, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        mock_conn = self._make_conn(["activities", "assays"])
        mock_duckdb_connect.return_value = mock_conn

        transform_data()

        mock_conn.execute.assert_any_call("INSTALL sqlite;")
        mock_conn.execute.assert_any_call("LOAD sqlite;")
        mock_conn.execute.assert_any_call("SET arrow_large_buffer_size=true;")
        mock_conn.execute.assert_any_call(
            "ATTACH 'data/chembl_36/chembl_36_sqlite/chembl_36.db' AS chembl36 (TYPE sqlite);"
        )
        mock_conn.execute.assert_any_call("SHOW TABLES FROM chembl36;")
        mock_makedirs.assert_called_once_with("data/chembl_transform", exist_ok=True)
        mock_rmtree.assert_called_once_with("data/chembl_36")
        mock_conn.close.assert_called_once()

    @patch("duckdb.connect")
    @patch("os.makedirs")
    @patch("shutil.rmtree")
    def test_copy_called_for_nonempty_tables(
        self, mock_rmtree, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        """COPY … TO … (FORMAT PARQUET) must be issued for every non-empty table."""
        mock_conn = self._make_conn(["activities", "assays"], {"activities": 5, "assays": 3})
        mock_duckdb_connect.return_value = mock_conn

        transform_data()

        calls_str = [str(c) for c in mock_conn.execute.call_args_list]
        copy_calls = [c for c in calls_str if "COPY" in c and "FORMAT PARQUET" in c]
        assert len(copy_calls) == 2

    # ------------------------------------------------------------------
    # Empty table handling
    # ------------------------------------------------------------------

    @patch("duckdb.connect")
    @patch("os.makedirs")
    def test_empty_tables_skipped(self, mock_makedirs, mock_duckdb_connect, temp_data_dir):
        mock_conn = self._make_conn(
            ["activities", "empty_table"], {"activities": 10, "empty_table": 0}
        )
        mock_duckdb_connect.return_value = mock_conn

        with patch("builtins.print") as mock_print:
            transform_data()

        calls_str = [str(c) for c in mock_conn.execute.call_args_list]
        copy_calls = [c for c in calls_str if "COPY" in c and "FORMAT PARQUET" in c]
        assert len(copy_calls) == 1  # only activities

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "empty_table" in printed
        assert "skipping" in printed

    # ------------------------------------------------------------------
    # Failure / crash paths
    # ------------------------------------------------------------------

    @patch("duckdb.connect")
    def test_duckdb_connection_failure(self, mock_duckdb_connect, temp_data_dir):
        mock_duckdb_connect.side_effect = Exception("Connection failed")
        with pytest.raises(Exception, match="Connection failed"):
            transform_data()

    @patch("duckdb.connect")
    def test_sqlite_installation_failure(self, mock_duckdb_connect, temp_data_dir):
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn
        mock_conn.execute.side_effect = [
            Exception("Failed to install sqlite"),
        ]
        with pytest.raises(Exception, match="Failed to install sqlite"):
            transform_data()

    @patch("duckdb.connect")
    def test_database_attach_failure(self, mock_duckdb_connect, temp_data_dir):
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        def side_effect(query):
            if "ATTACH" in query:
                raise Exception("Database not found")
            return MagicMock()

        mock_conn.execute.side_effect = side_effect
        with pytest.raises(Exception, match="Database not found"):
            transform_data()

    @patch("duckdb.connect")
    @patch("os.makedirs")
    def test_copy_failure_propagates(self, mock_makedirs, mock_duckdb_connect, temp_data_dir):
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn

        def side_effect(query):
            result = MagicMock()
            if "SHOW TABLES" in query:
                result.fetchall.return_value = [("activities",)]
            elif "SELECT COUNT(*)" in query:
                result.fetchone.return_value = (100,)
            elif "COPY" in query:
                raise OSError("Disk full")
            return result

        mock_conn.execute.side_effect = side_effect
        with pytest.raises(IOError, match="Disk full"):
            transform_data()

    @patch("duckdb.connect")
    def test_connection_closed_on_exception(self, mock_duckdb_connect, temp_data_dir):
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn
        mock_conn.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            transform_data()

        mock_conn.close.assert_called_once()

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_invalid_chembl_version_rejected(self, temp_data_dir):
        with pytest.raises(ValueError, match="Invalid chembl_version"):
            transform_data(chembl_version="36; DROP TABLE foo; --")

    def test_non_digit_version_rejected(self, temp_data_dir):
        with pytest.raises(ValueError, match="Invalid chembl_version"):
            transform_data(chembl_version="abc")

    # ------------------------------------------------------------------
    # Cleanup behaviour
    # ------------------------------------------------------------------

    @patch("duckdb.connect")
    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    def test_cleanup_success(
        self, mock_rmtree, mock_exists, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        mock_conn = self._make_conn([])
        mock_duckdb_connect.return_value = mock_conn
        mock_exists.return_value = True

        transform_data()

        mock_exists.assert_called_once_with("data/chembl_36")
        mock_rmtree.assert_called_once_with("data/chembl_36")

    @patch("duckdb.connect")
    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    def test_cleanup_failure_does_not_raise(
        self, mock_rmtree, mock_exists, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        mock_conn = self._make_conn([])
        mock_duckdb_connect.return_value = mock_conn
        mock_exists.return_value = True
        mock_rmtree.side_effect = PermissionError("Permission denied")

        with patch("builtins.print"):
            transform_data()  # must not raise

    @patch("duckdb.connect")
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_cleanup_skipped_when_dir_absent(
        self, mock_exists, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        mock_conn = self._make_conn([])
        mock_duckdb_connect.return_value = mock_conn
        mock_exists.return_value = False

        with patch("builtins.print") as mock_print:
            transform_data()

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "does not exist" in printed

    # ------------------------------------------------------------------
    # Multiple tables
    # ------------------------------------------------------------------

    @patch("duckdb.connect")
    @patch("os.makedirs")
    @patch("shutil.rmtree")
    def test_multiple_tables_all_copied(
        self, mock_rmtree, mock_makedirs, mock_duckdb_connect, temp_data_dir
    ):
        tables = ["activities", "assays", "compounds", "targets"]
        mock_conn = self._make_conn(tables)
        mock_duckdb_connect.return_value = mock_conn

        transform_data()

        calls_str = [str(c) for c in mock_conn.execute.call_args_list]
        copy_calls = [c for c in calls_str if "COPY" in c and "FORMAT PARQUET" in c]
        assert len(copy_calls) == len(tables)
        for table in tables:
            assert any(table in c for c in copy_calls)
