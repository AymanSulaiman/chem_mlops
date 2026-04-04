import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, mock_open, patch

import httpx
import pytest

from app.scripts.flows.initial_data_transformation.collect_data import collect_data


class TestCollectData:
    """Test suite for collect_data function."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        yield temp_dir

        os.chdir(original_cwd)
        import shutil

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_response(self):
        """Create a mock httpx response object."""
        response = MagicMock()
        response.headers = {"content-length": "1024"}
        response.iter_bytes.return_value = [b"test_chunk_1", b"test_chunk_2"]
        response.raise_for_status.return_value = None
        return response

    @pytest.fixture
    def mock_tarfile_content(self, temp_data_dir):
        """Create a mock tar.gz file with sample content."""
        # Create a temporary file to add to tar
        sample_file = Path(temp_data_dir) / "sample.txt"
        sample_file.write_text("sample content")

        # Create tar.gz archive
        archive_path = Path(temp_data_dir) / "test_archive.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(sample_file, arcname="chembl_36_sqlite/sample.txt")

        return archive_path

    @patch("httpx.stream")
    @patch("tarfile.open")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    @patch("app.scripts.flows.initial_data_transformation.collect_data.tqdm")
    def test_collect_data_success(
        self,
        mock_tqdm,
        mock_remove,
        mock_exists,
        mock_tarfile,
        mock_httpx_stream,
        temp_data_dir,
        mock_response,
    ):
        """Test successful data collection."""
        # Setup mocks
        mock_httpx_stream.return_value.__enter__.return_value = mock_response
        mock_tarfile.return_value.__enter__.return_value.extractall = MagicMock()
        mock_tqdm.return_value.__enter__.return_value.update = MagicMock()

        # Mock file operations
        with patch("builtins.open", mock_open()) as mock_file:
            collect_data()

        # Verify httpx.stream was called with correct URL and a timeout
        expected_url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_sqlite.tar.gz"
        mock_httpx_stream.assert_called_once_with("GET", expected_url, follow_redirects=True, timeout=ANY)

        # Verify file was opened for writing
        mock_file.assert_called_once_with(
            os.path.join("data", "chembl_36_sqlite.tar.gz"), "wb"
        )

        # Verify tarfile operations
        mock_tarfile.assert_called_once_with(
            os.path.join("data", "chembl_36_sqlite.tar.gz"), "r:gz"
        )

        # Verify cleanup
        mock_remove.assert_called_once_with(
            os.path.join("data", "chembl_36_sqlite.tar.gz")
        )

    @patch("httpx.stream")
    def test_download_request_failure(self, mock_httpx_stream, temp_data_dir):
        """Test handling of HTTP request failures."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock()
        )
        mock_httpx_stream.return_value.__enter__.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            collect_data()

    @patch("httpx.stream")
    @patch("tarfile.open")
    def test_tarfile_extraction_failure(
        self, mock_tarfile, mock_httpx_stream, temp_data_dir, mock_response
    ):
        """Test handling of tar extraction failures."""
        mock_httpx_stream.return_value.__enter__.return_value = mock_response

        mock_tar = MagicMock()
        mock_tar.extractall.side_effect = tarfile.TarError("Extraction failed")
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        with patch("builtins.open", mock_open()):
            with pytest.raises(tarfile.TarError, match="Extraction failed"):
                collect_data()

    @patch("httpx.stream")
    @patch("tarfile.open")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_file_cleanup_failure(
        self, mock_remove, mock_exists, mock_tarfile, mock_httpx_stream, temp_data_dir, mock_response
    ):
        """Test handling of file cleanup failures."""
        mock_httpx_stream.return_value.__enter__.return_value = mock_response
        mock_tarfile.return_value.__enter__.return_value.extractall = MagicMock()
        mock_remove.side_effect = OSError("Permission denied")

        with patch("builtins.open", mock_open()):
            with pytest.raises(OSError, match="Permission denied"):
                collect_data()

    @patch("httpx.stream")
    @patch("tarfile.open")
    @patch("os.remove")
    @patch("app.scripts.flows.initial_data_transformation.collect_data.tqdm")
    def test_progress_bar_updates(
        self, mock_tqdm, mock_remove, mock_tarfile, mock_httpx_stream, temp_data_dir
    ):
        """Test that progress bar is updated correctly."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "20"}
        mock_response.iter_bytes.return_value = [b"chunk1", b"chunk2"]
        mock_response.raise_for_status.return_value = None
        mock_httpx_stream.return_value.__enter__.return_value = mock_response

        mock_tarfile.return_value.__enter__.return_value.extractall = MagicMock()
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress_bar

        with patch("builtins.open", mock_open()):
            collect_data()

        # Verify progress bar was created with correct parameters
        mock_tqdm.assert_called_once_with(
            total=20, unit="B", unit_scale=True, desc="Downloading"
        )

        # Verify progress bar was updated for each chunk
        assert mock_progress_bar.update.call_count == 2
        mock_progress_bar.update.assert_any_call(6)  # len(b"chunk1")
        mock_progress_bar.update.assert_any_call(6)  # len(b"chunk2")

    @patch("httpx.stream")
    @patch("tarfile.open")
    @patch("os.remove")
    def test_data_directory_creation(
        self, mock_remove, mock_tarfile, mock_httpx_stream, temp_data_dir, mock_response
    ):
        """Test that data directory is created."""
        mock_httpx_stream.return_value.__enter__.return_value = mock_response
        mock_tarfile.return_value.__enter__.return_value.extractall = MagicMock()

        with patch("builtins.open", mock_open()):
            collect_data()

        data_dir = Path("data")
        assert data_dir.exists()
        assert data_dir.is_dir()

    @patch("httpx.stream")
    @patch("tarfile.open")
    @patch("os.remove")
    def test_missing_content_length_header(
        self, mock_remove, mock_tarfile, mock_httpx_stream, temp_data_dir
    ):
        """Test handling when content-length header is missing."""
        mock_response = MagicMock()
        mock_response.headers = {}  # No content-length
        mock_response.iter_bytes.return_value = [b"test_chunk"]
        mock_response.raise_for_status.return_value = None
        mock_httpx_stream.return_value.__enter__.return_value = mock_response

        mock_tarfile.return_value.__enter__.return_value.extractall = MagicMock()

        with patch("builtins.open", mock_open()):
            with patch("app.scripts.flows.initial_data_transformation.collect_data.tqdm") as mock_tqdm:
                mock_progress_bar = MagicMock()
                mock_tqdm.return_value.__enter__.return_value = mock_progress_bar

                collect_data()

                mock_tqdm.assert_called_once_with(
                    total=0, unit="B", unit_scale=True, desc="Downloading"
                )

    @patch("httpx.stream")
    @patch("builtins.open", side_effect=OSError("Disk full"))
    def test_file_write_failure(
        self, mock_open, mock_httpx_stream, temp_data_dir, mock_response
    ):
        """Test handling of file write failures."""
        mock_httpx_stream.return_value.__enter__.return_value = mock_response

        with pytest.raises(IOError, match="Disk full"):
            collect_data()

    @patch("os.makedirs")
    @patch("httpx.stream")
    def test_makedirs_permission_error(
        self, mock_httpx_stream, mock_makedirs, temp_data_dir
    ):
        """Test handling of directory creation permission errors."""
        mock_makedirs.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError, match="Permission denied"):
            collect_data()

    @patch("httpx.stream")
    @patch("tarfile.open")
    @patch("os.remove")
    @patch("app.scripts.flows.initial_data_transformation.collect_data.tqdm")
    def test_integration_with_real_tar_operations(
        self,
        mock_tqdm,
        mock_remove,
        mock_tarfile,
        mock_httpx_stream,
        temp_data_dir,
        mock_response,
        mock_tarfile_content,
    ):
        """Integration test with actual tar file operations."""
        archive_path = os.path.join("data", "chembl_36_sqlite.tar.gz")

        os.makedirs("data", exist_ok=True)
        import shutil

        shutil.copy2(mock_tarfile_content, archive_path)

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(os.path.getsize(archive_path))}
        mock_resp.iter_bytes.return_value = [Path(archive_path).read_bytes()]
        mock_resp.raise_for_status.return_value = None
        mock_httpx_stream.return_value.__enter__.return_value = mock_resp

        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        with patch("builtins.open", mock_open(read_data=Path(archive_path).read_bytes())):
            collect_data()

        mock_tarfile.assert_called_once_with(archive_path, "r:gz")
        mock_tar.extractall.assert_called_once_with(path="data", filter="data")
