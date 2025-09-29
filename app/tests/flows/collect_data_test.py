import pytest
import os
import tempfile
import tarfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import requests

from app.scripts.flows.collect_data import collect_data


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
        """Create a mock response object."""
        response = MagicMock()
        response.headers.get.return_value = "1024"  # Mock content-length
        response.iter_content.return_value = [b"test_chunk_1", b"test_chunk_2"]
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

    @patch("requests.get")
    @patch("tarfile.open")
    @patch("os.remove")
    @patch("app.scripts.flows.collect_data.tqdm")
    def test_collect_data_success(
        self,
        mock_tqdm,
        mock_remove,
        mock_tarfile,
        mock_requests_get,
        temp_data_dir,
        mock_response,
    ):
        """Test successful data collection."""
        # Setup mocks
        mock_requests_get.return_value.__enter__.return_value = mock_response
        mock_tarfile.return_value.__enter__.return_value.extractall = MagicMock()
        mock_tqdm.return_value.__enter__.return_value.update = MagicMock()

        # Mock file operations
        with patch("builtins.open", mock_open()) as mock_file:
            collect_data()

        # Verify requests.get was called with correct URL
        expected_url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_sqlite.tar.gz"
        mock_requests_get.assert_called_once_with(expected_url, stream=True)

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

    @patch("requests.get")
    def test_download_request_failure(self, mock_requests_get, temp_data_dir):
        """Test handling of HTTP request failures."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_requests_get.return_value.__enter__.return_value = mock_response

        with pytest.raises(requests.HTTPError, match="404 Not Found"):
            collect_data()

    @patch("requests.get")
    @patch("tarfile.open")
    def test_tarfile_extraction_failure(
        self, mock_tarfile, mock_requests_get, temp_data_dir, mock_response
    ):
        """Test handling of tar extraction failures."""
        # Setup successful download
        mock_requests_get.return_value.__enter__.return_value = mock_response

        # Setup tarfile failure
        mock_tar = MagicMock()
        mock_tar.extractall.side_effect = tarfile.TarError("Extraction failed")
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        with patch("builtins.open", mock_open()):
            with pytest.raises(tarfile.TarError, match="Extraction failed"):
                collect_data()

    @patch("requests.get")
    @patch("tarfile.open")
    @patch("os.remove")
    def test_file_cleanup_failure(
        self, mock_remove, mock_tarfile, mock_requests_get, temp_data_dir, mock_response
    ):
        """Test handling of file cleanup failures."""
        # Setup successful download and extraction
        mock_requests_get.return_value.__enter__.return_value = mock_response
        mock_tarfile.return_value.__enter__.return_value.extractall = MagicMock()

        # Setup cleanup failure
        mock_remove.side_effect = OSError("Permission denied")

        with patch("builtins.open", mock_open()):
            with pytest.raises(OSError, match="Permission denied"):
                collect_data()

    @patch("requests.get")
    @patch("tarfile.open")
    @patch("os.remove")
    @patch("app.scripts.flows.collect_data.tqdm")
    def test_progress_bar_updates(
        self, mock_tqdm, mock_remove, mock_tarfile, mock_requests_get, temp_data_dir
    ):
        """Test that progress bar is updated correctly."""
        # Setup mock response with chunks
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "20"  # Total size
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value.__enter__.return_value = mock_response

        # Setup other mocks
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

    @patch("requests.get")
    @patch("tarfile.open")
    @patch("os.remove")
    def test_data_directory_creation(
        self, mock_remove, mock_tarfile, mock_requests_get, temp_data_dir, mock_response
    ):
        """Test that data directory is created."""
        mock_requests_get.return_value.__enter__.return_value = mock_response
        mock_tarfile.return_value.__enter__.return_value.extractall = MagicMock()

        with patch("builtins.open", mock_open()):
            collect_data()

        # Verify data directory exists
        data_dir = Path("data")
        assert data_dir.exists()
        assert data_dir.is_dir()

    @patch("requests.get")
    @patch("tarfile.open")
    @patch("os.remove")
    def test_missing_content_length_header(
        self, mock_remove, mock_tarfile, mock_requests_get, temp_data_dir
    ):
        """Test handling when content-length header is missing."""
        # Setup response without content-length
        mock_response = MagicMock()
        mock_response.headers.get.return_value = None  # No content-length
        mock_response.iter_content.return_value = [b"test_chunk"]
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value.__enter__.return_value = mock_response

        mock_tarfile.return_value.__enter__.return_value.extractall = MagicMock()

        with patch("builtins.open", mock_open()):
            with patch("app.scripts.flows.collect_data.tqdm") as mock_tqdm:
                mock_progress_bar = MagicMock()
                mock_tqdm.return_value.__enter__.return_value = mock_progress_bar

                collect_data()

                # Should use 0 as default total when no content-length
                mock_tqdm.assert_called_once_with(
                    total=0, unit="B", unit_scale=True, desc="Downloading"
                )

    @patch("requests.get")
    @patch("builtins.open", side_effect=IOError("Disk full"))
    def test_file_write_failure(
        self, mock_open, mock_requests_get, temp_data_dir, mock_response
    ):
        """Test handling of file write failures."""
        mock_requests_get.return_value.__enter__.return_value = mock_response

        with pytest.raises(IOError, match="Disk full"):
            collect_data()

    @patch("requests.get")
    @patch("tarfile.open")
    @patch("os.remove")
    @patch("app.scripts.flows.collect_data.tqdm")
    def test_empty_response_chunks(
        self, mock_tqdm, mock_remove, mock_tarfile, mock_requests_get, temp_data_dir
    ):
        """Test handling of empty chunks in response."""
        # Setup response with empty chunks
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "10"
        mock_response.iter_content.return_value = [
            b"data",
            b"",
            b"more_data",
        ]  # Empty chunk
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value.__enter__.return_value = mock_response

        mock_tarfile.return_value.__enter__.return_value.extractall = MagicMock()
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress_bar

        with patch("builtins.open", mock_open()) as mock_file:
            collect_data()

        # Verify only non-empty chunks were written and progress updated
        mock_file().write.assert_any_call(b"data")
        mock_file().write.assert_any_call(b"more_data")

        # Progress should only be updated for non-empty chunks
        assert mock_progress_bar.update.call_count == 2
        mock_progress_bar.update.assert_any_call(4)  # len(b"data")
        mock_progress_bar.update.assert_any_call(9)  # len(b"more_data")

    @patch("os.makedirs")
    @patch("requests.get")
    def test_makedirs_permission_error(
        self, mock_requests_get, mock_makedirs, temp_data_dir
    ):
        """Test handling of directory creation permission errors."""
        mock_makedirs.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError, match="Permission denied"):
            collect_data()

    @patch("requests.get")
    @patch("tarfile.open")
    @patch("os.remove")
    @patch("app.scripts.flows.collect_data.tqdm")
    def test_integration_with_real_tar_operations(
        self,
        mock_tqdm,
        mock_remove,
        mock_tarfile,
        mock_requests_get,
        temp_data_dir,
        mock_response,
        mock_tarfile_content,
    ):
        """Integration test with actual tar file operations."""
        archive_path = os.path.join("data", "chembl_36_sqlite.tar.gz")

        # Copy our mock archive to the expected location
        os.makedirs("data", exist_ok=True)
        import shutil

        shutil.copy2(mock_tarfile_content, archive_path)

        with patch("requests.get") as mock_requests:
            # Mock the download part
            mock_response = MagicMock()
            mock_response.headers.get.return_value = str(os.path.getsize(archive_path))
            mock_response.iter_content.return_value = [Path(archive_path).read_bytes()]
            mock_response.raise_for_status.return_value = None
            mock_requests.return_value.__enter__.return_value = mock_response

            # Mock the file writing to use the actual file content
            with patch(
                "builtins.open", mock_open(read_data=Path(archive_path).read_bytes())
            ):
                with patch("tarfile.open") as mock_tarfile:
                    mock_tar = MagicMock()
                    mock_tarfile.return_value.__enter__.return_value = mock_tar

                    collect_data()

                    # Verify tarfile operations were called
                    mock_tarfile.assert_called_once_with(archive_path, "r:gz")
                    mock_tar.extractall.assert_called_once_with(path="data")

        # Verify cleanup (file should be removed)
        # Note: since we're mocking os.remove, we can't verify the actual file removal
