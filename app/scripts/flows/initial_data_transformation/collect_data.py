import os
import tarfile

import httpx
from tqdm import tqdm

# 10-second connect timeout, 5-minute read timeout for large downloads
_DOWNLOAD_TIMEOUT = httpx.Timeout(10.0, read=300.0)


def collect_data(chembl_version: str = "36") -> None:
    os.makedirs("data", exist_ok=True)
    url: str = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_{chembl_version}_sqlite.tar.gz"
    archive_path: str = os.path.join("data", f"chembl_{chembl_version}_sqlite.tar.gz")

    try:
        print("Downloading ChEMBL SQLite archive...")
        with httpx.stream("GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0) or 0)
            with (
                open(archive_path, "wb") as f,
                tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading"
                ) as pbar,
            ):
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print("Extracting archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path="data", filter="data")
    finally:
        if os.path.exists(archive_path):
            print("Removing archive...")
            os.remove(archive_path)

    print("Done.")
