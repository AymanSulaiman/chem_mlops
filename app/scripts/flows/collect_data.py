import os
import requests
import tarfile
from tqdm import tqdm  # Add this import


def collect_data():
    os.makedirs("data", exist_ok=True)
    url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_sqlite.tar.gz"
    archive_path = os.path.join("data", "chembl_36_sqlite.tar.gz")

    print("Downloading ChEMBL SQLite archive...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with (
            open(archive_path, "wb") as f,
            tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print("Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path="data")

    print("Removing archive...")
    os.remove(archive_path)
    print("Done.")
