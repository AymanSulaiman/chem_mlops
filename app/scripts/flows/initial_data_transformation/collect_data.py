import tarfile
from pathlib import Path

import httpx
from tqdm import tqdm

# 10-second connect timeout, 5-minute read timeout for large downloads
_DOWNLOAD_TIMEOUT = httpx.Timeout(10.0, read=300.0)


def collect_data(chembl_version: str = "37") -> None:
    Path("data").mkdir(exist_ok=True)
    url = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_{chembl_version}_sqlite.tar.gz"
    archive_path = Path("data") / f"chembl_{chembl_version}_sqlite.tar.gz"
    db_path = Path("data") / f"chembl_{chembl_version}" / f"chembl_{chembl_version}_sqlite"
    if db_path.exists():
        print(f"{db_path} already extracted, skipping download.")
        return

    # EBI throttles each connection (~50 kB/s), so a 5.8 GB pull takes over a day.
    # Faster: download it yourself with parallel connections, then re-run this:
    #   aria2c -x16 -s16 -d data <url>
    if not archive_path.exists():
        print("Downloading ChEMBL SQLite archive...")
        partial = archive_path.with_suffix(".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0) or 0)
            with (
                partial.open("wb") as f,
                tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar,
            ):
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        partial.rename(archive_path)  # only a finished download gets the real name

    print("Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path="data", filter="data")
    print("Removing archive...")
    archive_path.unlink()

    print("Done.")

if __name__ == "__main__":
    collect_data()
