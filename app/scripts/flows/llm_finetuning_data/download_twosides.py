"""
Download the TWOSIDES polypharmacy side-effect dataset from the Tatonetti Lab S3 bucket,
decompress it in memory, and save it as Parquet via Polars.

TWOSIDES contains drug-pair adverse event signals derived from FDA FAERS (Adverse Event
Reporting System). Each row is a (drug_1, drug_2, side_effect) triple with a
Proportional Reporting Ratio (PRR) indicating disproportionate co-reporting.

Reference: Tatonetti et al., Science Translational Medicine 2012.

Output: data/twosides/TWOSIDES.parquet  (~30–50 MB, column-compressed)
"""

import gzip
import io
from pathlib import Path

import httpx
import polars as pl
from tqdm import tqdm

TWOSIDES_URL = "https://tatonettilab-resources.s3.us-west-1.amazonaws.com/nsides/TWOSIDES.csv.gz"
TWOSIDES_DIR = Path("data/twosides")
TWOSIDES_PATH = TWOSIDES_DIR / "TWOSIDES.parquet"

_DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, read=600.0)


def download_twosides(
    url: str = TWOSIDES_URL,
    output_path: Path = TWOSIDES_PATH,
    force: bool = False,
) -> Path:
    """Stream-download TWOSIDES.csv.gz, decompress in memory, and write as Parquet.

    No intermediate CSV file is written to disk — the gzipped bytes are
    decompressed into a BytesIO buffer, parsed by Polars, and flushed straight
    to Parquet. This avoids ~600 MB of temporary disk usage and produces a file
    that Polars can scan ~10× faster on subsequent reads.

    Args:
        url:         Source URL (Tatonetti Lab S3).
        output_path: Destination Parquet path.
        force:       Re-download and rewrite even if the file already exists.

    Returns:
        Path to the written Parquet file.
    """
    if output_path.exists() and not force:
        print(f"TWOSIDES already present: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: stream compressed bytes into memory ───────────────────────────
    print(f"Downloading TWOSIDES from {url} ...")
    buf = io.BytesIO()
    with httpx.stream("GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0) or 0)
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in r.iter_bytes(chunk_size=65_536):
                buf.write(chunk)
                pbar.update(len(chunk))

    compressed_mb = buf.tell() / 1e6
    print(f"Downloaded {compressed_mb:.1f} MB compressed")

    # ── Step 2: decompress gzip in memory ────────────────────────────────────
    print("Decompressing ...")
    buf.seek(0)
    csv_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf) as gz:
        csv_buf.write(gz.read())

    decompressed_mb = csv_buf.tell() / 1e6
    print(f"Decompressed to {decompressed_mb:.1f} MB")

    # ── Step 3: parse CSV with Polars directly from BytesIO ──────────────────
    # Read everything as String to sidestep two real-world quirks in TWOSIDES:
    #   1. drug_1_rxnorn_id has non-integer values past row ~1000
    #   2. The file has an embedded duplicate header row mid-file (PRR == "PRR")
    # We strip the dupe header and cast numerics here so the Parquet has proper types.
    print("Parsing CSV ...")
    csv_buf.seek(0)
    df = pl.read_csv(csv_buf, encoding="utf8-lossy", infer_schema_length=0)

    before = len(df)
    df = df.filter(pl.col("PRR") != "PRR")
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} embedded header row(s)")

    df = df.with_columns([
        pl.col("A").cast(pl.Int32, strict=False),
        pl.col("B").cast(pl.Int32, strict=False),
        pl.col("C").cast(pl.Int32, strict=False),
        pl.col("D").cast(pl.Int32, strict=False),
        pl.col("PRR").cast(pl.Float32, strict=False),
        pl.col("PRR_error").cast(pl.Float32, strict=False),
        pl.col("mean_reporting_frequency").cast(pl.Float32, strict=False),
    ])
    print(f"Parsed {len(df):,} rows × {len(df.columns)} columns")

    # ── Step 4: write Parquet ─────────────────────────────────────────────────
    print(f"Writing Parquet to {output_path} ...")
    df.write_parquet(output_path, compression="zstd")
    parquet_mb = output_path.stat().st_size / 1e6
    print(f"Done. {parquet_mb:.1f} MB Parquet (compression ratio: {decompressed_mb / parquet_mb:.1f}×)")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download TWOSIDES and save as Parquet")
    parser.add_argument("--force", action="store_true", help="Re-download if already present")
    args = parser.parse_args()
    download_twosides(force=args.force)
