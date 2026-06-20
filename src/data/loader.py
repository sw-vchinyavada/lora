"""
Data Loader - Credit default datasets.

Primary: Default of Credit Card Clients (UCI) - 30K samples, 23 features.
Fallback: German Credit (UCI) for reproducibility.
Both downloadable without Kaggle API.
"""

import io
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path
from typing import Optional

# UCI Default of Credit Card Clients (Yeh & Lien, 2009)
# 30,000 Taiwan credit card clients, binary default prediction
UCI_CREDIT_CARD_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

# UCI German Credit (Hofmann, 1994)
GERMAN_CREDIT_URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"


def load_credit_card_default(
    data_dir: Optional[str] = None,
    url: str = UCI_CREDIT_CARD_URL,
) -> pd.DataFrame:
    """
    Load Default of Credit Card Clients from UCI.
    30,000 instances, 23 features (demographics, payment history, bill amounts).
    Target: default.payment.next.month (1 = default)
    """
    data_dir = Path(data_dir or "data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "credit_card_default.csv"

    if not csv_path.exists():
        print("Downloading Default of Credit Card Clients from UCI...")
        with urllib.request.urlopen(url) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            for name in z.namelist():
                if "xls" in name.lower():
                    df = pd.read_excel(z.open(name), header=1, engine="xlrd")
                    break
                elif "csv" in name.lower():
                    df = pd.read_csv(z.open(name))
                    break
            else:
                df = pd.read_excel(z.open(z.namelist()[0]), header=1, engine="xlrd")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        for col in df.columns:
            if "default" in col and col != "default":
                df = df.rename(columns={col: "default"})
                break
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to {csv_path}")

    return pd.read_csv(csv_path)


def load_german_credit(
    data_dir: Optional[str] = None,
    url: str = GERMAN_CREDIT_URL,
) -> pd.DataFrame:
    """Load Statlog German Credit from UCI. 1000 instances, 20 features."""
    data_dir = Path(data_dir or "data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    numeric_path = data_dir / "german.data-numeric"

    if not numeric_path.exists():
        import shutil
        zip_path = data_dir / "german_credit.zip"
        print("Downloading German Credit from UCI...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        for f in data_dir.rglob("*numeric*"):
            if f.is_file() and f.suffix != ".zip":
                shutil.copy(f, numeric_path)
                break
        zip_path.unlink(missing_ok=True)

    df = pd.read_csv(numeric_path, sep=r"\s+", header=None)
    n_cols = df.shape[1]
    df.columns = [f"attr_{i}" for i in range(n_cols - 1)] + ["class"]
    df["default"] = (df["class"] == 2).astype(int)
    return df


def load_zimbabwe_synthetic(
    n_samples: int = 50000,
    data_dir: Optional[str] = None,
    force_regenerate: bool = False,
    default_rate: float = 0.10,
) -> pd.DataFrame:
    """Load synthetic Zimbabwe alternative data (dissertation §3.2)."""
    from .zimbabwe_synthetic import load_zimbabwe_synthetic as _load_synthetic
    return _load_synthetic(n_samples=n_samples, data_dir=data_dir, force_regenerate=force_regenerate, default_rate=default_rate)


def load_dataset(
    source: str = "zimbabwe_synthetic",
    data_dir: Optional[str] = None,
    n_samples: int = 50000,
    default_rate: float = 0.10,
) -> pd.DataFrame:
    """
    Load dataset by name.
    source: 'zimbabwe_synthetic' | 'credit_card_default' | 'german_credit' | path to CSV
    """
    if source == "zimbabwe_synthetic":
        return load_zimbabwe_synthetic(n_samples=n_samples, data_dir=data_dir, default_rate=default_rate)
    if source == "credit_card_default":
        return load_credit_card_default(data_dir)
    if source == "german_credit":
        return load_german_credit(data_dir)
    path = Path(source)
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError(f"Dataset not found: {source}")
