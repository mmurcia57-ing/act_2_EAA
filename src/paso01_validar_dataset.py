"""Valida el dataset original del Paso 01 sin modificarlo."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "housing_train.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_path}")
    data = pd.read_csv(dataset_path)
    rows, columns = data.shape
    print(f"Archivo: {dataset_path.relative_to(project_root)}")
    print(f"Shape: ({rows}, {columns})")
    print(f"Columnas: {list(data.columns)}")
    print(f"SalePrice presente: {'si' if 'SalePrice' in data.columns else 'no'}")
    print(f"Filas esperadas: 1460; diferencia: {rows - 1460}")
    print(f"Duplicados completos: {int(data.duplicated().sum())}")
    print(f"Total de columnas: {columns}")
    print(f"SHA-256: {sha256_file(dataset_path)}")

if __name__ == "__main__":
    main()
