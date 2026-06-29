from pathlib import Path
import pandas as pd
import xarray as xr

C3S_DIR = Path("/home/rocky/t1_data")
ERA5_DIR = Path("/home/rocky/t1_data/adrian")

print("\n=== C3S Parquet files ===")
parquet_files = sorted(C3S_DIR.rglob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

for f in parquet_files[:10]:
    print(f)

if parquet_files:
    f = parquet_files[0]
    print("\n=== First Parquet file ===")
    print(f)

    df = pd.read_parquet(f)
    print("\nShape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes)

    print("\nHead:")
    print(df.head())

    print("\nUnique values preview:")
    for col in df.columns:
        try:
            nunique = df[col].nunique()
            print(f"{col}: {nunique} unique")
            if nunique < 20:
                print(df[col].unique())
        except Exception as e:
            print(f"{col}: cannot inspect unique values: {e}")

print("\n=== ERA5-Land files ===")
era_files = sorted(ERA5_DIR.rglob("*.nc"))
print(f"Found {len(era_files)} NetCDF files")

for f in era_files[:10]:
    print(f)

if era_files:
    f = era_files[0]
    print("\n=== First ERA5 file ===")
    print(f)

    ds = xr.open_dataset(f)
    print(ds)

    print("\nVariables:")
    print(list(ds.data_vars))

    print("\nCoordinates:")
    print(list(ds.coords))
