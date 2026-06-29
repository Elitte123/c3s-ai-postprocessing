from pathlib import Path
import pandas as pd
import re

ROOT = Path("")
OUT = Path("inventory_c3s_t2m.csv")

records = []

for d in sorted(ROOT.glob("seasonal-original-single-levels_*_2m_temperature")):
    m = re.search(r"seasonal-original-single-levels_(\d{4})_(\d{2})_2m_temperature", d.name)
    if not m:
        continue

    year = int(m.group(1))
    month = int(m.group(2))

    files = sorted(d.glob("forecast_period=*/part.0.parquet"))

    for f in files:
        records.append({
            "year": year,
            "month": month,
            "forecast_dir": str(d),
            "file": str(f),
            "forecast_period": f.parent.name.replace("forecast_period=", "")
        })

inv = pd.DataFrame(records)
print(inv.head())
print(inv.tail())
print(inv.shape)

print("\nYears:")
print(inv["year"].min(), inv["year"].max())

print("\nMonths:")
print(sorted(inv["month"].unique()))

print("\nFiles per year-month:")
print(inv.groupby(["year", "month"]).size().describe())

inv.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
