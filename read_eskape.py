import json
from pathlib import Path
import pandas as pd

out = Path(r"C:\project\chemio\eskape_output")

# --- JSON ---
jf = out / "eskape_input.csv.json"
print("=" * 50)
print("СОДЕРЖИМОЕ JSON:")
print("=" * 50)
if jf.exists():
    try:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"⚠️ Не читается как JSON: {e}")
        print(f"Первые 500 символов: {jf.read_text()[:500]}")
else:
    print("❌ JSON отсутствует")

# --- RF-предсказания ---
print("\n" + "=" * 50)
print("RF-ПРЕДСКАЗАНИЯ (по файлам *_rf.csv):")
print("=" * 50)
results = {}
for f in sorted(out.glob("*_rf.csv")):
    pathogen = f.stem.split("-")[-1].replace("_rf", "")
    try:
        df = pd.read_csv(f, header=None)
        # Ищем первое числовое значение
        num = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        if len(num):
            results[pathogen] = float(num.iloc[0])
            print(f"  {pathogen}: {results[pathogen]:.4f}")
        else:
            print(f"  {pathogen}: ⚠️ нет числового значения")
            print(f"    Содержимое файла: {f.read_text()!r}")
    except Exception as e:
        print(f"  {pathogen}: ❌ {e}")

if results:
    print("\n--- Итог по RF ---")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        flag = "🔴 Active" if v > 0.5 else "🟢 Inactive"
        print(f"  {k}: {v:.4f}  {flag}")
