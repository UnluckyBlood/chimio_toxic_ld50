"""
full_profile.py — полный профиль молекулы.
Читает результаты ESKAPE напрямую из eskape_output/*.json,
обходя сломанный parse_results в eskape_model.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

WORKDIR = Path(__file__).resolve().parent
DB_PATH = WORKDIR / "db"

# Расшифровка аббревиатур ESKAPE
PATHOGEN_NAMES = {
    "AB":  "A. baumannii",
    "BW":  "B. subtilis (контроль)",
    "DKO": "A. baumannii (KPC)",
    "EF":  "E. faecium",
    "KP":  "K. pneumoniae",
    "PA":  "P. aeruginosa",
    "SA":  "S. aureus (MRSA)",
}


def load_mol(path: Path):
    if path.suffix.lower() == ".mol":
        mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
    else:
        mol = Chem.MolFromSmiles(path.read_text().strip().split()[0])
    if mol is None:
        raise ValueError(f"Не удалось прочитать {path}")
    return mol, Chem.MolToSmiles(mol)


def descriptors(mol):
    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "HDonors": Lipinski.NumHDonors(mol),
        "HAcceptors": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
    }


def toxicity(smiles):
    try:
        from admet_ai import ADMETModel
    except ImportError:
        return None, "ADMET-AI не установлен"
    try:
        preds = ADMETModel().predict(smiles=smiles)
        keys = {"hERG": "hERG", "DILI": "DILI", "Ames": "AMES", "LD50 (log mol/kg)": "LD50_Zhu"}
        out = {}
        if isinstance(preds, dict):
            for l, k in keys.items():
                if k in preds:
                    out[l] = float(preds[k])
        else:
            for l, c in keys.items():
                if c in preds.columns:
                    out[l] = float(preds[c].iloc[0])
        return out, None
    except Exception as e:
        return None, f"ADMET-AI: {e}"


def find_eskape():
    exe = shutil.which("eskape_model")
    if exe:
        return exe
    for n in ("eskape_model.exe", "eskape_model"):
        c = Path(sys.executable).parent / n
        if c.exists():
            return str(c)
    return None


def run_eskape(smiles: str):
    """Запускает ESKAPE и читает JSON напрямую (обходит parse_results)."""
    exe = find_eskape()
    if exe is None:
        return None, "eskape_model не найден"

    out_dir = WORKDIR / "eskape_output"
    out_dir.mkdir(exist_ok=True)
    # Очищаем старые файлы, чтобы не смешивать результаты
    for f in out_dir.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass

    csv_in = WORKDIR / "eskape_input.csv"
    pd.DataFrame({"smiles": [smiles]}).to_csv(csv_in, index=False)

    cmd = [
        exe,
        "--input_file", str(csv_in),
        "--output_directory", str(out_dir),
        "--models_directory", str(DB_PATH),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=900)
    except Exception as e:
        return None, f"Запуск ESKAPE: {e}"

    # Ищем JSON (parse_results падает, но JSON пишется до него)
    jsons = list(out_dir.glob("*.json"))
    if not jsons:
        return None, f"JSON не создан. stderr:\n{p.stderr[-800:]}"

    try:
        with open(jsons[0], encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, f"JSON не читается: {e}"

    # Структура: { "<input_file>": { "<smiles>": { "rf": {...}, "chemprop": {...}, ... } } }
    for _, by_smiles in data.items():
        for _, result in by_smiles.items():
            return result, None
    return None, "JSON пуст"


def antifungal_heuristics(mol):
    alerts = []
    if mol.HasSubstructMatch(Chem.MolFromSmarts("n1cncc1")):
        alerts.append("Азольное кольцо (ингибитор CYP51)")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("C=C-C=C-C=C")):
        alerts.append("Полиеновая система (связывание с эргостеролом)")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("c[Cl,Br,F]")):
        alerts.append("Галогенированное ароматическое кольцо")
    return alerts


def plot(tox, eskape, png):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Токсичность
    if tox:
        labs = list(tox.keys())
        vals = [tox[k] for k in labs]
        colors = ["#d62728" if (k != "LD50 (log mol/kg)" and tox[k] > 0.5) else "#2ca02c" for k in labs]
        ax[0].barh(labs, vals, color=colors)
        ax[0].set_xlim(0, 1)
        ax[0].axvline(0.5, color="gray", ls="--")
        ax[0].set_title("Токсичность (вероятность)")
        for i, v in enumerate(vals):
            ax[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    else:
        ax[0].text(0.5, 0.5, "ADMET-AI недоступен", ha="center")
        ax[0].set_title("Токсичность")

    # ESKAPE
    if eskape:
        # объединяем rf и chemprop (chemprop может быть пустым)
        merged = {}
        for method in ("rf", "chemprop", "rdkit"):
            branch = eskape.get(method) or {}
            for path, val in branch.items():
                if val not in ("", None):
                    merged.setdefault(path, []).append((method, float(val)))

        if merged:
            pathogens = sorted(merged.keys())
            # среднее по всем доступным методам
            means = [sum(v for _, v in merged[p]) / len(merged[p]) for p in pathogens]
            colors = ["#d62728" if m > 0.5 else "#2ca02c" for m in means]
            ax[1].barh(pathogens, means, color=colors)
            ax[1].set_xlim(0, 1)
            ax[1].axvline(0.5, color="gray", ls="--")
            ax[1].set_title("ESKAPE (среднее по доступным моделям)")
            for i, m in enumerate(means):
                ax[1].text(m + 0.01, i, f"{m:.3f}", va="center", fontsize=9)
        else:
            ax[1].text(0.5, 0.5, "Нет данных ESKAPE", ha="center")
            ax[1].set_title("ESKAPE")
    else:
        ax[1].text(0.5, 0.5, "ESKAPE недоступен", ha="center")
        ax[1].set_title("ESKAPE")

    plt.tight_layout()
    plt.savefig(png, dpi=150)
    print(f"\n📊 График: {png}")


def main():
    mol_path = WORKDIR / "MJ221101.mol"
    if not mol_path.exists():
        print(f"❌ Нет файла {mol_path}")
        return

    mol, smiles = load_mol(mol_path)
    print(f"SMILES: {smiles}\n")

    print("--- Дескрипторы ---")
    for k, v in descriptors(mol).items():
        print(f"{k}: {v}")

    print("\n--- Токсичность (ADMET-AI) ---")
    tox, err = toxicity(smiles)
    if tox:
        for k, v in tox.items():
            print(f"{k}: {v:.3f}")
    else:
        print(f"ℹ️  {err}")

    print("\n--- Антибактериальная активность (ESKAPE) ---")
    eskape, err = run_eskape(smiles)
    if eskape:
        # Печатаем по методам
        for method in ("rf", "chemprop", "rdkit"):
            branch = eskape.get(method) or {}
            if any(v not in ("", None) for v in branch.values()):
                print(f"\n  {method.upper()}:")
                for path, val in branch.items():
                    if val in ("", None):
                        continue
                    flag = "🔴 Active" if float(val) > 0.5 else "🟢 Inactive"
                    name = PATHOGEN_NAMES.get(path, path)
                    print(f"    {path:<4} {name:<24} {float(val):.4f}  {flag}")

        # Итог: среднее по доступным моделям
        merged = {}
        for method in ("rf", "chemprop", "rdkit"):
            for path, val in (eskape.get(method) or {}).items():
                if val not in ("", None):
                    merged.setdefault(path, []).append(float(val))
        if merged:
            print("\n  --- ИТОГ (среднее) ---")
            for path in sorted(merged, key=lambda p: -sum(merged[p])/len(merged[p])):
                m = sum(merged[path]) / len(merged[path])
                flag = "🔴 Active" if m > 0.5 else "🟢 Inactive"
                name = PATHOGEN_NAMES.get(path, path)
                print(f"    {path:<4} {name:<24} {m:.4f}  {flag}")
    else:
        print(f"ℹ️  {err}")

    print("\n--- Antifungal эвристики ---")
    alerts = antifungal_heuristics(mol)
    if alerts:
        for a in alerts:
            print("⚠️ ", a)
    else:
        print("✅ Типичных antifungal-мотивов не найдено.")

    plot(tox, eskape, WORKDIR / "profile_plots.png")


if __name__ == "__main__":
    main()
