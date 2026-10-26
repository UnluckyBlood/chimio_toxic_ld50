"""
show_profile.py — полный профиль молекулы.
LD50 автоматически переводится в мг/кг + GHS класс.
Запуск: python show_profile.py
"""
import io, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Draw import rdMolDraw2D

WORKDIR = Path(r"C:\project\chemio")
ESKAPE_OUT = WORKDIR / "eskape_output"

PATHOGEN_NAMES = {
    "AB": "A. baumannii", "BW": "B. subtilis (контроль)",
    "DKO": "A. baumannii KPC", "EF": "E. faecium",
    "KP": "K. pneumoniae", "PA": "P. aeruginosa",
    "SA": "S. aureus (MRSA)",
}

def ld50_mgkg(log_molkg, mw):
    return (10 ** (-log_molkg)) * mw * 1000

def ghs_class(mg):
    if mg is None: return "—"
    if mg < 5:    return "1 (смертельно)"
    if mg < 50:   return "2 (смертельно)"
    if mg < 300:  return "3 (токсично)"
    if mg < 2000: return "4 (вредно)"
    if mg < 5000: return "5 (малоопасно)"
    return "> 5000 (практически нетоксично)"

def format_ld50(log_molkg, mw):
    mg = ld50_mgkg(log_molkg, mw)
    return f"LD50 ≈ {mg:.0f} мг/кг (log mol/kg = {log_molkg:.2f}) — GHS {ghs_class(mg)}"

def load_mol(path):
    if path.suffix.lower() == ".mol":
        mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
    else:
        mol = Chem.MolFromSmiles(path.read_text().strip().split()[0])
    if mol is None: raise ValueError(f"Не удалось прочитать {path}")
    return mol, Chem.MolToSmiles(mol)

def descriptors(mol):
    return {
        "MW, г/моль": round(Descriptors.MolWt(mol), 1),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA, Å²": round(Descriptors.TPSA(mol), 1),
        "H-доноры": Lipinski.NumHDonors(mol),
        "H-акцепторы": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "Аромат. кольца": Lipinski.NumAromaticRings(mol),
    }

def toxicity(smiles):
    try:
        from admet_ai import ADMETModel
        preds = ADMETModel().predict(smiles=smiles)
        keys = {"hERG": "hERG", "DILI": "DILI", "Ames": "AMES", "LD50": "LD50_Zhu"}
        out = {}
        for l, k in keys.items():
            if isinstance(preds, dict) and k in preds:
                out[l] = float(preds[k])
            elif not isinstance(preds, dict) and k in preds.columns:
                out[l] = float(preds[k].iloc[0])
        return out
    except Exception as e:
        print(f"ADMET-AI: {e}")
        return {}

def get_eskape():
    jsons = list(ESKAPE_OUT.glob("*.json")) if ESKAPE_OUT.exists() else []
    if not jsons: return {}
    try:
        with open(jsons[0], encoding="utf-8") as f:
            data = json.load(f)
        for _, by_smiles in data.items():
            for _, result in by_smiles.items():
                merged = {}
                for method in ("rf", "chemprop", "rdkit"):
                    for p, v in (result.get(method) or {}).items():
                        if v not in ("", None):
                            merged.setdefault(p, []).append(float(v))
                return {p: sum(vs)/len(vs) for p, vs in merged.items()}
    except Exception:
        return {}
    return {}

def antifungal(mol):
    out = []
    if mol.HasSubstructMatch(Chem.MolFromSmarts("n1cncc1")):
        out.append("Азольное кольцо (ингибитор CYP51)")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("C=C-C=C-C=C")):
        out.append("Сопряжённая полиеновая система")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("c[Cl,Br,F]")):
        out.append("Галогенированное ароматическое кольцо")
    return out

def mol_to_image(mol, size=(500, 500)):
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return plt.imread(io.BytesIO(d.GetDrawingText()), format="png")

def build_dashboard(mol, smiles, desc, tox, esk, antif, out_png):
    fig = plt.figure(figsize=(17, 11), facecolor="#fafafa")
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3,
                  left=0.04, right=0.97, top=0.93, bottom=0.05)
    fig.suptitle("Профиль молекулы MJ221101", fontsize=22, fontweight="bold", y=0.97)
    fig.text(0.5, 0.935, smiles, ha="center", fontsize=9,
             family="monospace", color="#555")

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(mol_to_image(mol)); ax1.axis("off")
    ax1.set_title("Структура", fontsize=13, fontweight="bold", pad=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_title("Физико-химические свойства", fontsize=13, fontweight="bold", pad=8)
    tbl = ax2.table(cellText=[[k, str(v)] for k, v in desc.items()],
                    colLabels=["Параметр", "Значение"],
                    cellLoc="left", loc="center", colWidths=[0.55, 0.35])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.7)
    for (r, _), c in tbl.get_celld().items():
        c.set_edgecolor("#dddddd")
        if r == 0:
            c.set_facecolor("#34495e")
            c.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            c.set_facecolor("#f4f6f7")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.set_title("Вердикт", fontsize=13, fontweight="bold", pad=8)
    max_esk = max(esk.values()) if esk else 0
    herg = tox.get("hERG", 0)
    dil = tox.get("DILI", 0)
    ld50_log = tox.get("LD50")
    ld50_mg = ld50_mgkg(ld50_log, desc["MW, г/моль"]) if ld50_log is not None else None
    verdict = [
        ("Антибактериальная", "НЕТ" if max_esk < 0.3 else "ДА",
         "#2ca02c" if max_esk < 0.3 else "#d62728"),
        ("Антифунгальная", "НЕТ" if not antif else "МОЖЕТ БЫТЬ",
         "#2ca02c" if not antif else "#f39c12"),
        ("Кардиориск (hERG)", "ВЫСОКИЙ" if herg > 0.5 else "НИЗКИЙ",
         "#d62728" if herg > 0.5 else "#2ca02c"),
        ("Гепатотоксичность", "ВЫСОКАЯ" if dil > 0.5 else "НИЗКАЯ",
         "#d62728" if dil > 0.5 else "#2ca02c"),
    ]
    if ld50_mg is not None:
        cls = ghs_class(ld50_mg)
        color = "#d62728" if ld50_mg < 300 else "#f39c12" if ld50_mg < 2000 else "#2ca02c"
        verdict.append((f"LD50 ≈ {ld50_mg:.0f} мг/кг", f"GHS {cls}", color))
    y = 0.92
    step = 0.9 / max(len(verdict), 1)
    for label, status, color in verdict:
        ax3.add_patch(FancyBboxPatch((0.03, y - step + 0.03), 0.94, step * 0.85,
                                     boxstyle="round,pad=0.02", facecolor=color,
                                     alpha=0.15, edgecolor=color, linewidth=1.5,
                                     transform=ax3.transAxes))
        ax3.text(0.07, y - step/2, label, fontsize=10, transform=ax3.transAxes)
        ax3.text(0.93, y - step/2, status, fontsize=10, fontweight="bold",
                 color=color, ha="right", transform=ax3.transAxes)
        y -= step

    ax4 = fig.add_subplot(gs[1, 0:2])
    prob_keys = [k for k in tox if not k.startswith("LD50")]
    if prob_keys:
        vals = [tox[k] for k in prob_keys]
        colors = ["#d62728" if v > 0.5 else "#2ca02c" for v in vals]
        bars = ax4.barh(prob_keys, vals, color=colors, edgecolor="white", height=0.5)
        ax4.set_xlim(0, 1)
        ax4.axvline(0.5, color="#888", ls="--", lw=1)
        ax4.set_title("Токсичность (вероятность)", fontsize=13, fontweight="bold", pad=8)
        for b, v in zip(bars, vals):
            ax4.text(v + 0.01, b.get_y() + b.get_height()/2,
                     f"{v:.3f}", va="center", fontsize=11, fontweight="bold")
        ax4.grid(axis="x", alpha=0.3); ax4.set_axisbelow(True)

    ax5 = fig.add_subplot(gs[1, 2])
    if esk:
        paths = sorted(esk, key=lambda p: esk[p])
        vals = [esk[p] for p in paths]
        labels = [PATHOGEN_NAMES.get(p, p) for p in paths]
        colors = ["#d62728" if v > 0.5 else "#2ca02c" for v in vals]
        bars = ax5.barh(labels, vals, color=colors, edgecolor="white", height=0.5)
        ax5.set_xlim(0, 1)
        ax5.axvline(0.5, color="#888", ls="--", lw=1)
        ax5.set_title("ESKAPE (среднее)", fontsize=13, fontweight="bold", pad=8)
        ax5.tick_params(labelsize=9)
        for b, v in zip(bars, vals):
            ax5.text(v + 0.02, b.get_y() + b.get_height()/2,
                     f"{v:.3f}", va="center", fontsize=9)
        ax5.grid(axis="x", alpha=0.3); ax5.set_axisbelow(True)

    ax6 = fig.add_subplot(gs[2, 0])
    ax6.axis("off")
    ax6.set_title("LD50 (острая токсичность)", fontsize=13, fontweight="bold", pad=8)
    if ld50_log is not None:
        cls = ghs_class(ld50_mg)
        color = "#d62728" if ld50_mg < 300 else "#f39c12" if ld50_mg < 2000 else "#2ca02c"
        ax6.text(0.5, 0.72, f"{ld50_mg:.0f}", fontsize=48, fontweight="bold",
                 ha="center", color=color, transform=ax6.transAxes)
        ax6.text(0.5, 0.52, "мг/кг", fontsize=16, ha="center", transform=ax6.transAxes)
        ax6.text(0.5, 0.38, f"log mol/kg = {ld50_log:.2f}", fontsize=11,
                 ha="center", color="#666", transform=ax6.transAxes,
                 family="monospace")
        ax6.add_patch(FancyBboxPatch((0.1, 0.12), 0.8, 0.16,
                                     boxstyle="round,pad=0.02", facecolor=color,
                                     alpha=0.2, edgecolor=color, linewidth=2,
                                     transform=ax6.transAxes))
        ax6.text(0.5, 0.20, f"GHS класс {cls}", fontsize=12, fontweight="bold",
                 ha="center", color=color, transform=ax6.transAxes)

    ax7 = fig.add_subplot(gs[2, 1:3])
    ax7.axis("off")
    ax7.set_title("Противогрибковые мотивы и итог", fontsize=13,
                  fontweight="bold", pad=8)
    lines = []
    if antif:
        for a in antif:
            lines.append(f"⚠️  Обнаружено: {a}")
    else:
        lines.append("✅ Типичных противогрибковых мотивов не найдено")
    if esk:
        mp = max(esk, key=esk.get); mv = esk[mp]
        lines.append(f"🦠 Максимальная антибактериальная активность: "
                     f"{PATHOGEN_NAMES.get(mp, mp)} = {mv:.3f} "
                     f"({'активно' if mv > 0.5 else 'неактивно'})")
    if ld50_mg is not None:
        lines.append(f"☠️  Прогноз LD50 ≈ {ld50_mg:.0f} мг/кг (GHS {ghs_class(ld50_mg)})")
    lines.append(f"💊 hERG = {herg:.3f} — "
                 f"{'высокий кардиориск' if herg > 0.5 else 'низкий кардиориск'}")
    for i, line in enumerate(lines):
        ax7.text(0.02, 0.85 - i * 0.2, line, fontsize=12, transform=ax7.transAxes)

    plt.savefig(out_png, dpi=130, facecolor="#fafafa", bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Дашборд сохранён: {out_png}")

def main():
    mol_path = WORKDIR / "MJ221101.mol"
    if not mol_path.exists():
        print(f"❌ Файл не найден: {mol_path}")
        sys.exit(1)
    print("=" * 60)
    print("ПРОФИЛЬ МОЛЕКУЛЫ MJ221101")
    print("=" * 60)
    mol, smiles = load_mol(mol_path)
    print(f"\nSMILES: {smiles}\n")
    desc = descriptors(mol)
    print("--- Дескрипторы ---")
    for k, v in desc.items(): print(f"  {k}: {v}")
    print("\n--- Токсичность (ADMET-AI) ---")
    tox = toxicity(smiles)
    if tox:
        mw = desc["MW, г/моль"]
        for k, v in tox.items():
            if k == "LD50":
                print(f"  {format_ld50(v, mw)}")
            else:
                flag = "🔴 риск" if v > 0.5 else "🟢 ок"
                print(f"  {k}: {v:.3f}  {flag}")
    print("\n--- Антибактериальная активность (ESKAPE) ---")
    esk = get_eskape()
    if esk:
        for p in sorted(esk, key=lambda x: -esk[x]):
            v = esk[p]
            flag = "🔴 Active" if v > 0.5 else "🟢 Inactive"
            print(f"  {p:<4} {PATHOGEN_NAMES.get(p, p):<24} {v:.4f}  {flag}")
    else:
        print("  JSON не найден")
    print("\n--- Противогрибковые мотивы ---")
    antif = antifungal(mol)
    if antif:
        for a in antif: print(f"  ⚠️  {a}")
    else:
        print("  ✅ Типичных противогрибковых мотивов не найдено")
    print("\n📊 Строю дашборд...")
    build_dashboard(mol, smiles, desc, tox, esk, antif, WORKDIR / "dashboard_full.png")

if __name__ == "__main__":
    main()
