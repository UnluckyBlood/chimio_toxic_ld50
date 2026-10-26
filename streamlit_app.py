"""
streamlit_app.py — единая панель профиля молекулы (как сайт).
Запуск: python -m streamlit run streamlit_app.py
"""
import base64, json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

WORKDIR = Path(__file__).resolve().parent
ESKAPE_OUT = WORKDIR / "eskape_output"

PATHOGEN_NAMES = {
    "AB": "A. baumannii", "BW": "B. subtilis (контроль)",
    "DKO": "A. baumannii KPC", "EF": "E. faecium",
    "KP": "K. pneumoniae", "PA": "P. aeruginosa",
    "SA": "S. aureus (MRSA)",
}

st.set_page_config(page_title="Chemio Profiler", page_icon="🧪", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
    h1 {margin-bottom: 0.2rem;}
    .stMetric {background: #f8f9fa; padding: 10px; border-radius: 8px;}
    .card {padding: 14px; border-radius: 10px; margin-bottom: 8px;}
    .card-green  {background: #e8f5e9; border-left: 5px solid #2e7d32;}
    .card-red    {background: #fdecea; border-left: 5px solid #c62828;}
    .card-yellow {background: #fff8e1; border-left: 5px solid #f9a825;}
    .card-gray   {background: #f5f5f5; border-left: 5px solid #616161;}
    .big {font-size: 42px; font-weight: 700; line-height: 1;}
    /* FIX: тёмный текст на светлых карточках */
    .card, .card b, .card strong, .card div {color: #212121 !important;}
    .card-green,  .card-green b  {color: #1b5e20 !important;}
    .card-red,    .card-red b    {color: #b71c1c !important;}
    .card-yellow, .card-yellow b {color: #7a4a00 !important;}
    .card-gray,   .card-gray b   {color: #333333 !important;}
    .card-green  .big {color: #1b5e20 !important;}
    .card-red    .big {color: #b71c1c !important;}
    .card-yellow .big {color: #7a4a00 !important;}
    .card-gray   .big {color: #333333 !important;}
    h1, h2, h3, h4, h5 {color: #212121 !important;}
</style>
""", unsafe_allow_html=True)

# ---------- Хелперы ----------
def ld50_mgkg(log_molkg, mw):
    return (10 ** (-log_molkg)) * mw * 1000

def ghs_class(mg):
    if mg is None: return "—"
    if mg < 5:    return "1 — смертельно"
    if mg < 50:   return "2 — смертельно"
    if mg < 300:  return "3 — токсично"
    if mg < 2000: return "4 — вредно"
    if mg < 5000: return "5 — малоопасно"
    return "> 5000 — практически нетоксично"

def color_for(mg):
    if mg is None: return "gray"
    if mg < 300:   return "red"
    if mg < 2000:  return "yellow"
    return "green"

def mol_svg(mol, w=340, h=340):
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    b64 = base64.b64encode(d.GetDrawingText().encode()).decode()
    return f'<div style="background:#fff;padding:10px;border-radius:10px;text-align:center"><img src="data:image/svg+xml;base64,{b64}" width="{w}"/></div>'

@st.cache_data(show_spinner="ADMET-AI считает (10–20 с)…")
def get_tox(smiles):
    try:
        from admet_ai import ADMETModel
        p = ADMETModel().predict(smiles=smiles)
        keys = {"hERG": "hERG", "DILI": "DILI", "Ames": "AMES", "LD50": "LD50_Zhu"}
        out = {}
        for l, k in keys.items():
            if isinstance(p, dict) and k in p:
                out[l] = float(p[k])
            elif not isinstance(p, dict) and k in p.columns:
                out[l] = float(p[k].iloc[0])
        return out
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False)
def get_eskape():
    js = list(ESKAPE_OUT.glob("*.json")) if ESKAPE_OUT.exists() else []
    if not js: return {}
    try:
        with open(js[0], encoding="utf-8") as f:
            data = json.load(f)
        for _, by_s in data.items():
            for _, r in by_s.items():
                m = {}
                for method in ("rf", "chemprop", "rdkit"):
                    for p, v in (r.get(method) or {}).items():
                        if v not in ("", None):
                            m.setdefault(p, []).append(float(v))
                return {p: sum(vs)/len(vs) for p, vs in m.items()}
    except Exception:
        return {}
    return {}

def antifungal(mol):
    out = []
    if mol.HasSubstructMatch(Chem.MolFromSmarts("n1cncc1")):
        out.append("Азольное кольцо (ингибитор CYP51)")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("C=C-C=C-C=C")):
        out.append("Полиеновая система (связывание с эргостеролом)")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("c[Cl,Br,F]")):
        out.append("Галогенированное ароматическое кольцо")
    return out

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🧪 Ввод")
    mode = st.radio("Формат:", ["SMILES", "InChI"], horizontal=True)

    if mode == "SMILES":
        text_in = st.text_area(
            "SMILES",
            "CN(C)CCCN(CCCN(C)C)C(=O)C(Cc1ccccc1)NC(=O)C12CC3CC(CC(C3)C1)C2",
            height=90,
        )
    else:
        text_in = st.text_area(
            "InChI",
            "InChI=1S/C31H45N5O2/c1-34(2)17-9-15-32-29(37)26(21-23-11-7-8-12-24(23)25-22-36-16-10-14-28(25)33-26)35-30(38)31-18-20-5-6-27(31)13-19(31)4-3-6/h3-4,6-8,10-12,14,16,19-20,22,26-27,33,36H,5,9,13,15,17-18,21H2,1-2,4H3,(H,32,37)(H,35,38)",
            height=140,
        )

    with st.expander("📋 Справочные LD50 (вручную)"):
        st.caption("Модель ADMET-AI не различает путь введения. Внесите литературные значения для сравнения.")
        ref_oral = st.number_input("Перорально, мг/кг", min_value=0.0, value=0.0, step=1.0, format="%.1f")
        ref_ip   = st.number_input("Внутрибрюшинно (i.p.), мг/кг", min_value=0.0, value=0.0, step=1.0, format="%.1f")
        ref_iv   = st.number_input("Внутривенно (i.v.), мг/кг", min_value=0.0, value=0.0, step=1.0, format="%.1f")
        ref_notes = st.text_input("Источник / примечание", "")
        st.divider()
        calc_ip = st.checkbox("Рассчитать i.p. из oral (эмпирически)", value=True,
                              help="LD50(i.p.) ≈ LD50(oral) × (RAF_oral / RAF_ip) = LD50(oral) × 0.33")
        ip_factor = st.number_input("Коэффициент oral → i.p.", min_value=0.01, max_value=1.0,
                                    value=0.33, step=0.01, format="%.2f",
                                    disabled=not calc_ip)

    refresh = st.button("🔄 Пересчитать", use_container_width=True, type="primary")
    st.caption("ESKAPE берётся из кэша eskape_output\\*.json")

# --- Парсинг: SMILES или InChI ---
if not text_in or not text_in.strip():
    st.title("🧪 Chemio Profiler")
    st.info("Введите SMILES или InChI в боковой панели.")
    st.stop()

if mode == "SMILES":
    mol = Chem.MolFromSmiles(text_in.strip())
else:
    mol = Chem.MolFromInchi(text_in.strip())

if mol is None:
    st.title("🧪 Chemio Profiler")
    st.error(f"Не удалось распарсить {mode}. Проверьте строку — вероятно, битый фрагмент.")
    st.stop()

smiles = Chem.MolToSmiles(mol)
if refresh:
    get_tox.clear()
    get_eskape.clear()

tox = get_tox(smiles)
esk = get_eskape()

desc = {
    "MW, г/моль": round(Descriptors.MolWt(mol), 1),
    "LogP": round(Descriptors.MolLogP(mol), 2),
    "TPSA, Å²": round(Descriptors.TPSA(mol), 1),
    "H-доноры": Lipinski.NumHDonors(mol),
    "H-акцепторы": Lipinski.NumHAcceptors(mol),
    "RotBonds": Lipinski.NumRotatableBonds(mol),
    "Аромат. кольца": Lipinski.NumAromaticRings(mol),
}

# ---------- Заголовок ----------
st.markdown("# 🧪 Профиль молекулы")
st.caption(f"SMILES: `{smiles}`  ·  Формула: **{rdMolDescriptors.CalcMolFormula(mol)}**")

herg = tox.get("hERG", 0)
dil = tox.get("DILI", 0)
ld50_log = tox.get("LD50")
ld50 = ld50_mgkg(ld50_log, desc["MW, г/моль"]) if ld50_log is not None else None
max_esk = max(esk.values()) if esk else 0
antif = antifungal(mol)

# ---------- Верхний ряд: молекула + вердикт + дескрипторы ----------
c1, c2, c3 = st.columns([1, 1.1, 1.4])

with c1:
    st.markdown("#### Структура")
    st.markdown(mol_svg(mol), unsafe_allow_html=True)

with c2:
    st.markdown("#### Вердикт")

    def card(label, status, style):
        st.markdown(
            f'<div class="card card-{style}"><b>{label}:</b> {status}</div>',
            unsafe_allow_html=True,
        )

    card("Антибактериальная",
         "ДА" if max_esk >= 0.5 else "НЕТ",
         "red" if max_esk >= 0.5 else "green")
    card("Антифунгальная",
         "МОЖЕТ БЫТЬ" if antif else "НЕТ",
         "yellow" if antif else "green")
    card("Кардиориск (hERG)",
         f"{'ВЫСОКИЙ' if herg > 0.5 else 'НИЗКИЙ'} ({herg:.3f})",
         "red" if herg > 0.5 else "green")
    card("Гепатотоксичность",
         f"{'ВЫСОКАЯ' if dil > 0.5 else 'НИЗКАЯ'} ({dil:.3f})",
         "red" if dil > 0.5 else "green")

with c3:
    st.markdown("#### Физико-химические свойства")
    st.dataframe(
        pd.DataFrame({"Значение": [str(v) for v in desc.values()]},
                     index=list(desc.keys())),
        use_container_width=True, height=290,
    )

# ---------- Средний ряд: LD50 + токсичность + ESKAPE ----------
c4, c5, c6 = st.columns([1, 1.4, 1.4])

with c4:
    st.markdown("#### LD50 (острая токсичность)")
    if ld50 is not None:
        col = color_for(ld50)
        st.markdown(
            f'<div class="card card-{col}" style="text-align:center">'
            f'<div class="big">{ld50:.0f}</div>'
            f'<div style="font-size:14px;color:#555;margin-top:4px">мг/кг (перорально, крысы)</div>'
            f'<div style="margin-top:10px;font-family:monospace;font-size:12px;color:#777">'
            f'log mol/kg = {ld50_log:.2f}</div>'
            f'<div style="margin-top:8px;font-weight:700;color:{"#c62828" if col == "red" else "#f9a825" if col == "yellow" else "#2e7d32"}">'
            f'GHS класс {ghs_class(ld50)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("ADMET-AI не вернул LD50")

    if ld50 is not None:
        st.caption("⚠️ Модель не различает путь введения — значение усреднённое.")
        _refs = {"Перорально": ref_oral, "i.p.": ref_ip, "i.v.": ref_iv}
        _refs = {k: v for k, v in _refs.items() if v > 0}
        if _refs:
            st.markdown("**Литературные LD50:**")
            for _lbl, _v in _refs.items():
                _delta = _v - ld50
                _sign = "🔺" if _delta > 0 else "🔻" if _delta < 0 else "="
                st.markdown(f"- {_lbl}: **{_v:.0f} мг/кг** ({_sign} {abs(_delta):.0f} от прогноза)")
            if ref_notes:
                st.caption(f"Источник: {ref_notes}")

with c5:
    st.markdown("#### Токсичность (вероятности)")
    prob_keys = [k for k in tox if not k.startswith("LD50") and k != "error"]
    if prob_keys:
        vals = [tox[k] for k in prob_keys]
        colors = ["#d62728" if v > 0.5 else "#2ca02c" for v in vals]
        fig = go.Figure(go.Bar(
            x=vals, y=prob_keys, orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in vals], textposition="auto",
        ))
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_range=[0, 1], height=260, showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("нет данных")

with c6:
    st.markdown("#### Антибактериальная (ESKAPE)")
    if esk:
        paths = sorted(esk, key=lambda p: esk[p])
        vals = [esk[p] for p in paths]
        labels = [PATHOGEN_NAMES.get(p, p) for p in paths]
        colors = ["#d62728" if v > 0.5 else "#2ca02c" for v in vals]
        fig = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in vals], textposition="auto",
        ))
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_range=[0, 1], height=260, showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет eskape_output\\*.json — сначала запустите ESKAPE.")

# ---------- Нижний ряд: antifungal + сводка ----------
c7, c8 = st.columns([1.4, 1])

with c7:
    st.markdown("#### Противогрибковые мотивы")
    if antif:
        for a in antif:
            st.warning(f"⚠️ {a}")
    else:
        st.success("✅ Типичных противогрибковых мотивов не найдено")

with c8:
    st.markdown("#### Итог")
    lines = []
    if esk:
        mp = max(esk, key=esk.get)
        lines.append(f"🦠 Макс. ESKAPE: **{PATHOGEN_NAMES.get(mp, mp)}** = {esk[mp]:.3f} "
                     f"({'активно' if esk[mp] > 0.5 else 'неактивно'})")
    if ld50 is not None:
        lines.append(f"☠️ LD50 ≈ **{ld50:.0f} мг/кг** (GHS {ghs_class(ld50)})")
    lines.append(f"💊 hERG = **{herg:.3f}** — "
                 f"{'высокий кардиориск' if herg > 0.5 else 'низкий кардиориск'}")
    if dil > 0.5:
        lines.append(f"🫀 DILI = **{dil:.3f}** — риск гепатотоксичности")
    for line in lines:
        st.markdown(f"- {line}")








