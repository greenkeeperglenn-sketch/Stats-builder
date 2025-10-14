# streamlit_app.py
# Genstat-style RB ANOVA + Matrix with your alpha toggles and multi-sheet parser

import io, math, re
from dataclasses import dataclass
import numpy as np, pandas as pd
import streamlit as st
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

st.set_page_config(page_title="Genstat-Style Analyzer", layout="wide")
st.title("Genstat-Style RB ANOVA & Matrix")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")
    alpha_options = {
        "Fungicide (0.05)": 0.05,
        "Biologicals in lab (0.10)": 0.10,
        "Biologicals in field (0.15)": 0.15,
        "Custom": None,
    }
    alpha_label = st.radio("Significance level:", list(alpha_options.keys()))
    alpha = (
        st.number_input("Custom alpha", 0.0001, 0.2, 0.05, 0.005, format="%.4f")
        if alpha_options[alpha_label] is None
        else alpha_options[alpha_label]
    )

    adjust_opts = {"None": "none", "Bonferroni": "bonferroni", "Holm": "holm", "Benjamini–Hochberg (FDR)": "fdr_bh"}
    adj_method_label = st.selectbox("Multiple testing correction across columns", list(adjust_opts.keys()), index=0)
    adj_method = adjust_opts[adj_method_label]

    input_mode = st.radio("Excel layout:", ["Single sheet (wide)", "Multi-sheet (each sheet = date)"])

# ---------- Helpers ----------
def _clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace(" ", "_") for c in out.columns]
    return out

def guess_key_columns(df: pd.DataFrame):
    def find_one(cands):
        for c in df.columns:
            cl = str(c).lower()
            if any(tag in cl for tag in cands):
                return c
        return None
    plot_col = find_one(["plot", "id"]) or "Plot"
    block_col = find_one(["block", "rep", "replicate"]) or "Block"
    trt_col = find_one(["treat", "trt"]) or "Treatment"
    excl = {plot_col, block_col, trt_col}
    numeric_cols = [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
    return plot_col, block_col, trt_col, numeric_cols

@dataclass
class AovResult:
    aov_table: pd.DataFrame
    means: pd.DataFrame
    ese: float
    sed: float
    lsd: float
    alpha: float
    df_resid: int
    mse: float
    grand_mean: float
    cv_block_units: float
    se_block_units: float
    se_block: float | None
    cv_block: float | None
    p_treatment: float
    letters: pd.DataFrame | None
    warnings: list

def randomized_block_anova(df: pd.DataFrame, response: str, trt: str, block: str, alpha: float) -> AovResult:
    warnings = []
    d = df[[response, trt, block]].dropna().copy()
    d[trt] = d[trt].astype("category")
    d[block] = d[block].astype("category")

    model = smf.ols(f"Q('{response}') ~ C(Q('{trt}')) + C(Q('{block}'))", data=d).fit()
    aov = anova_lm(model, typ=2)

    df_resid = int(aov.loc["Residual", "df"])
    mse = float(aov.loc["Residual", "sum_sq"] / aov.loc["Residual", "df"]) if df_resid > 0 else float("nan")
    se_block_units = math.sqrt(mse) if mse == mse else float("nan")
    grand_mean = float(d[response].mean())
    cv_block_units = 100.0 * se_block_units / grand_mean if grand_mean != 0 and not np.isnan(se_block_units) else float("nan")

    counts = d.groupby(trt)[response].count().to_dict()
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        warnings.append("Design appears unbalanced; e.s.e./s.e.d./LSD use harmonic reps.")
        r = stats.hmean(np.array(list(counts.values()), dtype=float))
    else:
        r = float(unique_counts.pop())
    ese = math.sqrt(mse / r) if r and r > 0 and not np.isnan(mse) else float("nan")
    sed = math.sqrt(2 * mse / r) if r and r > 0 and not np.isnan(mse) else float("nan")
    tcrit = stats.t.ppf(1 - alpha/2, df_resid) if df_resid > 0 else float("nan")
    lsd = tcrit * sed if not any(np.isnan([tcrit, sed])) else float("nan")

    means = d.groupby(trt, observed=True)[response].mean().rename("Mean").to_frame()
    means.index.name = "Treatment"
    p_treatment = float(aov.loc[f"C(Q('{trt}'))", "PR(>F)"])

    letters_df = None
    if p_treatment < alpha and not np.isnan(lsd):
        letters_df = _fisher_lsd_letters(means["Mean"], lsd)
    else:
        warnings.append("Fisher’s protected LSD not calculated (Treatment not significant).")

    aov_out = aov.rename(columns={"df": "d.f.", "sum_sq": "s.s.", "mean_sq": "m.s.", "F": "v.r.", "PR(>F)": "F pr."})
    for c in ['d.f.', 's.s.', 'm.s.', 'v.r.', 'F pr.']:
        if c not in aov_out.columns: aov_out[c] = np.nan
    aov_out = aov_out[['d.f.', 's.s.', 'm.s.', 'v.r.', 'F pr.']]

    return AovResult(
        aov_table=aov_out, means=means, ese=ese, sed=sed, lsd=lsd, alpha=alpha,
        df_resid=df_resid, mse=mse, grand_mean=grand_mean, cv_block_units=cv_block_units,
        se_block_units=se_block_units, se_block=None, cv_block=None,
        p_treatment=p_treatment, letters=letters_df, warnings=warnings
    )

def _fisher_lsd_letters(means: pd.Series, lsd: float) -> pd.DataFrame:
    m = means.sort_values()  # ascending
    names, values = list(m.index), m.values
    letters = [""] * len(names)
    assigned = [False] * len(names)
    current_letter = ord("a")
    for i in range(len(names)):
        if assigned[i]: continue
        group = [i]
        for j in range(i+1, len(names)):
            if abs(values[j] - values[i]) <= lsd:
                group.append(j)
        ch = chr(current_letter)
        for idx in group:
            letters[idx] += ch; assigned[idx] = True
        current_letter += 1
    out = pd.DataFrame({"Mean": m.values, "Letters": letters}, index=names)
    out.index.name = "Treatment"
    return out

def _adjust_pvalues(pvals: np.ndarray, method: str) -> np.ndarray:
    m = len(pvals)
    if method == "none": return pvals
    if method == "bonferroni": return np.minimum(1.0, pvals * m)
    if method == "holm":
        order = np.argsort(pvals); adj = np.empty_like(pvals)
        for rank, idx in enumerate(order, 1): adj[idx] = (m - rank + 1) * pvals[idx]
        adj_sorted = np.maximum.accumulate(adj[order][::-1])[::-1]; adj[order] = np.minimum(adj_sorted, 1.0); return adj
    if method in ("fdr_bh", "bh"):
        order = np.argsort(pvals); ranked = np.arange(1, m+1)
        adj_vals = pvals[order] * m / ranked; adj_vals = np.minimum.accumulate(adj_vals[::-1])[::-1]
        out = np.empty_like(pvals); out[order] = np.minimum(adj_vals, 1.0); return out
    return pvals

def _format_p(p: float, alpha: float) -> str:
    if np.isnan(p): return ""
    if p >= alpha: return "NS"
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def _build_matrix_table(results: dict[str, AovResult]) -> pd.DataFrame:
    first = next(iter(results.values()))
    trt_index = list(first.means.index)
    core = {}
    for col, res in results.items():
        m = res.means.reindex(trt_index)
        s_vals = m["Mean"].round(1).astype(str)
        if res.letters is not None:
            letters = res.letters.reindex(trt_index)["Letters"].fillna("")
            s_vals = s_vals + letters.apply(lambda x: (" " + x) if x else "")
        core[col] = s_vals
    table = pd.DataFrame(core, index=pd.Index(trt_index, name="Treatment"))
    p_row  = pd.Series({c: _format_p(res.p_treatment, res.alpha) for c, res in results.items()}, name="P")
    lsd_row= pd.Series({c: (f"{res.lsd:.3f}" if res.p_treatment < res.alpha and not np.isnan(res.lsd) else "-") for c, res in results.items()}, name="LSD")
    df_row = pd.Series({c: str(res.df_resid) for c, res in results.items()}, name="d.f.")
    cv_row = pd.Series({c: (f"{res.cv_block_units:.1f}" if not np.isnan(res.cv_block_units) else "") for c, res in results.items()}, name="%c.v.")
    out = pd.concat([table, pd.DataFrame([p_row, lsd_row, df_row, cv_row])])
    out.index = [f"[{i+1}] {idx}" if i < len(trt_index) else idx for i, idx in enumerate(out.index)]
    return out

def _build_export_workbook(results: dict[str, AovResult]) -> bytes:
    with pd.ExcelWriter(io.BytesIO(), engine="xlsxwriter") as writer:
        for rname, res in results.items():
            res.aov_table.to_excel(writer, sheet_name=f"{rname}_ANOVA")
            res.means.to_excel(writer, sheet_name=f"{rname}_Means")
            pd.DataFrame({
                "alpha":[res.alpha], "df_resid":[res.df_resid], "MSE":[res.mse],
                "e.s.e.":[res.ese], "s.e.d.":[res.sed], "l.s.d.":[res.lsd],
                "grand_mean":[res.grand_mean], "se_residual":[res.se_block_units],
                "cv%_residual":[res.cv_block_units], "p_treatment":[res.p_treatment],
            }).to_excel(writer, sheet_name=f"{rname}_SEs", index=False)
            if res.letters is not None:
                res.letters.to_excel(writer, sheet_name=f"{rname}_LSD_letters")
        if len(results) > 1:
            _build_matrix_table(results).to_excel(writer, sheet_name="Matrix")
        writer.book.close()
        return writer.path.getvalue()

# ---------- File input ----------
uploaded = st.file_uploader("Drag & drop Excel file (.xlsx/.xls)", type=["xlsx","xls"])

if not uploaded:
    st.info("Upload an Excel file to begin.")
else:
    if input_mode == "Single sheet (wide)":
        raw = pd.read_excel(uploaded, sheet_name=None)
        sheet = st.selectbox("Sheet", list(raw.keys()), index=0)
        df = _clean_colnames(raw[sheet])
        st.write(":clipboard: **Preview**"); st.dataframe(df.head(20), use_container_width=True)
        plot_col, block_col, trt_col, numeric_cols = guess_key_columns(df)
        c1,c2,c3 = st.columns(3)
        with c1: plot_col = st.selectbox("Plot column", df.columns, index=list(df.columns).index(plot_col) if plot_col in df.columns else 0)
        with c2: block_col = st.selectbox("Block column", df.columns, index=list(df.columns).index(block_col) if block_col in df.columns else 0)
        with c3: trt_col  = st.selectbox("Treatment column", df.columns, index=list(df.columns).index(trt_col)  if trt_col  in df.columns else 0)
        response_cols = st.multiselect("Response variables (one or more)", [c for c in df.columns if c not in {plot_col, block_col, trt_col}], default=numeric_cols)

        if response_cols:
            per_resp: dict[str, AovResult] = {}
            for rcol in response_cols:
                per_resp[rcol] = randomized_block_anova(df, rcol, trt_col, block_col, alpha=alpha)
            if len(per_resp) > 1 and adj_method != "none":
                labels, pvals = zip(*[(k, v.p_treatment) for k, v in per_resp.items()])
                adj = _adjust_pvalues(np.array(pvals, float), method=adj_method)
                st.subheader("Adjusted p-values across responses")
                st.dataframe(pd.DataFrame({"Response":labels, "p (Treatment)":pvals, f"p_adj [{adj_method}]":adj}).style.format({"p (Treatment)":"{:.3g}", f"p_adj [{adj_method}]":"{:.3g}"}), use_container_width=True)
            st.subheader("Matrix (treatments × selected responses)")
            st.dataframe(_build_matrix_table(per_resp), use_container_width=True)
            st.subheader("Export"); 
            if st.button("Build Excel output (.xlsx)"):
                st.download_button("Download Excel results", data=_build_export_workbook(per_resp), file_name="genstat_style_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    else:  # Multi-sheet (each sheet = date)
        xls = pd.ExcelFile(uploaded)
        all_data = []
        for sheet in xls.sheet_names:
            try:
                preview = pd.read_excel(xls, sheet_name=sheet, nrows=20)
                header_row = None
                for i, row in preview.iterrows():
                    vals = [str(v).lower() for v in row.values if pd.notna(v)]
                    if any("block" in v for v in vals) and any("treat" in v for v in vals):
                        header_row = i; break
                if header_row is None: continue
                df = pd.read_excel(xls, sheet_name=sheet, skiprows=header_row)
                df.columns = df.iloc[0]; df = df.drop(df.index[0]).dropna(axis=1, how="all")
                df.columns = [str(c).strip() for c in df.columns]
                col_map = {c: re.sub(r"\\W+","",c).lower() for c in df.columns}
                block_col = next((o for o,n in col_map.items() if "block" in n), None)
                plot_col  = next((o for o,n in col_map.items() if "plot"  in n), None)
                treat_col = next((o for o,n in col_map.items() if "treat" in n or "trt" in n), None)
                if not (block_col and treat_col): continue
                treat_idx = df.columns.get_loc(treat_col)
                assess_list = df.columns[treat_idx+1:].tolist()
                id_vars = [block_col, treat_col] + ([plot_col] if plot_col else [])
                dfl = df.melt(id_vars=id_vars, value_vars=assess_list, var_name="Assessment", value_name="Value")
                dfl = dfl.rename(columns={block_col:"Block", treat_col:"Treatment"})
                dfl["DateLabel"] = sheet
                all_data.append(dfl)
            except Exception:
                continue
        if not all_data:
            st.error("No valid tables found in this file.")
        else:
            data = pd.concat(all_data, ignore_index=True)
            assessments = sorted(data["Assessment"].dropna().unique(), key=lambda x: str(x))
            assess_choice = st.selectbox("Assessment variable", assessments, index=0)
            df_sub = data[data["Assessment"] == assess_choice].copy()
            df_sub["Value"] = pd.to_numeric(df_sub["Value"], errors="coerce")
            df_sub = df_sub.dropna(subset=["Value"])
            dates = list(dict.fromkeys(df_sub["DateLabel"].tolist()))

            results = {}
            for date_label in dates:
                d = df_sub[df_sub["DateLabel"] == date_label].copy()
                d = d.rename(columns={"Value": assess_choice})
                if not d.empty:
                    results[date_label] = randomized_block_anova(d, assess_choice, "Treatment", "Block", alpha=alpha)
            if results:
                st.subheader("Treatment × date table (with letters + footer)")
                st.dataframe(_build_matrix_table(results), use_container_width=True)
                if adj_method != "none" and len(results) > 1:
                    labels, pvals = zip(*[(k, v.p_treatment) for k, v in results.items()])
                    adj = _adjust_pvalues(np.array(pvals, float), method=adj_method)
                    st.caption("Adjusted p-values across dates:")
                    st.dataframe(pd.DataFrame({"Date":labels, "p (Treatment)":pvals, f"p_adj [{adj_method}]":adj}).style.format({"p (Treatment)":"{:.3g}", f"p_adj [{adj_method}]":"{:.3g}"}), use_container_width=True)
                st.subheader("Export")
                if st.button("Build Excel output (.xlsx)"):
                    st.download_button("Download Excel results", data=_build_export_workbook(results), file_name="genstat_style_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
