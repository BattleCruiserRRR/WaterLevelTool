# app.py
import os
import tempfile
import pandas as pd
import streamlit as st

from utils import (
    load_excel_with_auto_header,
    wide_to_long_4cols,
    lower_set_max_and_counts_strict_gt,
    plot_water_level_timeseries,
    plot_ecdf_with_threshold_counts,
)

st.set_page_config(page_title="Water Level Offline Tool", layout="wide")
st.title("水位分析")

st.markdown(
    """
请上传 Excel（.xls/.xlsx），支持两种数据格式：

1) 宽表（原始水闸表格式）
- 每个月为一组：日期 / 水位 / 闸外 / 备注（4列为一组，多个组横向拼接）
- 这类格式会使用 wide_to_long_4cols 转换成长表

2) 长表（推荐）
- 必须有一列时间（日期/时间）
- 必须有一列要分析的数值（水位列，列名可不同，顺序不重要）
- 上传后需要选择“时间列”和“分析列”，再点击“开始计算”
"""
)

# -------------------------
# session_state 初始化
# -------------------------
def ss_init(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default

ss_init("computed", False)
ss_init("df_long_year", None)
ss_init("result_df", None)
ss_init("per_year_res", None)
ss_init("meta_read", None)
ss_init("meta_long", None)
ss_init("input_signature", None)

# 长表列选择固定
ss_init("date_col_selected", None)
ss_init("value_col_selected", None)

uploaded = st.file_uploader("上传 Excel 文件", type=["xls", "xlsx"])

colA, colB, colC, colD = st.columns(4)
with colA:
    percentile = st.slider("Percentile（严格 >t 的覆盖率要求）", 1, 99, 95)
with colB:
    data_mode = st.selectbox("数据格式", ["宽表（每月4列一组）", "长表（一列时间 + 一列数值）"])
with colC:
    force_year = st.checkbox("强制年份（仅用于宽表日期缺失/修正）", value=False)
with colD:
    correct_year = st.number_input("年份", min_value=1900, max_value=2100, value=2024, step=1)


def save_upload_to_tmp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(uploaded_file.getbuffer())
    f.close()
    return f.name


def build_long_df_from_selected_cols(df_raw, date_col, value_col):
    d = df_raw[[date_col, value_col]].copy()
    d.columns = ["date", "water_level"]
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["water_level"] = pd.to_numeric(d["water_level"], errors="coerce")
    d = d.dropna(subset=["date", "water_level"]).sort_values("date").reset_index(drop=True)
    return d


def run_per_year_analysis(df_long, percentile_value):
    df_long = df_long.copy()
    df_long["year"] = pd.to_datetime(df_long["date"], errors="coerce").dt.year
    df_long = df_long.dropna(subset=["year"])
    df_long["year"] = df_long["year"].astype(int)

    years = sorted(df_long["year"].unique().tolist())
    rows = []
    per_year_res = {}

    for y in years:
        sub = df_long[df_long["year"] == y].copy()
        try:
            res = lower_set_max_and_counts_strict_gt(sub, percentile_value)
            per_year_res[y] = res

            rows.append({
                "year": y,
                "percentile": res["percentile"],
                "threshold_t": res["threshold_t"],
                "n_total": res["n_total"],
                "count_lt": res["count_lt"],
                "count_eq": res["count_eq"],
                "count_gt": res["count_gt"],
                "gt_ratio": (res["count_gt"] / res["n_total"]) if res["n_total"] else None,
                "min": res["eda_summary"]["min"],
                "max": res["eda_summary"]["max"],
                "mean": res["eda_summary"]["mean"],
                "median": res["eda_summary"]["median"],
                "std": res["eda_summary"]["std"],
            })
        except Exception as e:
            rows.append({
                "year": y,
                "percentile": percentile_value,
                "threshold_t": None,
                "n_total": int(sub["water_level"].dropna().shape[0]),
                "count_lt": None,
                "count_eq": None,
                "count_gt": None,
                "gt_ratio": None,
                "min": float(pd.to_numeric(sub["water_level"], errors="coerce").min()) if sub.shape[0] else None,
                "max": float(pd.to_numeric(sub["water_level"], errors="coerce").max()) if sub.shape[0] else None,
                "mean": float(pd.to_numeric(sub["water_level"], errors="coerce").mean()) if sub.shape[0] else None,
                "median": float(pd.to_numeric(sub["water_level"], errors="coerce").median()) if sub.shape[0] else None,
                "std": float(pd.to_numeric(sub["water_level"], errors="coerce").std()) if sub.shape[0] else None,
                "error": str(e),
            })

    result_df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return df_long, result_df, per_year_res


def current_input_signature():
    file_sig = (uploaded.name, uploaded.size) if uploaded is not None else None
    return (
        file_sig,
        percentile,
        data_mode,
        force_year,
        correct_year,
        st.session_state.get("date_col_selected"),
        st.session_state.get("value_col_selected"),
    )


# -------------------------
# 清理缓存：当输入发生变化时
# -------------------------
sig = current_input_signature()
if st.session_state["input_signature"] is None:
    st.session_state["input_signature"] = sig
else:
    if sig != st.session_state["input_signature"]:
        st.session_state["computed"] = False
        st.session_state["df_long_year"] = None
        st.session_state["result_df"] = None
        st.session_state["per_year_res"] = None
        st.session_state["meta_read"] = None
        st.session_state["meta_long"] = None
        st.session_state["input_signature"] = sig


# -------------------------
# 长表：固定显示列选择（并把“开始计算”放在选择之后）
# -------------------------
df_raw_for_select = None
meta_for_select = None
can_run = False  # 是否允许显示并点击开始计算

if uploaded is not None and data_mode.startswith("长表"):
    tmp_path = save_upload_to_tmp(uploaded)
    try:
        df_raw_for_select, meta_for_select = load_excel_with_auto_header(tmp_path, head_show=5, crop=False)

        st.subheader("长表列选择")
        st.write(f"识别到的表头行(0-index): {meta_for_select['header_row']}")
        st.write(f"表格大小: {meta_for_select['shape']}")
        st.dataframe(meta_for_select["preview"], use_container_width=True)

        cols = list(df_raw_for_select.columns)
        if len(cols) < 2:
            st.error("当前表格列数不足，长表至少需要两列：时间列 + 数值列。")
            st.stop()

        guess_date_cols = [c for c in cols if any(k in str(c) for k in ["日期", "时间", "date", "Date", "time", "Time"])]
        default_date = guess_date_cols[0] if guess_date_cols else cols[0]

        if st.session_state["date_col_selected"] is None or st.session_state["date_col_selected"] not in cols:
            st.session_state["date_col_selected"] = default_date

        date_col = st.selectbox(
            "选择时间列",
            options=cols,
            index=cols.index(st.session_state["date_col_selected"]),
            key="date_col_selected",
        )

        # 候选数值列
        candidate_value_cols = []
        for c in cols:
            if c == date_col:
                continue
            tmp_num = pd.to_numeric(df_raw_for_select[c], errors="coerce")
            if tmp_num.notna().sum() > 0:
                candidate_value_cols.append(c)

        if not candidate_value_cols:
            st.error("未找到可解析为数值的分析列。请检查数据是否为数值，或换一列。")
            st.stop()

        guess_val_cols = [c for c in candidate_value_cols if any(k in str(c) for k in ["水位", "water", "level", "WL", "Water"])]
        default_val = guess_val_cols[0] if guess_val_cols else candidate_value_cols[0]

        if st.session_state["value_col_selected"] is None or st.session_state["value_col_selected"] not in candidate_value_cols:
            st.session_state["value_col_selected"] = default_val

        value_col = st.selectbox(
            "选择要分析的数值列",
            options=candidate_value_cols,
            index=candidate_value_cols.index(st.session_state["value_col_selected"]),
            key="value_col_selected",
        )

        # 选择完成后才允许 run
        can_run = True

        # 把开始计算按钮放在这里（紧跟在两个选择框之后）
        run = st.button("开始计算", disabled=not can_run)

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

else:
    # 宽表模式：按钮放在转换说明之后（不依赖长表列选择）
    st.subheader("宽表模式")
    run = st.button("开始计算", disabled=(uploaded is None))


# -------------------------
# 点击“开始计算”：做重计算并写入 session_state
# -------------------------
if run:
    if uploaded is None:
        st.error("请先上传一个 Excel 文件。")
        st.stop()

    tmp_path = save_upload_to_tmp(uploaded)

    try:
        # 读取
        if data_mode.startswith("宽表"):
            df_raw, meta_read = load_excel_with_auto_header(tmp_path, head_show=5, crop=True)
        else:
            df_raw, meta_read = load_excel_with_auto_header(tmp_path, head_show=5, crop=False)

        st.session_state["meta_read"] = meta_read

        # 构造 df_long
        if data_mode.startswith("宽表"):
            if force_year:
                df_long, meta_long = wide_to_long_4cols(df_raw, correct_year=correct_year, head_show=5)
            else:
                df_long, meta_long = wide_to_long_4cols(df_raw, head_show=5)

            st.session_state["meta_long"] = meta_long

        else:
            date_col = st.session_state.get("date_col_selected")
            value_col = st.session_state.get("value_col_selected")
            if date_col is None or value_col is None:
                st.error("请先选择时间列与数值列。")
                st.stop()

            df_long = build_long_df_from_selected_cols(df_raw, date_col, value_col)
            st.session_state["meta_long"] = {
                "shape": df_long.shape,
                "preview": df_long.head(5)
            }

        if "date" not in df_long.columns or "water_level" not in df_long.columns:
            st.error("内部格式错误：df_long 必须包含 date 与 water_level 两列。")
            st.stop()

        # 按年份分析并缓存
        df_long_year, result_df, per_year_res = run_per_year_analysis(df_long, percentile)

        st.session_state["df_long_year"] = df_long_year
        st.session_state["result_df"] = result_df
        st.session_state["per_year_res"] = per_year_res
        st.session_state["computed"] = True

    except Exception as e:
        st.exception(e)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# -------------------------
# 展示区域：computed=True 就持续显示
# -------------------------
if st.session_state["computed"]:
    meta_read = st.session_state["meta_read"]
    meta_long = st.session_state["meta_long"]
    df_long_year = st.session_state["df_long_year"]
    result_df = st.session_state["result_df"]
    per_year_res = st.session_state["per_year_res"]

    st.subheader("读取预览")
    if meta_read is not None:
        st.write(f"识别到的表头行(0-index): {meta_read['header_row']}")
        st.write(f"表格大小: {meta_read['shape']}")
        st.dataframe(meta_read["preview"], use_container_width=True)

    st.subheader("清洗/转换后预览")
    if meta_long is not None:
        st.write(f"表格大小: {meta_long['shape']}")
        st.dataframe(meta_long["preview"], use_container_width=True)

    st.subheader("按年份统计（每个年份单独计算阈值）")
    st.dataframe(result_df, use_container_width=True)

    available_years = result_df["year"].tolist()
    if available_years:
        show_year = st.selectbox(
            "选择年份查看图表",
            options=available_years,
            index=len(available_years) - 1,
            key="show_year"
        )

        sub_show = df_long_year[df_long_year["year"] == show_year].copy()

        # 这里输出：读取了多少没有被清理的数据（即有效样本数）
        # 有效定义：date 与 water_level 都不缺失（build_long / wide_to_long 已保证）
        valid_n = int(sub_show.dropna(subset=["date", "water_level"]).shape[0])
        st.write(f"{show_year} 年用于分析的有效数据条数: {valid_n}")

        row_show = result_df[result_df["year"] == show_year].iloc[0]
        t = row_show["threshold_t"]

        st.subheader("时间序列图")
        fig_ts = plot_water_level_timeseries(
            sub_show,
            date_col="date",
            level_col="water_level",
            title=f"Water Level Time Series - {show_year}"
        )
        st.pyplot(fig_ts, clear_figure=True)

        st.subheader("ECDF + 阈值线")
        if t is None or (isinstance(t, float) and pd.isna(t)):
            st.warning("该年份无法计算 strict 阈值（可能数据量不足或 percentile 过高）。")
        else:
            fig_ecdf = plot_ecdf_with_threshold_counts(
                sub_show,
                threshold_t=float(t),
                level_col="water_level",
                title=f"ECDF - {show_year} (t={float(t):.3f})"
            )
            st.pyplot(fig_ecdf, clear_figure=True)

            st.write("该年份阈值结果：")
            st.json(per_year_res.get(show_year, {}))

    st.subheader("下载")
    st.download_button(
        "下载按年统计结果（CSV）",
        data=result_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="water_level_result_by_year.csv",
        mime="text/csv",
    )