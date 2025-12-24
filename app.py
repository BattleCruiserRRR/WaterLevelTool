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

st.write("上传 Excel（.xls/.xlsx）后，立即输出阈值统计、时间序列图、ECDF 图，并可下载结果。")

uploaded = st.file_uploader("上传 Excel 文件", type=["xls", "xlsx"])

colA, colB, colC = st.columns(3)
with colA:
    percentile = st.slider("Percentile", 1, 99, 90)
with colB:
    force_year = st.checkbox("强制年份（用于日期缺失/修正）", value=False)
with colC:
    correct_year = st.number_input("年份", min_value=1900, max_value=2100, value=2024, step=1)

run = st.button("开始计算")

def save_upload_to_tmp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(uploaded_file.getbuffer())
    f.close()
    return f.name

if run:
    if uploaded is None:
        st.error("请先上传一个 Excel 文件。")
        st.stop()

    tmp_path = save_upload_to_tmp(uploaded)

    try:
        df_raw, meta = load_excel_with_auto_header(tmp_path, head_show=5)

        st.write(f"初步清理完成，表格的大小为: {meta['shape']}")
        st.write(f"识别到的表头行(0-index): {meta['header_row']}")
        st.dataframe(meta["preview"], use_container_width=True)

        if force_year:
            df_long, meta = wide_to_long_4cols(df_raw, correct_year=correct_year, head_show=5)
        else:
            df_long, meta = wide_to_long_4cols(df_raw, head_show=5)


        st.write(f"深度清洗完成，表格的大小为: {meta['shape']}")
        st.dataframe(meta["preview"], use_container_width=True)

        res = lower_set_max_and_counts_strict_gt(df_long, percentile)

        st.subheader("统计结果")
        st.json(res)

        # 图表
        st.subheader("时间序列图")
        fig_ts = plot_water_level_timeseries(df_long)
        st.pyplot(fig_ts, clear_figure=True)

        st.subheader("ECDF + 阈值线")

        # 你新版返回的是 threshold_t
        t = res["threshold_t"]

        fig_ecdf = plot_ecdf_with_threshold_counts(df_long, threshold_t=t)
        st.pyplot(fig_ecdf, clear_figure=True)

        # 导出结果：csv（字段对齐新版返回）
        st.subheader("下载")

        out_row = {
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
        }

        out_df = pd.DataFrame([out_row])

        st.download_button(
            "下载统计结果（CSV）",
            data=out_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="water_level_result.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.exception(e)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass