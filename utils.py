# utils.py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_excel_with_auto_header(path, header_keywords=None, max_scan_rows=3, head_show=5, crop=True):
    """
    自动识别 Excel 表头行，并正确读取数据。
    固定裁剪规则：只保留前 31 行、前 48 列（硬性业务要求）。
    """
    if header_keywords is None:
        header_keywords = ["日期", "水位", "闸外", "备注"]

    # 选择引擎（避免打包/离线时 .xls/.xlsx 读取不稳定）
    lower = str(path).lower()
    engine = "xlrd" if lower.endswith(".xls") else "openpyxl" if lower.endswith(".xlsx") else None

    # 1) 先读前几行用于找表头（更快）
    df_raw = pd.read_excel(path, header=None, nrows=max_scan_rows, engine=engine)

    # 2) 查找表头行
    header_row = None
    for i in range(len(df_raw)):
        row = df_raw.iloc[i].astype(str).fillna("")
        keyword_hits = sum(any(k in cell for k in header_keywords) for cell in row)
        if keyword_hits >= 2:
            header_row = i
            break

    if header_row is None:
        raise ValueError("未在前几行中识别到表头行")

    # 3) 用识别到的表头行重新读取
    df = pd.read_excel(path, header=header_row, engine=engine)
    if crop:
        df = df.iloc[:31, :48]

    # 4) 不在 utils 里 print/display，把需要展示的信息返回给 UI
    meta = {
        "header_row": header_row,
        "shape": df.shape,
        "preview": df.head(head_show)
    }

    return df, meta


def wide_to_long_4cols(df, cols_per_group=4, correct_year=None, head_show=5):
    """
    将宽表（每个月 4 列一组：日期/水位/闸外/备注）转换为长表：
    输出列：date, water_level, gate_out, remark

    返回
    -------
    long_df : pd.DataFrame
    meta : dict
        包含 shape、preview（用于 UI 展示，替代 print/display）
    """
    df = df.copy()

    # 1) 确保列数是 4 的倍数（否则直接截断）
    ncols = df.shape[1]
    if ncols % cols_per_group != 0:
        df = df.iloc[:, : (ncols // cols_per_group) * cols_per_group]

    # 2) 每 4 列切成一个 block
    blocks = []

    for i in range(0, df.shape[1], cols_per_group):
        block = df.iloc[:, i:i + cols_per_group].copy()
        block.columns = ["date", "water_level", "gate_out", "remark"]

        block["date"] = pd.to_datetime(block["date"], errors="coerce")
        block["water_level"] = pd.to_numeric(block["water_level"], errors="coerce")
        block["gate_out"] = pd.to_numeric(block["gate_out"], errors="coerce")

        # 丢掉全空行
        block = block.dropna(how="all")

        # 日期向前填充（你的原逻辑）
        block["date"] = block["date"].ffill()
        block = block.dropna(subset=["date"])

        blocks.append(block)

    long_df = pd.concat(blocks, ignore_index=True)

    # 3) 再次清洗与排序（保持你原逻辑）
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df["water_level"] = pd.to_numeric(long_df["water_level"], errors="coerce")
    long_df["gate_out"] = pd.to_numeric(long_df["gate_out"], errors="coerce")

    long_df = (
        long_df
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # 4) 修正年份（原逻辑，保持）
    if correct_year:
        long_df["date"] = long_df["date"].apply(
            lambda d: pd.Timestamp(correct_year, d.month, d.day)
            if pd.notna(d) else pd.NaT
        )

    # 5) 返回 meta（替代 print + display）
    meta = {
        "shape": long_df.shape,
        "preview": long_df.head(head_show)
    }

    return long_df, meta


def plot_water_level_timeseries(
    df_long,
    date_col="date",
    level_col="water_level",
    title="Water Level Time Series",
    figsize=(12, 5)
):
    """
    时间序列图：返回 matplotlib Figure（不要 plt.show）
    """
    fig, ax = plt.subplots(figsize=figsize)
    s = df_long.sort_values(date_col)
    ax.plot(s[date_col], s[level_col], linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(date_col)
    ax.set_ylabel(level_col)
    fig.autofmt_xdate()
    return fig


def plot_ecdf_with_threshold_counts(
    df,
    threshold_t,
    level_col="water_level",
    title="ECDF of Water Level with Threshold",
    figsize=(8, 5)
):
    s = pd.to_numeric(df[level_col], errors="coerce").dropna().to_numpy(dtype=float)
    if s.size == 0:
        raise ValueError("没有有效的 water_level 数据")

    x = np.sort(s)
    n = x.size
    y = np.arange(1, n + 1) / n  # ECDF: F(x)=P(X<=x)

    n_lt = int(np.sum(s < threshold_t))
    n_eq = int(np.sum(s == threshold_t))
    n_gt = int(np.sum(s > threshold_t))
    if n_lt + n_eq + n_gt != n:
        raise RuntimeError("计数不一致：n_lt + n_eq + n_gt != n")

    F_t = (n_lt + n_eq) / n

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(x, y, where="post")

    ax.axvline(
        threshold_t,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"t = {threshold_t:.3f}"
    )

    ax.axhline(
        F_t,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label=f"F(t)=P(X<=t)={F_t:.3%}"
    )

    text = (
        f"Total: {n}\n"
        f"< t : {n_lt} ({n_lt/n:.2%})\n"
        f"= t : {n_eq} ({n_eq/n:.2%})\n"
        f"> t : {n_gt} ({n_gt/n:.2%})"
    )

    ax.text(
        0.02, 0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top"
    )

    ax.set_xlabel("Water Level")
    ax.set_ylabel("ECDF  F(x) = P(X ≤ x)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig


def lower_set_max_and_counts_strict_gt(
    df,
    percentile,
    date_col="date",
    level_col="water_level"
):

    if not (0 < percentile < 100):
        raise ValueError("percentile 必须在 (0, 100) 之间")

    df = df.copy()

    # 类型保证
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[level_col] = pd.to_numeric(df[level_col], errors="coerce")

    # 只保留有效 water_level
    df = df.dropna(subset=[level_col])
    print("找到{}条没有水位缺失的数据".format(df.shape[0]))

    if len(df) == 0:
        raise ValueError("没有有效 water_level 数据")

    s = df[level_col].to_numpy(dtype=float)
    n_total = int(len(s))

    # EDA
    eda_summary = {
        "count": n_total,
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "std": float(np.std(s, ddof=1)) if n_total > 1 else 0.0,
    }

    target_ratio = percentile / 100.0
    k = int(math.ceil(n_total * target_ratio))  # 需要至少 k 个点满足 >t

    # 严格 > 的极限：无法让所有点都严格大于 t（t 取自样本值）
    if k >= n_total:
        raise ValueError(
            f"n_total={n_total} 时，要求严格大于覆盖率 >= {percentile}% 等价于需要 >t 的数量 >= {k}。"
            "由于 t 必须取自样本值，无法保证所有点都严格大于 t。"
            "请降低 percentile，或把条件改为 >=t。"
        )

    # 统计 unique value -> freq（按值降序）
    vc_desc = pd.Series(s).value_counts().sort_index(ascending=False)

    cum = 0
    critical_value = None
    for value, freq in vc_desc.items():
        cum += int(freq)
        if cum >= k:
            critical_value = float(value)
            break

    if critical_value is None:
        raise RuntimeError("无法找到满足条件的临界值（理论上不应发生）")

    # 为满足严格 >t 且 t 取自样本值，我们需要 t < critical_value
    # 且希望 t 尽可能大 => 取 “小于 critical_value 的最大样本值”
    uniq_asc = vc_desc.sort_index(ascending=True).index.to_list()  # unique values asc

    pos = uniq_asc.index(critical_value)
    if pos == 0:
        # critical_value 已是最小值，则要让 >t 覆盖 k，就需要 t < min
        # 但 t 要取自样本值做不到
        raise ValueError(
            "无法满足严格 >t 的要求：需要 t 小于最小水位值，但 t 必须取自样本值。"
        )

    t = float(uniq_asc[pos - 1])

    # 统一口径计算 < = >（确保加总一致）
    n_lt = int(np.sum(s < t))
    n_eq = int(np.sum(s == t))
    n_gt = int(np.sum(s > t))

    if n_lt + n_eq + n_gt != n_total:
        raise RuntimeError("计数口径不一致：count_lt + count_eq + count_gt != n_total")

    gt_ratio = n_gt / n_total
    if gt_ratio + 1e-12 < target_ratio:
        raise RuntimeError(
            f"Guarantee violated: actual_gt_ratio={gt_ratio:.6f} < target={target_ratio:.6f}"
        )

    plot_ecdf_with_threshold_counts(
        df,
        threshold_t=t
    )

    return {
        "eda_summary": eda_summary,
        "percentile": float(percentile),
        "n_total": n_total,
        "threshold_t": t,
        "count_lt": n_lt,
        "count_eq": n_eq,
        "count_gt": n_gt
    }