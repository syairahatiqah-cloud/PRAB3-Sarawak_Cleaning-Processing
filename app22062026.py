# app22062026.py
# ============================================================
# Streamlit App: FSM Imputation for WL/SF (Upload File Only)
# - Upload Excel/CSV
# - Choose datetime + value column
# - TAB 1: Raw time series (Plotly + HTML + PNG via Matplotlib) + SHOW PNG
# - TAB 2: Missing data summary (Monthly / Yearly) + downloads + SHOW PNG (with % labels)
# - TAB 3: FSM Imputation + CSV download
#          - Time series plot: Original + FSM IMPUTED SEGMENTS ONLY (red dashed)
#            + SHOW PNG
#          - Monthly seasonality: Original vs FULL infilled series (blue vs red)
#
# NOTE: Plotly PNG export (fig.to_image) is disabled because Streamlit Cloud
# may not have Chrome available for Kaleido. PNGs are generated via Matplotlib.
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# ============================================================
# Data-cleaning helpers
# ============================================================
DEFAULT_MISSING_TEXT = {
    "", "NA", "N/A", "NAN", "NULL", "NONE", "MISSING",
    "NO DATA", "NODATA", "NOT AVAILABLE", "-", "--"
}


def parse_datetime_column(values: pd.Series, date_order: str) -> pd.Series:
    """Parse uploaded timestamps using an explicit, user-selected date order."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")

    order = str(date_order).split()[0].upper()
    dayfirst = order in {"DMY", "AUTO"}
    yearfirst = order == "YMD"

    # format="mixed" prevents newer pandas versions from forcing every row to
    # match the format inferred from only the first timestamp.
    try:
        return pd.to_datetime(
            values,
            errors="coerce",
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            format="mixed"
        )
    except (TypeError, ValueError):
        # Compatibility fallback for older pandas versions.
        return pd.to_datetime(
            values,
            errors="coerce",
            dayfirst=dayfirst,
            yearfirst=yearfirst
        )


def clean_numeric_series(values: pd.Series, numeric_markers) -> tuple[pd.Series, pd.Series]:
    """
    Convert common text/numeric missing markers to real NaN values.

    Returns
    -------
    cleaned : numeric Series
    marker_mask : True only where an explicit missing marker was detected
    """
    text = values.astype("string").str.strip()
    upper = text.str.upper()

    text_marker_mask = values.isna() | upper.isna() | upper.isin(DEFAULT_MISSING_TEXT)
    numeric = pd.to_numeric(text.mask(text_marker_mask), errors="coerce")

    numeric_marker_mask = pd.Series(False, index=values.index)
    for marker in numeric_markers:
        numeric_marker_mask |= np.isclose(
            numeric.to_numpy(dtype=float, na_value=np.nan),
            float(marker),
            equal_nan=False
        )

    marker_mask = text_marker_mask.fillna(True) | numeric_marker_mask
    cleaned = numeric.mask(marker_mask)
    return cleaned.astype(float), marker_mask.astype(bool)


def parse_numeric_markers(marker_text: str):
    """Read comma-separated numeric missing markers entered in the app."""
    markers = []
    for item in str(marker_text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            markers.append(float(item))
        except ValueError:
            pass
    return markers


def prepare_plot_series(dt: pd.Series, values: pd.Series, display_mode: str):
    """Prepare an optional aggregated/smoothed view without changing source data."""
    plot_df = pd.DataFrame({"DateTime": dt, "Value": values}).dropna(subset=["DateTime"])
    plot_df = plot_df.sort_values("DateTime").set_index("DateTime")

    if display_mode == "Daily mean":
        out = plot_df["Value"].resample("D").mean()
    elif display_mode == "7-day rolling mean":
        daily = plot_df["Value"].resample("D").mean()
        out = daily.rolling(7, min_periods=1).mean()
    elif display_mode == "Monthly mean":
        out = plot_df["Value"].resample("MS").mean()
    else:
        out = plot_df["Value"]

    return out.index, out


# ============================================================
# Helper: infer y-axis label from selected column name
# ============================================================
def infer_yaxis_label(col_name: str) -> str:
    s = str(col_name).lower()

    wl_keys = ["wl", "water level", "waterlevel", "stage", "river level", "level (m)"]
    sf_keys = ["sf", "streamflow", "stream flow", "discharge", "flow", "m3/s", "m³/s"]

    if any(k in s for k in wl_keys):
        return "Water Level (m)"
    if any(k in s for k in sf_keys):
        return "Streamflow (m³/s)"
    return "Value"


# ============================================================
# Helper: PNG bytes via Matplotlib (Cloud-safe)
# ============================================================
def line_png_bytes(
    x, y_list, labels, title, xlab, ylab,
    figsize=(12, 4), rotate_xticks=45,
    colors=None, linestyles=None, markers=None
):
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = [None] * len(y_list)
    if linestyles is None:
        linestyles = ["-"] * len(y_list)
    if markers is None:
        markers = [None] * len(y_list)

    for y, lab, c, ls, mk in zip(y_list, labels, colors, linestyles, markers):
        ax.plot(x, y, label=lab, color=c, linestyle=ls, marker=mk, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    ax.legend()

    for tick in ax.get_xticklabels():
        tick.set_rotation(rotate_xticks)
        tick.set_ha("right")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def bar_png_bytes(
    x, y, title, xlab, ylab,
    figsize=None,
    rotate_xticks=45,
    show_value_labels=False,
    max_xticks=24,
    label_min_value=0.05
):
    """
    Create a cleaner bar PNG for monthly/yearly missing summaries.

    Fixes overcrowding when many Year-Month bars exist by:
    1) increasing figure width automatically,
    2) showing only a limited number of x-axis tick labels,
    3) labelling only non-zero / meaningful missing percentages.
    """
    x = list(x)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if figsize is None:
        # Dynamic width: enough space for many months, capped to avoid huge PNGs.
        width = min(max(12, n * 0.18), 28)
        figsize = (width, 4.8)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(n), y)

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, min(110, max(105, np.nanmax(y) + 8 if len(y) else 105)))

    # ✅ Add labels only for non-zero / meaningful Missing % values.
    # This avoids hundreds of overlapping "0.0%" labels.
    if show_value_labels:
        for b, h in zip(bars, y):
            if np.isnan(h) or h < label_min_value:
                continue
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 1,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if n > 36 else 0
            )

    # ✅ Reduce crowded Year-Month tick labels.
    if n == 0:
        tick_idx = []
    elif n <= max_xticks:
        tick_idx = list(range(n))
    else:
        step = int(np.ceil(n / max_xticks))
        tick_idx = list(range(0, n, step))
        if (n - 1) not in tick_idx:
            tick_idx.append(n - 1)

    ax.set_xticks(tick_idx)
    ax.set_xticklabels([x[i] for i in tick_idx], rotation=rotate_xticks, ha="right")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# Helper: missing summary (monthly + yearly)
# ============================================================
def compute_missing_summaries(df_dt: pd.Series, series: pd.Series, value_name: str):
    tmp = pd.DataFrame({"dt": df_dt, value_name: series})
    tmp = tmp.dropna(subset=["dt"]).copy()

    tmp["Year"] = tmp["dt"].dt.year
    tmp["Month"] = tmp["dt"].dt.month

    # Monthly
    miss_m = (
        tmp.groupby(["Year", "Month"])[value_name]
        .apply(lambda x: x.isnull().sum())
        .reset_index(name="Missing_Count")
    )
    tot_m = tmp.groupby(["Year", "Month"]).size().reset_index(name="Total_Observations")
    monthly = pd.merge(miss_m, tot_m, on=["Year", "Month"], how="left")
    monthly["Missing_Percentage"] = (monthly["Missing_Count"] / monthly["Total_Observations"]) * 100
    monthly["YearMonth"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)

    # Yearly
    miss_y = (
        tmp.groupby(["Year"])[value_name]
        .apply(lambda x: x.isnull().sum())
        .reset_index(name="Missing_Count")
    )
    tot_y = tmp.groupby(["Year"]).size().reset_index(name="Total_Observations")
    yearly = pd.merge(miss_y, tot_y, on=["Year"], how="left")
    yearly["Missing_Percentage"] = (yearly["Missing_Count"] / yearly["Total_Observations"]) * 100
    yearly["Year"] = yearly["Year"].astype(int)

    return monthly, yearly


# ============================================================
# FSM helper functions (your original logic)
# ============================================================
def find_na_gaps(x: pd.Series):
    is_na = x.isna().to_numpy()
    n = len(is_na)

    gaps = []
    in_gap = False
    start = None

    for i in range(n):
        if is_na[i] and not in_gap:
            in_gap = True
            start = i
        elif (not is_na[i]) and in_gap:
            in_gap = False
            gaps.append((start, i - 1))

    if in_gap:
        gaps.append((start, n - 1))

    return gaps


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    diff = a - b
    return float(np.sqrt(np.sum(diff * diff)))


def fsm_find_best_match(values: np.ndarray,
                        gap_start: int,
                        gap_end: int,
                        m: int,
                        n: int,
                        const_c: float = 0.0,
                        max_candidates: int = None):
    N = len(values)
    gap_len = gap_end - gap_start + 1

    left_start = max(0, gap_start - m)
    left_len = gap_start - left_start

    right_end = min(N - 1, gap_end + n)
    right_len = right_end - gap_end

    if left_len == 0 and right_len == 0:
        return None, 0, gap_len, 0

    I_parts = []
    if left_len > 0:
        I_parts.append(values[left_start:gap_start])
    I_parts.append(np.full(gap_len, const_c, dtype=float))
    if right_len > 0:
        I_parts.append(values[gap_end + 1:right_end + 1])
    I_full = np.concatenate(I_parts).astype(float)
    z = len(I_full)

    valid_mask = ~np.isnan(I_full)
    if not np.any(valid_mask):
        return None, left_len, gap_len, right_len

    I_valid = I_full[valid_mask]

    best_dist = np.inf
    best_S = None

    indices = np.arange(0, N - z + 1) if N >= z else np.array([], dtype=int)

    if max_candidates is not None and len(indices) > max_candidates:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(indices, size=max_candidates, replace=False))

    gap_pos_in_I = np.zeros(z, dtype=bool)
    gap_pos_in_I[left_len:left_len + gap_len] = True

    for start in indices:
        end = start + z - 1

        if not (end < gap_start or start > gap_end):
            continue

        S_window = values[start:end + 1].astype(float)

        if np.any(np.isnan(S_window[valid_mask])):
            continue

        S_for_dist = S_window.copy()
        S_for_dist[gap_pos_in_I] = const_c

        d = euclidean_distance(I_valid, S_for_dist[valid_mask])

        if d < best_dist:
            best_dist = d
            best_S = S_window

    if best_S is None:
        return None, left_len, gap_len, right_len

    return best_S, left_len, gap_len, right_len


def fsm_impute_gap_diff(values: np.ndarray,
                        gap_start: int,
                        gap_end: int,
                        S_window: np.ndarray,
                        left_len: int,
                        gap_len: int,
                        right_len: int):
    x = values.copy()

    if left_len > 0:
        prev_idx = gap_start - 1
        if np.isnan(x[prev_idx]):
            return None

        s_pos = left_len
        current = x[prev_idx]
        for k in range(gap_len):
            if s_pos + k - 1 < 0:
                return None
            diff = S_window[s_pos + k] - S_window[s_pos + k - 1]
            current = current + diff
            x[gap_start + k] = current
        return x

    if right_len > 0:
        next_idx = gap_end + 1
        if np.isnan(x[next_idx]):
            return None

        s_pos_last = left_len + gap_len - 1
        current = x[next_idx]
        for k in range(gap_len):
            offset = gap_len - 1 - k
            if s_pos_last + 1 >= len(S_window):
                return None
            diff = S_window[s_pos_last + 1] - S_window[s_pos_last]
            current = current - diff
            x[gap_start + offset] = current
            s_pos_last -= 1
        return x

    return None


def fsm_impute_gap_scale(values: np.ndarray,
                         gap_start: int,
                         gap_end: int,
                         S_window: np.ndarray,
                         left_len: int,
                         gap_len: int,
                         right_len: int):
    x = values.copy()

    parts = []
    if left_len > 0:
        parts.append(x[gap_start - left_len:gap_start])
    parts.append(np.full(gap_len, np.nan))
    if right_len > 0:
        parts.append(x[gap_end + 1:gap_end + 1 + right_len])
    I_full = np.concatenate(parts).astype(float)

    known_mask = ~np.isnan(I_full)
    if not np.any(known_mask):
        return None

    query_known = I_full[known_mask]
    S_known = S_window[known_mask]

    q_min, q_max = np.nanmin(query_known), np.nanmax(query_known)
    s_min, s_max = np.nanmin(S_known), np.nanmax(S_known)

    if np.isclose(s_max - s_min, 0.0):
        scale = 1.0
    else:
        scale = (q_max - q_min) / (s_max - s_min)

    shift = q_min - s_min * scale
    S_scaled = S_window * scale + shift

    s_gap_start = left_len
    s_gap_end = left_len + gap_len
    x[gap_start:gap_end + 1] = S_scaled[s_gap_start:s_gap_end]
    return x


def impute_series_fsm(series: pd.Series,
                      mode: str = "FSM_scale",
                      m_factor: float = 1.0,
                      const_c: float = 0.0,
                      max_candidates: int = None,
                      verbose: bool = True) -> pd.Series:
    x = series.astype(float).to_numpy()
    original_index = series.index

    gaps = find_na_gaps(series)
    if verbose:
        st.write(f"Found {len(gaps)} gaps.")

    prog = st.progress(0.0)
    total = max(1, len(gaps))

    for gi, (start, end) in enumerate(gaps, 1):
        gap_len = end - start + 1
        m = max(1, int(m_factor * gap_len))
        n = m

        if verbose:
            st.write(f"[Gap {gi}] indices {start}–{end} (len={gap_len}), m=n={m}")

        S_window, left_len, g_len, right_len = fsm_find_best_match(
            x, gap_start=start, gap_end=end, m=m, n=n,
            const_c=const_c, max_candidates=max_candidates
        )

        if S_window is None:
            if verbose:
                st.write("  -> No valid match found, gap left as NaN.")
            prog.progress(gi / total)
            continue

        if mode == "FSM_diff":
            x_new = fsm_impute_gap_diff(x, start, end, S_window, left_len, g_len, right_len)
        elif mode == "FSM_scale":
            x_new = fsm_impute_gap_scale(x, start, end, S_window, left_len, g_len, right_len)
        else:
            raise ValueError("mode must be 'FSM_scale' or 'FSM_diff'")

        if x_new is None:
            if verbose:
                st.write("  -> Imputation failed for this gap, leaving as NaN.")
            prog.progress(gi / total)
            continue

        x = x_new
        prog.progress(gi / total)

    return pd.Series(x, index=original_index, name=series.name)




# ============================================================
# Interpolation helper functions (Linear + Polynomial)
# ============================================================
def impute_series_interpolation(series: pd.Series, method: str = "linear", order: int = 2):
    """
    Fill NaN gaps using pandas interpolation.

    Parameters
    ----------
    series : pd.Series
        Input numeric series containing missing values.
    method : str
        "linear" or "polynomial".
    order : int
        Polynomial order used only when method="polynomial".

    Returns
    -------
    pd.Series or None
        Imputed series when successful, otherwise None if polynomial
        interpolation cannot be completed.
    """
    s = series.astype(float).copy()

    if method == "linear":
        return s.interpolate(
            method="linear",
            limit_direction="both"
        )

    if method == "polynomial":
        try:
            return s.interpolate(
                method="polynomial",
                order=int(order),
                limit_direction="both"
            )

        except ImportError:
            st.error(
                "Polynomial interpolation requires SciPy. "
                "Please add 'scipy' to requirements.txt and redeploy the Streamlit app."
            )
            return None

        except Exception as e:
            st.error(
                "Polynomial interpolation failed. "
                "Check that there are enough valid data points and try a lower polynomial order. "
                f"Error: {e}"
            )
            return None

    raise ValueError("method must be 'linear' or 'polynomial'")


def build_imputation_output_df(df: pd.DataFrame, time_col: str, val_col: str,
                               series: pd.Series, imputed_full: pd.Series,
                               imputed_col: str) -> pd.DataFrame:
    out_df = df.copy()
    out_df[val_col] = series.values
    out_df[imputed_col] = imputed_full.values

    # TRUE only where original data was missing and the method successfully filled it
    out_df["Imputed_Flag"] = out_df[val_col].isna() & out_df[imputed_col].notna()
    return out_df


def make_segments_plot(out_df: pd.DataFrame, time_col: str, val_col: str,
                       imputed_col: str, method_label: str, data_type: str, y_label: str):
    """
    Plot original series plus imputed segments only.
    Original = blue line.
    Imputed segments only = red dashed line.
    """
    imputed_mask = out_df["Imputed_Flag"].to_numpy()
    imputed_only = np.where(imputed_mask, out_df[imputed_col].to_numpy(), np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=out_df[time_col],
        y=out_df[val_col],
        mode="lines",
        name="Original",
        line=dict(color="blue", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=out_df[time_col],
        y=imputed_only,
        mode="lines",
        name=f"{method_label} Imputed (segments only)",
        line=dict(color="red", width=2, dash="dash")
    ))
    fig.update_layout(
        title=f"{data_type}: Original + {method_label} Imputed Segments",
        xaxis_title="Date and Time",
        yaxis_title=y_label,
        hovermode="x unified"
    )

    png = line_png_bytes(
        x=out_df[time_col],
        y_list=[out_df[val_col].to_numpy(), imputed_only],
        labels=["Original", f"{method_label} Imputed (segments only)"],
        title=f"{data_type}: Original + {method_label} Imputed Segments",
        xlab="Date and Time",
        ylab=y_label,
        colors=["blue", "red"],
        linestyles=["-", "--"]
    )

    return fig, png


def make_monthly_seasonality_png(out_df: pd.DataFrame, time_col: str, val_col: str,
                                 imputed_col: str, method_label: str,
                                 data_type: str, y_label: str):
    """
    Monthly mean seasonality:
    Original = monthly mean using only observed/original values.
    Full infilled = monthly mean using full imputed series.
    """
    season_df = out_df[[time_col, val_col, imputed_col]].copy()
    season_df["Month"] = pd.to_datetime(season_df[time_col]).dt.month

    avg_orig = season_df.groupby("Month")[val_col].mean().reset_index()
    avg_full = season_df.groupby("Month")[imputed_col].mean().reset_index()

    figm, ax = plt.subplots(figsize=(10, 4))
    ax.plot(avg_orig["Month"], avg_orig[val_col], marker="o", label="Original", color="blue")
    ax.plot(avg_full["Month"], avg_full[imputed_col], marker="x", linestyle="--",
            label=f"FULL {method_label} Infilled", color="red")
    ax.set_title(f"Monthly Seasonality ({data_type}): Original vs FULL {method_label} Infilled")
    ax.set_xlabel("Month")
    ax.set_ylabel(y_label)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    figm.tight_layout()

    buf = io.BytesIO()
    figm.savefig(buf, format="png", dpi=300)
    plt.close(figm)
    buf.seek(0)
    return buf.getvalue()


def render_interpolation_tab(method_key: str, method_label: str, default_order: int = 2):
    """
    Reusable Streamlit tab content for Linear and Polynomial interpolation.
    method_key: 'linear' or 'polynomial'
    """
    st.header(f"{method_label} Imputation")

    poly_order = default_order

    if method_key == "polynomial":
        poly_order = st.number_input(
            "Polynomial order",
            min_value=2,
            max_value=5,
            value=int(default_order),
            step=1,
            key="poly_order_tab5"
        )

    run_interp = st.button(
        f"Run {method_label} Imputation",
        key=f"run_{method_key}"
    )

    if run_interp:
        with st.spinner(f"Running {method_label} imputation..."):
            imputed_full = impute_series_interpolation(
                series,
                method=method_key,
                order=int(poly_order)
            )

        # Stop this run if interpolation failed.
        if imputed_full is None:
            st.stop()

        suffix = (
            "Linear"
            if method_key == "linear"
            else f"Polynomial_Order{int(poly_order)}"
        )

        imputed_col = f"{val_col}_{suffix}_Imputed"

        out_df = build_imputation_output_df(
            df=df,
            time_col=time_col,
            val_col=val_col,
            series=series,
            imputed_full=imputed_full,
            imputed_col=imputed_col
        )

        n_filled = int(out_df["Imputed_Flag"].sum())

        st.success(
            f"{method_label} imputation completed. "
            f"Filled missing values: {n_filled}"
        )

        st.subheader("Preview imputed data")
        st.dataframe(
            out_df[[time_col, val_col, imputed_col, "Imputed_Flag"]].head(50),
            use_container_width=True
        )

        st.download_button(
            f"Download {method_label} imputed CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{val_col}_{suffix}_imputed.csv",
            mime="text/csv",
            key=f"download_csv_{method_key}"
        )

        # --------------------------------------------------------
        # Time series plot: Original + Imputed segments only
        # --------------------------------------------------------
        st.subheader(
            f"Original + {method_label} Imputed Segments (Imputed only)"
        )

        seg_fig, seg_png = make_segments_plot(
            out_df=out_df,
            time_col=time_col,
            val_col=val_col,
            imputed_col=imputed_col,
            method_label=method_label,
            data_type=data_type,
            y_label=y_label
        )

        st.plotly_chart(seg_fig, use_container_width=True)

        st.download_button(
            f"Download {method_label} segments plot (HTML)",
            data=seg_fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
            file_name=f"{val_col}_{suffix}_imputed_segments.html",
            mime="text/html",
            key=f"download_html_{method_key}"
        )

        # SHOW PNG inside app
        st.image(
            seg_png,
            caption=(
                f"Original + {method_label} Imputed Segments "
                "(PNG via Matplotlib)"
            ),
            use_container_width=True
        )

        st.download_button(
            f"Download {method_label} segments plot (PNG)",
            data=seg_png,
            file_name=f"{val_col}_{suffix}_imputed_segments.png",
            mime="image/png",
            key=f"download_png_{method_key}"
        )

        # --------------------------------------------------------
        # Monthly seasonality: Original vs FULL infilled series
        # --------------------------------------------------------
        st.subheader(
            f"Monthly Seasonality (Original vs FULL {method_label} Infilled)"
        )

        season_png = make_monthly_seasonality_png(
            out_df=out_df,
            time_col=time_col,
            val_col=val_col,
            imputed_col=imputed_col,
            method_label=method_label,
            data_type=data_type,
            y_label=y_label
        )

        # SHOW PNG inside app
        st.image(
            season_png,
            caption=(
                f"Monthly Seasonality: Original vs FULL "
                f"{method_label} Infilled"
            ),
            use_container_width=True
        )

        st.download_button(
            f"Download {method_label} monthly seasonality plot (PNG)",
            data=season_png,
            file_name=(
                f"{val_col}_{suffix}_monthly_seasonality_"
                "original_vs_full_infilled.png"
            ),
            mime="image/png",
            key=f"download_season_png_{method_key}"
        )



# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="FSM WL/SF App", layout="wide")
st.title("FSM Imputation App (WL / SF) — Upload File Only")

uploaded = st.file_uploader("Upload your Excel/CSV file", type=["xlsx", "xls", "csv"])
if uploaded is None:
    st.info("Upload an Excel (.xlsx/.xls) or CSV file to start.")
    st.stop()

try:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Cannot read file: {e}")
    st.stop()

df.columns = df.columns.astype(str).str.strip()

st.subheader("Uploaded Data Preview (Before Cleaning)")
st.dataframe(df.head(50), use_container_width=True)

cols = list(df.columns)
if len(cols) < 2:
    st.error("Your file must have at least 2 columns (datetime + WL/SF).")
    st.stop()

time_col = st.selectbox("Select datetime column", cols, index=0)
val_col = st.selectbox("Select WL/SF column", cols, index=1)

date_order = st.selectbox(
    "Date format/order",
    [
        "DMY (DD/MM/YYYY) — recommended for Malaysia",
        "YMD (YYYY-MM-DD)",
        "MDY (MM/DD/YYYY)",
        "AUTO (prefer day first)"
    ],
    index=0,
    help=(
        "Select DMY for dates such as 01/09/2020 = 1 September 2020. "
        "An incorrect order can make the time-series line appear irregular."
    )
)

missing_marker_text = st.text_input(
    "Numeric missing-value markers (comma-separated)",
    value="-99999, -9999, -999.99, -999, -99.99, 9999, 99999",
    help=(
        "These exact numeric values will be converted to NaN. Text markers such "
        "as NA, N/A, NaN, NULL, blank, missing and no data are handled automatically."
    )
)

original_row_count = len(df)
df[time_col] = parse_datetime_column(df[time_col], date_order)
invalid_datetime_count = int(df[time_col].isna().sum())
df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

numeric_markers = parse_numeric_markers(missing_marker_text)
series, explicit_marker_mask = clean_numeric_series(df[val_col], numeric_markers)

# Store the cleaned values in the working dataframe so all tabs and downloads
# use proper NaN values instead of sentinel codes such as -9999.
df[val_col] = series

duplicate_count = int(df[time_col].duplicated(keep=False).sum())

# Diagnose completely absent timestamps separately from NaN/sentinel values.
# The most common positive interval is treated as the expected sampling interval.
unique_times = df[time_col].drop_duplicates().sort_values()
positive_deltas = unique_times.diff().dropna()
positive_deltas = positive_deltas[positive_deltas > pd.Timedelta(0)]
if len(positive_deltas):
    delta_mode = positive_deltas.mode()
    expected_interval = delta_mode.iloc[0] if len(delta_mode) else positive_deltas.median()
    gap_deltas = positive_deltas[positive_deltas > expected_interval]
    time_gap_count = int(len(gap_deltas))
    estimated_absent_timestamps = int(
        sum(max(int(round(delta / expected_interval)) - 1, 0) for delta in gap_deltas)
    )
else:
    expected_interval = None
    time_gap_count = 0
    estimated_absent_timestamps = 0

y_label = infer_yaxis_label(val_col)
data_type = "Water Level" if y_label.startswith("Water Level") else ("Streamflow" if y_label.startswith("Streamflow") else "Value")

st.write(f"Rows uploaded: {original_row_count}")
st.write(f"Rows after datetime parsing: {len(df)}")
st.write(f"Invalid/unreadable timestamps removed: {invalid_datetime_count}")
st.write(f"Missing values in selected series: {int(series.isna().sum())}")
st.write(f"Explicit missing markers detected: {int(explicit_marker_mask.sum())}")
st.write(f"Rows involved in duplicate timestamps: {duplicate_count}")
st.write(
    "Most common sampling interval: "
    f"{expected_interval if expected_interval is not None else 'Not available'}"
)
st.write(f"Time gaps longer than the common interval: {time_gap_count}")
st.write(f"Estimated timestamps absent from the file: {estimated_absent_timestamps}")
st.write(f"Detected variable type: **{data_type}** → y-axis label: **{y_label}**")

with st.expander("View cleaned datetime and value data", expanded=False):
    st.dataframe(df[[time_col, val_col]].head(100), use_container_width=True)

if invalid_datetime_count > 0:
    st.warning(
        f"{invalid_datetime_count} timestamp(s) could not be parsed. "
        "Check the selected date format/order."
    )
if duplicate_count > 0:
    st.warning(
        f"The file contains {duplicate_count} row(s) involved in duplicate timestamps. "
        "The plot will show them, but review duplicates before final imputation."
    )

cleaned_download_df = df.copy()
cleaned_download_df[val_col] = series
st.download_button(
    "Download cleaned data (CSV)",
    data=cleaned_download_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{val_col}_cleaned_missing_values.csv",
    mime="text/csv"
)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["1) Raw Plot", "2) Missing Summary", "3) FSM Imputation", "4) Linear Interpolation", "5) Polynomial Interpolation"])

# ============================================================
# TAB 1) Raw time series (HTML + PNG) + SHOW PNG
# ============================================================
with tab1:
    st.header("1) Raw Time Series Plot")

    display_mode = st.selectbox(
        "Plot display",
        ["Original observations", "Daily mean", "7-day rolling mean", "Monthly mean"],
        index=0,
        help=(
            "Choose an averaged view for easier visual inspection. This changes only "
            "the displayed plot; imputation and downloads continue to use the cleaned "
            "original observations."
        )
    )
    plot_x, plot_y = prepare_plot_series(df[time_col], series, display_mode)

    raw_fig = go.Figure()
    raw_fig.add_trace(go.Scatter(
        x=plot_x,
        y=plot_y,
        mode="lines",
        name=display_mode,
        connectgaps=False,
        line=dict(width=1.2)
    ))
    raw_fig.update_layout(
        title=f"{data_type} Time Series: {val_col} ({display_mode})",
        xaxis_title="Date and Time",
        yaxis_title=y_label,
        hovermode="x unified"
    )
    st.plotly_chart(raw_fig, use_container_width=True)

    st.download_button(
        "Download raw plot (HTML)",
        data=raw_fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
        file_name=f"{val_col}_raw_time_series.html",
        mime="text/html"
    )

    raw_png = line_png_bytes(
        x=plot_x,
        y_list=[plot_y.to_numpy()],
        labels=[display_mode],
        title=f"{data_type} Time Series: {val_col} ({display_mode})",
        xlab="Date and Time",
        ylab=y_label,
        colors=["blue"],
        linestyles=["-"]
    )

    # ✅ SHOW PNG inside app
    st.image(raw_png, caption="Raw Time Series (PNG via Matplotlib)", use_container_width=True)

    st.download_button(
        "Download raw plot (PNG)",
        data=raw_png,
        file_name=f"{val_col}_raw_time_series.png",
        mime="image/png"
    )

# ============================================================
# TAB 2) Missing data summary (Monthly / Yearly) + SHOW PNG WITH % LABELS
# ============================================================
with tab2:
    st.header("2) Missing Data Summary (Monthly / Yearly)")

    monthly_summary, yearly_summary = compute_missing_summaries(df[time_col], series, val_col)
    choice = st.radio("Choose summary type:", ["Monthly", "Yearly"], horizontal=True, key="missing_choice")

    if choice == "Monthly":
        st.subheader("Monthly Missing Data Summary")
        st.dataframe(monthly_summary, use_container_width=True)

        miss_fig = go.Figure()
        miss_fig.add_trace(go.Bar(x=monthly_summary["YearMonth"], y=monthly_summary["Missing_Percentage"], name="Missing %"))
        miss_fig.update_layout(
            title=f"Monthly Missing Data Percentage: {val_col}",
            xaxis_title="Year-Month",
            yaxis_title="Missing Percentage (%)",
            xaxis_tickangle=-45,
            hovermode="x unified"
        )
        st.plotly_chart(miss_fig, use_container_width=True)

        miss_png = bar_png_bytes(
            x=monthly_summary["YearMonth"].tolist(),
            y=monthly_summary["Missing_Percentage"].tolist(),
            title=f"Monthly Missing Data Percentage: {val_col}",
            xlab="Year-Month",
            ylab="Missing Percentage (%)",
            show_value_labels=True,
            max_xticks=24,
            label_min_value=0.05
        )

        # ✅ SHOW PNG inside app
        st.image(miss_png, caption="Monthly Missing % (PNG with labels)", use_container_width=True)

        st.download_button(
            "Download MONTHLY missing summary (CSV)",
            data=monthly_summary.to_csv(index=False).encode("utf-8"),
            file_name=f"{val_col}_monthly_missing_summary.csv",
            mime="text/csv"
        )
        st.download_button(
            "Download MONTHLY missing plot (HTML)",
            data=miss_fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
            file_name=f"{val_col}_monthly_missing_plot.html",
            mime="text/html"
        )
        st.download_button(
            "Download MONTHLY missing plot (PNG)",
            data=miss_png,
            file_name=f"{val_col}_monthly_missing_plot.png",
            mime="image/png"
        )

    else:
        st.subheader("Yearly Missing Data Summary")
        st.dataframe(yearly_summary, use_container_width=True)

        miss_fig_y = go.Figure()
        miss_fig_y.add_trace(go.Bar(x=yearly_summary["Year"].astype(str), y=yearly_summary["Missing_Percentage"], name="Missing %"))
        miss_fig_y.update_layout(
            title=f"Yearly Missing Data Percentage: {val_col}",
            xaxis_title="Year",
            yaxis_title="Missing Percentage (%)",
            xaxis_tickangle=0,
            hovermode="x unified"
        )
        st.plotly_chart(miss_fig_y, use_container_width=True)

        miss_png_y = bar_png_bytes(
            x=yearly_summary["Year"].astype(str).tolist(),
            y=yearly_summary["Missing_Percentage"].tolist(),
            title=f"Yearly Missing Data Percentage: {val_col}",
            xlab="Year",
            ylab="Missing Percentage (%)",
            rotate_xticks=0,
            show_value_labels=True,
            max_xticks=30,
            label_min_value=0.05
        )

        # ✅ SHOW PNG inside app
        st.image(miss_png_y, caption="Yearly Missing % (PNG with labels)", use_container_width=True)

        st.download_button(
            "Download YEARLY missing summary (CSV)",
            data=yearly_summary.to_csv(index=False).encode("utf-8"),
            file_name=f"{val_col}_yearly_missing_summary.csv",
            mime="text/csv"
        )
        st.download_button(
            "Download YEARLY missing plot (HTML)",
            data=miss_fig_y.to_html(include_plotlyjs="cdn").encode("utf-8"),
            file_name=f"{val_col}_yearly_missing_plot.html",
            mime="text/html"
        )
        st.download_button(
            "Download YEARLY missing plot (PNG)",
            data=miss_png_y,
            file_name=f"{val_col}_yearly_missing_plot.png",
            mime="image/png"
        )

# ============================================================
# TAB 3) FSM Imputation + SHOW PNGs (segments) + seasonality already shown
# ============================================================
with tab3:
    st.header("3) FSM Imputation")

    mode = st.selectbox("FSM mode", ["FSM_scale", "FSM_diff"], index=0)
    m_factor = st.number_input("m_factor (context length factor)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    const_c = st.number_input("const_c (constant inside dummy gap)", value=0.0)
    max_candidates = st.number_input("max_candidates (0 = no cap)", min_value=0, value=0, step=1000)
    max_candidates = None if max_candidates == 0 else int(max_candidates)

    run = st.button("Run FSM Imputation")

    if run:
        with st.spinner("Running FSM imputation..."):
            imputed_full = impute_series_fsm(
                series,
                mode=mode,
                m_factor=float(m_factor),
                const_c=float(const_c),
                max_candidates=max_candidates,
                verbose=True
            )

        out_df = df.copy()
        imputed_col = f"{val_col}_FSM_{mode}"
        out_df[val_col] = series.values
        out_df[imputed_col] = imputed_full.values

        st.success("FSM imputation completed.")
        st.subheader("Preview imputed data")
        st.dataframe(out_df[[time_col, val_col, imputed_col]].head(50), use_container_width=True)

        st.download_button(
            "Download imputed CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{val_col}_FSM_imputed.csv",
            mime="text/csv"
        )

        # --------------------------------------------------------
        # Time series plot: Original + Imputed segments only
        # --------------------------------------------------------
        st.subheader("Original + FSM Imputed Segments (Imputed only)")

        imputed_mask = out_df[val_col].isna() & out_df[imputed_col].notna()
        imputed_only = np.where(imputed_mask.to_numpy(), out_df[imputed_col].to_numpy(), np.nan)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=out_df[time_col],
            y=out_df[val_col],
            mode="lines",
            name="Original",
            line=dict(color="blue", width=1)
        ))
        fig2.add_trace(go.Scatter(
            x=out_df[time_col],
            y=imputed_only,
            mode="lines",
            name="FSM Imputed (segments only)",
            line=dict(color="red", width=2, dash="dash")
        ))
        fig2.update_layout(
            title=f"{data_type}: Original + FSM Imputed Segments ({mode})",
            xaxis_title="Date and Time",
            yaxis_title=y_label,
            hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            "Download segments plot (HTML)",
            data=fig2.to_html(include_plotlyjs="cdn").encode("utf-8"),
            file_name=f"{val_col}_original_plus_fsm_imputed_segments.html",
            mime="text/html"
        )

        png2 = line_png_bytes(
            x=out_df[time_col],
            y_list=[out_df[val_col].to_numpy(), imputed_only],
            labels=["Original", "FSM Imputed (segments only)"],
            title=f"{data_type}: Original + FSM Imputed Segments ({mode})",
            xlab="Date and Time",
            ylab=y_label,
            colors=["blue", "red"],
            linestyles=["-", "--"]
        )

        # ✅ SHOW PNG inside app
        st.image(png2, caption="Original + FSM Imputed Segments (PNG via Matplotlib)", use_container_width=True)

        st.download_button(
            "Download segments plot (PNG)",
            data=png2,
            file_name=f"{val_col}_original_plus_fsm_imputed_segments.png",
            mime="image/png"
        )

        # --------------------------------------------------------
        # Monthly seasonality: Original vs FULL infilled series
        # --------------------------------------------------------
        st.subheader("Monthly Seasonality (Original vs FULL FSM Infilled)")

        season_df = out_df[[time_col, val_col, imputed_col]].copy()
        season_df["Month"] = pd.to_datetime(season_df[time_col]).dt.month

        avg_orig = season_df.groupby("Month")[val_col].mean().reset_index()
        avg_full = season_df.groupby("Month")[imputed_col].mean().reset_index()

        figm, ax = plt.subplots(figsize=(10, 4))
        ax.plot(avg_orig["Month"], avg_orig[val_col], marker="o", label="Original", color="blue")
        ax.plot(avg_full["Month"], avg_full[imputed_col], marker="x", linestyle="--", label="FULL FSM Infilled", color="red")
        ax.set_title(f"Monthly Seasonality ({data_type}): Original vs FULL FSM Infilled")
        ax.set_xlabel("Month")
        ax.set_ylabel(y_label)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(figm)

        buf = io.BytesIO()
        figm.tight_layout()
        figm.savefig(buf, format="png", dpi=300)
        plt.close(figm)
        buf.seek(0)

        st.download_button(
            "Download monthly seasonality plot (PNG)",
            data=buf.getvalue(),
            file_name=f"{val_col}_monthly_seasonality_original_vs_full_fsm.png",
            mime="image/png"
        )


# ============================================================
# TAB 4) Linear Interpolation Imputation + CSV download + SHOW PNGs
# ============================================================
with tab4:
    render_interpolation_tab(
        method_key="linear",
        method_label="Linear Interpolation"
    )

# ============================================================
# TAB 5) Polynomial Interpolation Imputation + CSV download + SHOW PNGs
# ============================================================
with tab5:
    render_interpolation_tab(
        method_key="polynomial",
        method_label="Polynomial Interpolation",
        default_order=2
    )
