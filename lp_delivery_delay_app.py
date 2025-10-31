import os
from typing import Optional
import time
import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
TABLE_FQN = "leap.ai.delivery_delay_predictions"  # change if different

# exact columns (as provided)
COLS = [
    "masterbill_internal_id",
    "volume_cm3",
    "distance_km",
    "total_weight",
    "hazardous_material_flag",
    "margin_value",
    "delivery_delay",
    "risk_factor",
    "risk_level",
    "load_id",
    "Source",
    "Destination",
    "carrier_name",
    "mode",
]

# -------------------------------------------------
# STREAMLIT PAGE
# -------------------------------------------------
st.set_page_config(page_title="Delay Prediction", page_icon="📦", layout="wide")
st.title("Delay Prediction")
st.caption("Predict potential delivery delays using advanced analytics")

# -------------------------------------------------
# DATABRICKS UTIL
# -------------------------------------------------
def _can_sql_connector() -> bool:
    return all((
        os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        os.getenv("DATABRICKS_HTTP_PATH"),
        os.getenv("DATABRICKS_TOKEN"),
    ))

WAREHOUSE_NAME = "Serverless (prod)"   # change if needed

def run_sql_df(query: str) -> pd.DataFrame:
    # 1) Prefer SQL connector if envs are present
    if _can_sql_connector():
        from databricks import sql as dbsql
        with dbsql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME").strip(),
            http_path=os.getenv("DATABRICKS_HTTP_PATH").strip(),
            access_token=os.getenv("DATABRICKS_TOKEN").strip(),
        ) as conn, conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall_arrow().to_pandas()

    # 2) Fallback: Statement Execution API via SDK
    ws = WorkspaceClient(config=Config())

    wh_id = (os.getenv("DATABRICKS_WAREHOUSE_ID") or "").strip()
    if not wh_id and WAREHOUSE_NAME:
        for wh in ws.warehouses.list():
            if wh.name == WAREHOUSE_NAME:
                wh_id = wh.id
                break

    if not wh_id:
        raise RuntimeError(
            "No warehouse available. Set DATABRICKS_WAREHOUSE_ID (or WAREHOUSE_NAME) "
            "or provide SQL connector envs (SERVER_HOSTNAME/HTTP_PATH/TOKEN)."
        )

    resp = ws.statement_execution.execute_statement(
        warehouse_id=wh_id,
        statement=query,
        wait_timeout="20s"
    )

    from databricks.sdk.service.sql import StatementState
    st_info = ws.statement_execution.get_statement(statement_id=resp.statement_id)
    while st_info.status.state in {StatementState.PENDING, StatementState.RUNNING}:
        time.sleep(0.5)
        st_info = ws.statement_execution.get_statement(statement_id=resp.statement_id)

    if st_info.status.state is not StatementState.SUCCEEDED:
        err = getattr(getattr(st_info.status, "error", None), "message", None)
        cause = err or getattr(st_info.status, "cause", None) or "unknown error"
        raise RuntimeError(f"Statement {resp.statement_id} {st_info.status.state}: {cause}")

    manifest = st_info.manifest
    col_names = [c.name for c in manifest.schema.columns]

    rows = []
    total_chunks = getattr(manifest, "total_chunk_count", None) or 1
    for i in range(total_chunks):
        chunk = ws.statement_execution.get_statement_result_chunk_n(
            statement_id=resp.statement_id,
            chunk_index=i
        )
        if getattr(chunk, "data_array", None):
            rows.extend(chunk.data_array)

    df = pd.DataFrame(rows, columns=col_names)
    return df

def qi(name: str) -> str:
    return f"`{name}`"

def _select_cols_sql(cols) -> str:
    return ", ".join(qi(c) for c in cols)

@st.cache_data(ttl=90, show_spinner=True)
def fetch_data(limit: int = 5000) -> pd.DataFrame:
    q = f"SELECT {_select_cols_sql(COLS)} FROM {TABLE_FQN} LIMIT {limit}"
    df = run_sql_df(q)
    # guarantee all expected columns exist
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return df[COLS]

# -------------------------------------------------
# RISK NORMALIZATION & DISPLAY HELPERS
# -------------------------------------------------
def norm_risk(label: Optional[str]) -> str:
    l = (label or "").strip().lower()
    if l in ("high", "high risk", "delayed", "delay", "late"):
        return "High"
    if l in ("medium", "warning"):
        return "Medium"
    if l in ("low", "at risk"):
        return "Low"
    return "Normal"

def to_percent(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            s = x.strip().replace("%", "").replace(",", "")
            if not s:
                return None
            v = float(s)
        else:
            v = float(x)
        v = 0.0 if v < 0 else (100.0 if v > 100 else v)
        return v
    except Exception:
        return None

def fallback_percent_from_label(label: Optional[str]) -> float:
    l = (label or "").strip().lower()
    if l in ("high risk", "delayed", "delay", "late"):   return 95.0
    if l in ("medium", "warning"):                       return 80.0
    if l in ("at risk", "low"):                          return 60.0
    return 25.0

def pct_text(percent: Optional[float]) -> str:
    return "—" if percent is None else f"{int(round(max(0.0, min(100.0, float(percent)))))}%"

def risk_badge(label: str) -> str:
    l = (label or "").strip().lower()
    if l in ("high", "high risk", "delayed", "delay", "late"):
        return "🛑 **High**"
    if l in ("medium", "warning"):
        return "🟠 **Medium**"
    if l in ("low", "at risk"):
        return "🟡 **Low**"
    return "✅ **Normal**"

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = fetch_data()
print("[DATA] Loaded:", len(df), "rows from", TABLE_FQN)

# -------------------------------------------------
# HEADER ROW: Title | Risk (multi-select dropdown) | Search
# -------------------------------------------------
st.markdown("""
<style>
/* Compact search */
div[data-testid="stTextInput"] > div > div > input {
    height: 36px; padding: 6px 10px; font-size: 14px; border-radius: 10px;
}
/* Compact multi-select (checkbox dropdown) */
div[data-baseweb="select"] > div {
    min-height: 36px; border-radius: 10px; font-size: 14px;
}
/* Subheader tight spacing */
div[data-testid="stMarkdownContainer"] h3 { margin-bottom: 0.2rem; }
</style>
""", unsafe_allow_html=True)

col_title, col_search, col_filter = st.columns([3.8, 2.4, 1.8])

with col_title:
    st.subheader("Active Shipments")

with col_search:
    search = st.text_input(
        "", placeholder="Search shipments…",
        label_visibility="collapsed",
        key="search_box_header"
    )

with col_filter:
    risk_selected = st.multiselect(
        label="",
        options=["Normal", "Low", "Medium", "High"],     # order: Normal → Low → Medium → High
        default=[],                                       # empty ⇒ All Risk Levels
        label_visibility="collapsed",
        placeholder="All Risk Levels",
        key="risk_multiselect_header"
    )

# final selected risk categories (empty list means no filtering by risk)
_selected_levels = list(risk_selected)

# -------------------------------------------------
# PREPARE DISPLAY DATA
# -------------------------------------------------
disp = df.copy()
disp["__shipment__"] = disp["masterbill_internal_id"].astype(str)
disp["__origin__"]   = disp["Source"].astype(str)
disp["__dest__"]     = disp["Destination"].astype(str)

# percent from risk_factor, fallback to label bucket if missing
p = disp["risk_factor"].apply(to_percent)
p = p.where(p.notna(), disp["risk_level"].apply(fallback_percent_from_label))
disp["__pct__"]   = p
disp["__label__"] = disp["risk_level"].fillna("On Time")
disp["__cat__"]   = disp["__label__"].apply(norm_risk)  # Normal/Low/Medium/High

def matches(row) -> bool:
    # risk filter
    if _selected_levels and row["__cat__"] not in _selected_levels:
        return False
    # search filter
    q = search.strip().lower()
    if not q:
        return True
    hay = " ".join([
        str(row["__shipment__"]),
        str(row["__origin__"]),
        str(row["__dest__"]),
        str(row.get("carrier_name", "")),
        str(row.get("load_id", "")),
    ]).lower()
    return q in hay

if not disp.empty:
    disp = disp[disp.apply(matches, axis=1)].reset_index(drop=True)

# -------------------------------------------------
# TABLE HEADERS
# -------------------------------------------------
h1, h2, h3, h4, h5 = st.columns([1.6, 2.0, 2.0, 1.2, 1.6])
h1.markdown("**SHIPMENT ID**")
h2.markdown("**SOURCE**")
h3.markdown("**DESTINATION**")
h4.markdown("**RISK LEVEL**")
h5.markdown("**DELAY RISK**")

# -------------------------------------------------
# PAGINATION (10 rows/page)
# -------------------------------------------------
rows_per_page = 10
total_rows = len(disp)
total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# clamp to available pages (in case filters changed)
st.session_state.current_page = max(1, min(st.session_state.current_page, total_pages))

start_idx = (st.session_state.current_page - 1) * rows_per_page
end_idx   = start_idx + rows_per_page
disp_page = disp.iloc[start_idx:end_idx]

# -------------------------------------------------
# ROWS
# -------------------------------------------------
if disp.empty:
    st.info("No shipments match your filters.")
else:
    for _, r in disp_page.iterrows():
        c1, c2, c3, c4, c5 = st.columns([1.6, 2.0, 2.0, 1.2, 1.6])
        with c1:
            st.markdown(f"**{r['__shipment__']}**")
        with c2:
            st.write(r["__origin__"])
        with c3:
            st.write(r["__dest__"])
        with c4:
            st.markdown(risk_badge(str(r["__label__"])))
        with c5:
            percent = float(r["__pct__"]) if pd.notna(r["__pct__"]) else 0.0
            percent = 0.0 if percent < 0 else (100.0 if percent > 100 else percent)

            if percent <= 50:
                color = "#2ecc71"
            elif percent <= 70:
                color = "#f1c40f"
            elif percent <= 90:
                color = "#e67e22"
            else:
                color = "#e74c3c"

            # Bar + right-aligned bold percent on one line
            bar_html = f"""
            <div style="display:flex;align-items:center;gap:8px;width:100%;">
            <div style="flex:1;background-color:#eee;border-radius:6px;height:18px;overflow:hidden;">
                <div style="height:100%;width:{percent:.1f}%;background-color:{color};transition:width 0.4s ease;"></div>
            </div>
            <span style="min-width:44px;text-align:right;font-weight:700;">{int(round(percent))}%</span>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

# -------------------------------------------------
# PAGER (uniform size, tight spacing, bottom-right)
# -------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)  # subtle gap below last row

# Make all pager buttons identical size; keep "Last" aligned right
spacer, c_first, c_prev, c_page, c_next, c_last = st.columns([5.6, 1, 1, 1.6, 1, 1])

st.markdown("""
<style>
div[data-testid="stButton"] button {
    width: 100%;
    height: 38px;
    font-size: 14px !important;
    padding: 4px 0 !important;
}
</style>
""", unsafe_allow_html=True)

with c_first:
    if st.button("⏮️ First", key="pager_first"):
        st.session_state.current_page = 1

with c_prev:
    if st.button("◀️ Prev", key="pager_prev"):
        st.session_state.current_page = max(1, st.session_state.current_page - 1)

with c_page:
    st.markdown(
        f"<div style='text-align:center;padding-top:6px;font-weight:500;white-space:nowrap;'>"
        f"Page {st.session_state.current_page} / {total_pages}</div>",
        unsafe_allow_html=True,
    )

with c_next:
    if st.button("Next ▶️", key="pager_next"):
        st.session_state.current_page = min(total_pages, st.session_state.current_page + 1)

with c_last:
    if st.button("Last ⏭️", key="pager_last"):
        st.session_state.current_page = total_pages

# -------------------------------------------------
# KPIs (optional)
# -------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

try:
    delayed = int((df["delivery_delay"] == 1).sum()) if "delivery_delay" in df.columns else None
    if delayed is not None:
        k1.metric("Delayed (label=1)", f"{delayed:,}")
except Exception:
    pass

try:
    k2.metric("Avg Distance (km)", f"{df['distance_km'].astype(float).mean():.1f}")
except Exception:
    pass

try:
    k3.metric("Avg Weight", f"{df['total_weight'].astype(float).mean():.1f}")
except Exception:
    pass

try:
    k4.metric("Hazardous Shipments", f"{int((df['hazardous_material_flag'] == 1).sum())}")
except Exception:
    pass