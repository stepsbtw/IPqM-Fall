import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from streamlit_shortcuts import shortcut_button

TRIALS_ROOT = Path("IPqM-Fall/trials_90hz")
META_FILE = "window_labels.csv"
FS = 90

st.set_page_config(layout="wide")

# =========================
# CACHE
# =========================
@st.cache_data
def load_meta():
    return pd.read_csv(META_FILE)

@st.cache_data
def load_trial(path):
    df = pd.read_parquet(path)
    return df.reset_index(drop=True)

@st.cache_data
def build_full_arrays(df, channels):
    full_x = np.arange(len(df)) / FS
    cached = {c: df[c].values for c in channels}
    return full_x, cached

def save_meta(updated_meta):
    updated_meta.to_csv(META_FILE, index=False)
    load_meta.clear()

# =========================
# LOAD DATA & INITIAL STATE
# =========================
meta = load_meta()
trial_files = sorted(meta["file"].unique())

if "state_initialized" not in st.session_state:
    unreviewed = meta[meta["reviewed"] == False]
    
    if len(unreviewed) > 0:
        first_unreviewed = unreviewed.iloc[0]
        initial_trial = first_unreviewed["file"]
        initial_trial_meta = meta[meta["file"] == initial_trial]
        initial_window_ids = initial_trial_meta["window_id"].tolist()
        
        st.session_state.trial_index = trial_files.index(initial_trial)
        st.session_state.selected_trial = initial_trial
        st.session_state.window_index = initial_window_ids.index(first_unreviewed["window_id"])
    else:
        st.session_state.trial_index = 0
        st.session_state.selected_trial = trial_files[0]
        st.session_state.window_index = 0
        
    st.session_state.state_initialized = True

# =========================
# HELPERS
# =========================
def get_window_key():
    return f"window_selectbox_{st.session_state.selected_trial}"

def set_current_window(index):
    st.session_state.window_index = index
    current_trial_meta = meta[meta["file"] == st.session_state.selected_trial]
    current_window_ids = current_trial_meta["window_id"].tolist()
    
    if len(current_window_ids) > 0:
        st.session_state[get_window_key()] = current_window_ids[index]

def sync_trial_state():
    st.session_state.trial_index = trial_files.index(st.session_state.selected_trial)
    set_current_window(0)

# =========================
# CALLBACK FUNCTIONS
# =========================
def save_and_advance_callback(label_value, current_window, current_trial, current_window_ids):
    # This executes BEFORE widgets render, preventing StreamlitAPIExceptions
    global meta
    meta.loc[meta["window_id"] == current_window, "label"] = label_value
    meta.loc[meta["window_id"] == current_window, "reviewed"] = True
    save_meta(meta)

    current_window_idx = current_window_ids.index(current_window)

    # Next window
    if current_window_idx < len(current_window_ids) - 1:
        set_current_window(current_window_idx + 1)
    # Next trial
    else:
        current_trial_idx = trial_files.index(current_trial)
        if current_trial_idx < len(trial_files) - 1:
            st.session_state.trial_index = current_trial_idx + 1
            st.session_state.selected_trial = trial_files[st.session_state.trial_index]
            set_current_window(0)

def apply_remaining_callback(current_label, current_window, current_trial, current_window_ids):
    global meta
    current_idx = current_window_ids.index(current_window)
    remaining_window_ids = current_window_ids[current_idx:]
    
    meta.loc[meta["window_id"].isin(remaining_window_ids), "label"] = current_label
    meta.loc[meta["window_id"].isin(remaining_window_ids), "reviewed"] = True
    save_meta(meta)

    current_trial_idx = trial_files.index(current_trial)
    if current_trial_idx < len(trial_files) - 1:
        st.session_state.trial_index = current_trial_idx + 1
        st.session_state.selected_trial = trial_files[st.session_state.trial_index]
        set_current_window(0)

def on_window_change():
    window_select_key = get_window_key()
    new_choice = st.session_state[window_select_key]
    
    # Safely align indices before plotting
    current_trial_meta = meta[meta["file"] == st.session_state.selected_trial]
    current_window_ids = current_trial_meta["window_id"].tolist()
    if new_choice in current_window_ids:
        st.session_state.window_index = current_window_ids.index(new_choice)

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("Trial Viewer")
trial_nav_cols = st.sidebar.columns(2)

with trial_nav_cols[0]:
    if st.button("⬅ Previous Trial", key="prev_trial_btn"):
        if st.session_state.trial_index > 0:
            st.session_state.trial_index -= 1
            st.session_state.selected_trial = trial_files[st.session_state.trial_index]
            set_current_window(0)
            st.rerun()

with trial_nav_cols[1]:
    if st.button("Next Trial ➡", key="next_trial_btn"):
        if st.session_state.trial_index < len(trial_files) - 1:
            st.session_state.trial_index += 1
            st.session_state.selected_trial = trial_files[st.session_state.trial_index]
            set_current_window(0)
            st.rerun()

selected_trial = st.sidebar.selectbox(
    "Select trial",
    trial_files,
    index=st.session_state.trial_index
)

if selected_trial != st.session_state.selected_trial:
    st.session_state.selected_trial = selected_trial
    sync_trial_state()
    st.rerun()

# Fetch metadata properties for selection
selected_trial = st.session_state.selected_trial
trial_meta = meta[meta["file"] == selected_trial]
window_ids = trial_meta["window_id"].tolist()

if len(window_ids) == 0:
    st.error("No windows found.")
    st.stop()

if st.session_state.window_index >= len(window_ids):
    set_current_window(0)

# Window Navigation UI Buttons
window_nav_cols = st.sidebar.columns(2)
with window_nav_cols[0]:
    if st.button("⬅ Previous Window", key="prev_window_btn"):
        if st.session_state.window_index > 0:
            set_current_window(st.session_state.window_index - 1)
            st.rerun()

with window_nav_cols[1]:
    if st.button("Next Window ➡", key="next_window_btn"):
        if st.session_state.window_index < len(window_ids) - 1:
            set_current_window(st.session_state.window_index + 1)
            st.rerun()

# =========================
# WINDOW SELECTOR WIDGET
# =========================
window_select_key = get_window_key()
if window_select_key not in st.session_state:
    st.session_state[window_select_key] = window_ids[st.session_state.window_index]

selected_window = st.sidebar.selectbox(
    "Select window",
    window_ids,
    key=window_select_key,
    on_change=on_window_change
)

# Robust fallback adjustment if internal indices lose track
if st.session_state.window_index >= len(window_ids) or window_ids[st.session_state.window_index] != st.session_state[window_select_key]:
    if st.session_state[window_select_key] in window_ids:
        st.session_state.window_index = window_ids.index(st.session_state[window_select_key])
    else:
        st.session_state.window_index = 0

selected_window = window_ids[st.session_state.window_index]

# =========================
# DATA DIGESTION
# =========================
trial_path = TRIALS_ROOT / selected_trial
trial_df = load_trial(trial_path)
channels = [c for c in trial_df.columns if c != "timestamp"]
full_x, cached_channels = build_full_arrays(trial_df, channels)

selected_row = trial_meta[trial_meta["window_id"] == selected_window].iloc[0]
start_idx = int(selected_row["start_idx"])
end_idx = int(selected_row["end_idx"])
window_df = trial_df.iloc[start_idx:end_idx]

# Progress Bar Dashboard
reviewed_count = int(meta["reviewed"].sum())
total_count = len(meta)
st.sidebar.markdown("---")
st.sidebar.write(f"Reviewed: {reviewed_count}/{total_count}")
st.sidebar.progress(reviewed_count / total_count)

# Labels Configuration
labels = [
    "ADL_1", "ADL_2", "ADL_3", "ADL_4", "ADL_5", "ADL_6", "ADL_7", "ADL_8",
    "ADL_11", "ADL_12", "ADL_13", "ADL_14", "ADL_15", "OM_1", "OM_2", "OM_3",
    "OM_4", "OM_5", "OM_6", "OM_7", "OM_8", "OM_9", "FALL_1", "FALL_2",
    "FALL_3", "FALL_5", "FALL_6", "UNKNOWN"
]

label_shortcuts = {
    "1": "ADL_1", "2": "ADL_2", "3": "ADL_3", "4": "ADL_4", "5": "ADL_5",
    "6": "ADL_6", "7": "ADL_7", "8": "ADL_8", "q": "ADL_11", "w": "ADL_12",
    "e": "ADL_13", "r": "ADL_14", "t": "ADL_15", "a": "OM_1", "s": "OM_2",
    "d": "OM_3", "f": "OM_4", "g": "OM_5", "h": "OM_6", "j": "OM_7",
    "k": "OM_8", "l": "OM_9", "z": "FALL_1", "x": "FALL_2", "c": "FALL_3",
    "v": "FALL_5", "b": "FALL_6", "0": "UNKNOWN"
}

current_label = selected_row["label"] if selected_row["label"] in labels else "UNKNOWN"

# =========================
# LAYOUT & VISUALIZATION
# =========================
left_col, right_col = st.columns([3, 2])

with left_col:
    fig = go.Figure()
    for c in channels:
        fig.add_trace(go.Scattergl(x=full_x, y=cached_channels[c], mode="lines", name=c))
    
    fig.add_vrect(
        x0=start_idx / FS, x1=end_idx / FS,
        fillcolor="rgba(0,255,0,0.25)", line_width=2, line_color="green"
    )
    fig.update_layout(height=500, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Seconds")
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    fig2 = go.Figure()
    window_x = np.arange(len(window_df)) / FS
    for c in channels:
        fig2.add_trace(go.Scattergl(x=window_x, y=window_df[c], mode="lines", name=c))
        
    fig2.update_layout(height=350, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Seconds")
    st.plotly_chart(fig2, use_container_width=True)

    st.write(f"Current label: **{current_label}** | Reviewed: **{selected_row['reviewed']}**")

    # Shortcut Grid
    shortcut_cols = st.columns(5)
    for idx, (key, label) in enumerate(label_shortcuts.items()):
        with shortcut_cols[idx % 5]:
            # Refactored execution pattern using on_click event handling args
            if shortcut_button(
                label=f"{key}: {label}", 
                shortcut=key, 
                on_click=save_and_advance_callback, 
                args=(label, selected_window, selected_trial, window_ids)
            ):
                pass

    # Secure Label Selector Dropdown
    selected_dropdown = st.selectbox(
        "Select label manually",
        labels,
        index=labels.index(current_label),
        key=f"dropdown_{selected_window}"
    )
    
    st.button(
        "Apply Dropdown Label 💾", 
        use_container_width=True,
        on_click=save_and_advance_callback,
        args=(selected_dropdown, selected_window, selected_trial, window_ids)
    )

    # Secondary Navigation Buttons
    st.button(
        f"Keep Current Label ({current_label})", 
        use_container_width=True,
        on_click=save_and_advance_callback,
        args=(current_label, selected_window, selected_trial, window_ids)
    )

    st.button(
        f"Apply '{current_label}' to Remaining Windows", 
        use_container_width=True,
        on_click=apply_remaining_callback,
        args=(current_label, selected_window, selected_trial, window_ids)
    )

    st.caption(" | ".join([f"{k}:{v}" for k, v in label_shortcuts.items()]))