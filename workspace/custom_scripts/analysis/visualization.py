# can i also have animation in that case like changing the slider value in a particular speed may be so that i can also in a way see the video
# /opt/miniconda/envs/analysis/bin/streamlit run custom_scripts/analysis/visualization.py --server.port 8501 --server.address 0.0.0.0

import streamlit as st
import pandas as pd
import plotly.express as px
import cv2
import os
import time

# === Load logs ===

exp_name = "teleop_test_2"
csv_file_path =f"custom_scripts/logs/ppo_factory/csv_files/{exp_name}.csv"
video_path = f"custom_scripts/logs/ppo_factory/{exp_name}/cam0_video.mp4"

df = pd.read_csv(csv_file_path)



# Load video using OpenCV
# video_path = "rollout.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
st.sidebar.text(f"Video FPS: {fps}, Frames: {n_frames}")

# === UI ===
st.title("Agent Rollout Visualization")

# Slider over step indices
step = st.slider("Step", 0, len(df)-1, 0)

## play button
play = st.button("Play")
steps_per_sec = st.sidebar.slider("Steps per second", 1, 50, int(fps))

# === Sidebar: choose signals ===
columns = st.sidebar.multiselect(
    "Select signals to plot",
    df.columns.drop("step"),
    default=["dof_torque_0"]
)

## creating a side widget that helps in creating a new pannel for visualization
# new_columns = st.sidebar.multiselect(
#     "Select signals to plot",
#     df.columns.drop("step"),
#     default=["dof_torque_0"]
# )


if "plots" not in st.session_state:
    st.session_state.plots = []

st.sidebar.header("Add a new graph")

# Select value(s) to plot
cols = st.sidebar.multiselect("Select columns", df.columns.tolist())
title = st.sidebar.text_input("Plot title")

if st.sidebar.button("Add Plot"):
    if cols:
        st.session_state.plots.append({
            "columns":cols,
            "title": title or ", ".join(cols),
        })



# Compute corresponding frame number (assuming 1-to-1 step->frame)
if play:
    for s in range(step, len(df)):
        st.session_state['step'] = s  # store current step
        frame_number = s
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, caption=f"Step {s} (Frame {frame_number})")
        else:
            st.write("Could not read frame")
        if columns:
            fig = px.line(df, x="step", y=columns, title="Signals over steps")
            fig.add_vline(x=s, line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        time.sleep(1.0 / steps_per_sec)
else:
    frame_number = step

    # Seek to the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        # OpenCV loads frames in BGR, convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption=f"Step {step} (Frame {frame_number})")
    else:
        st.write("Could not read frame")


    # Plot signals with vertical marker
    if columns:
        fig = px.line(df, x="step", y=columns, title="Signals over steps")
        fig.add_vline(x=step, line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    
    st.header("user Graphs")
    for i, plot in enumerate(st.session_state.plots):
        # st.subheader(plot["title"])
        # st.line_chart(df[plot["columns"]])

        fig = px.line(df, x="step", y=plot["columns"], title=plot["title"])
        fig.add_vline(x=step, line_color="red")
        st.plotly_chart(fig, use_container_width=True)

