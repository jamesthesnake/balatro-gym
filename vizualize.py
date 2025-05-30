import streamlit as st
import json
import os

EPISODE_DIR = "trajectories"
episodes = sorted([f for f in os.listdir(EPISODE_DIR) if f.endswith(".jsonl")])
selected = st.selectbox("Pick episode", episodes)

with open(os.path.join(EPISODE_DIR, selected)) as f:
    data = [json.loads(line) for line in f]

for i, step in enumerate(data):
    st.markdown(f"### Step {i}")
    st.json({
        "action": step["action"],
        "reward": step["reward"],
        "done": step["done"],
        "obs_summary": step["obs"].get("score", None)
    })
