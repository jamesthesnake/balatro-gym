import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any
import time

# Import shared data structures
from data_structures import (
    TrajectoryStep, EpisodeMetadata, DatasetMetadata, 
    BalatroActions, StrategyTags, validate_trajectory_step,
    compute_episode_statistics
)

# Page config
st.set_page_config(
    page_title="Balatro Dataset Viewer",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_dataset(dataset_path: str):
    """Load the trajectory dataset with caching"""
    dataset_path = Path(dataset_path)
    
    # Load metadata
    metadata_file = dataset_path / "dataset_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Load episodes
    episodes = []
    episode_files = sorted(dataset_path.glob("episode_*.pkl"))
    
    if not episode_files:
        st.error(f"No episode files found in {dataset_path}")
        return [], metadata
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, episode_file in enumerate(episode_files):
        try:
            with open(episode_file, 'rb') as f:
                episode = pickle.load(f)
                
                # Validate episode structure
                if isinstance(episode, list) and all(isinstance(step, TrajectoryStep) for step in episode):
                    episodes.append(episode)
                else:
                    st.warning(f"Episode {episode_file} has invalid structure")
            
            # Update progress
            progress = (i + 1) / len(episode_files)
            progress_bar.progress(progress)
            status_text.text(f"Loading episode {i+1}/{len(episode_files)}")
            
        except Exception as e:
            st.warning(f"Failed to load {episode_file}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    if episodes:
        st.success(f"Successfully loaded {len(episodes)} episodes!")
    else:
        st.error("No valid episodes found!")
    
    return episodes, metadata

def get_strategy_stats(episodes):
    """Compute strategy statistics across all episodes"""
    strategy_counts = Counter()
    strategy_rewards = defaultdict(list)
    strategy_transitions = defaultdict(lambda: defaultdict(int))
    
    for episode in episodes:
        prev_tags = None
        for step in episode:
            # Count strategy tags
            for tag in step.strategy_tags:
                strategy_counts[tag] += 1
                strategy_rewards[tag].append(step.reward)
            
            # Count transitions
            if prev_tags:
                for prev_tag in prev_tags:
                    for curr_tag in step.strategy_tags:
                        strategy_transitions[prev_tag][curr_tag] += 1
            
            prev_tags = step.strategy_tags
    
    # Compute average rewards
    avg_strategy_rewards = {
        tag: np.mean(rewards) for tag, rewards in strategy_rewards.items()
    }
    
    return strategy_counts, avg_strategy_rewards, dict(strategy_transitions)

def render_card(card_data, size="normal"):
    """Render a playing card with suit and rank"""
    if not card_data:
        return ""
    
    rank = card_data.get('rank', '?')
    suit = card_data.get('suit', '?')
    
    # Color based on suit
    color = 'red' if suit in ['♥', '♦'] else 'black'
    
    # Size settings
    if size == "small":
        width, height, font_size = "40px", "60px", "12px"
    else:
        width, height, font_size = "60px", "90px", "16px"
    
    return f"""
    <div style="
        border: 2px solid #333;
        border-radius: 8px;
        padding: 4px;
        text-align: center;
        background: white;
        color: {color};
        font-weight: bold;
        width: {width};
        height: {height};
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin: 2px;
        font-size: {font_size};
        box-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    ">
        {rank}<br>{suit}
    </div>
    """

def render_hand(cards, title="Hand", size="normal"):
    """Render a collection of cards"""
    if not cards:
        st.write(f"**{title}:** Empty")
        return
    
    st.write(f"**{title}:**")
    
    # Render cards in rows of 8
    for i in range(0, len(cards), 8):
        row_cards = cards[i:i+8]
        card_html = "".join([render_card(card, size) for card in row_cards])
        st.markdown(f'<div style="margin: 5px 0;">{card_html}</div>', unsafe_allow_html=True)

def render_game_state(state: Dict):
    """Render the complete game state"""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Cards section
        st.subheader("🃏 Cards")
        
        if 'hand' in state:
            render_hand(state['hand'], "Current Hand")
        
        if 'selected_cards' in state and state['selected_cards']:
            selected = [state['hand'][i] for i in state['selected_cards'] if i < len(state.get('hand', []))]
            render_hand(selected, "Selected Cards", "small")
        
        if 'played_cards' in state:
            render_hand(state['played_cards'], "Played Cards", "small")
    
    with col2:
        # Game stats
        st.subheader("📊 Game Stats")
        
        stats_data = {
            "Metric": [],
            "Value": []
        }
        
        # Core stats
        core_stats = ['chips', 'mult', 'money', 'ante', 'discards', 'hands_left']
        for stat in core_stats:
            if stat in state:
                stats_data["Metric"].append(stat.replace('_', ' ').title())
                stats_data["Value"].append(state[stat])
        
        if stats_data["Metric"]:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        # Jokers section
        st.subheader("🃏 Jokers")
        if 'jokers' in state and state['jokers']:
            for i, joker in enumerate(state['jokers']):
                joker_name = joker.get('name', f'Joker {i+1}')
                joker_desc = joker.get('description', 'No description')
                st.write(f"**{joker_name}**: {joker_desc}")
        else:
            st.write("No jokers active")

def render_step_analysis(step, step_idx):
    """Render detailed analysis of a single step"""
    
    # Step header
    st.subheader(f"📍 Step {step_idx} Analysis")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        action_name = BalatroActions.get_action_name(step.action)
        st.metric("Action", action_name.replace('_', ' ').title())
    
    with col2:
        st.metric("Reward", f"{step.reward:.2f}")
    
    with col3:
        st.metric("Reward-to-Go", f"{step.rtg:.2f}")
    
    with col4:
        st.metric("Strategy Tags", len(step.strategy_tags))
    
    # Strategy tags
    if step.strategy_tags:
        st.write("**Strategy Tags:**")
        tag_cols = st.columns(min(len(step.strategy_tags), 4))
        for i, tag in enumerate(step.strategy_tags):
            with tag_cols[i % 4]:
                st.code(tag)
    
    # State comparison
    if st.checkbox("Show State Comparison", key=f"state_comp_{step_idx}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Before Action:**")
            render_game_state(step.state)
        
        with col2:
            st.write("**After Action:**")
            render_game_state(step.next_state)

def main():
    st.title("🃏 Balatro Trajectory Dataset Viewer")
    st.markdown("Explore and analyze Balatro gameplay trajectories for reinforcement learning research.")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        # Dataset selection
        dataset_path = st.text_input(
            "Dataset Path", 
            value="./balatro_dataset",
            help="Path to the directory containing episode files"
        )
        
        if not Path(dataset_path).exists():
            st.error(f"Dataset path does not exist: {dataset_path}")
            st.stop()
        
        # Load dataset
        if st.button("Load Dataset", type="primary"):
            st.session_state.dataset_loaded = True
            # Clear cached data
            if 'episodes' in st.session_state:
                del st.session_state.episodes
                del st.session_state.metadata
        
        if 'dataset_loaded' not in st.session_state:
            st.session_state.dataset_loaded = False
        
        # Page selection
        page = st.selectbox(
            "Select Page",
            ["Dataset Overview", "Episode Browser", "Strategy Analysis", "State Inspector"]
        )
    
    # Load data if needed
    if st.session_state.dataset_loaded:
        if 'episodes' not in st.session_state:
            with st.spinner("Loading dataset..."):
                episodes, metadata = load_dataset(dataset_path)
                st.session_state.episodes = episodes
                st.session_state.metadata = metadata
        
        episodes = st.session_state.episodes
        metadata = st.session_state.metadata
        
        if not episodes:
            st.error("No episodes loaded. Please check your dataset path and files.")
            st.stop()
    else:
        st.info("👈 Please load a dataset using the sidebar to begin exploration.")
        st.stop()
    
    # Render selected page
    if page == "Dataset Overview":
        render_overview_page(episodes, metadata)
    elif page == "Episode Browser":
        render_episode_browser_page(episodes)
    elif page == "Strategy Analysis":
        render_strategy_analysis_page(episodes)
    elif page == "State Inspector":
        render_state_inspector_page(episodes)

def render_overview_page(episodes, metadata):
    """Render dataset overview page"""
    st.header("📈 Dataset Overview")
    
    # High-level metrics
    total_episodes = len(episodes)
    total_steps = sum(len(episode) for episode in episodes)
    total_reward = sum(sum(step.reward for step in episode) for episode in episodes)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Episodes", total_episodes)
    with col2:
        st.metric("Total Steps", f"{total_steps:,}")
    with col3:
        st.metric("Avg Steps/Episode", f"{total_steps/max(total_episodes,1):.1f}")
    with col4:
        st.metric("Avg Reward/Episode", f"{total_reward/max(total_episodes,1):.2f}")
    
    # Episode length distribution
    st.subheader("Episode Length Distribution")
    episode_lengths = [len(episode) for episode in episodes]
    
    fig = px.histogram(
        x=episode_lengths,
        nbins=30,
        title="Distribution of Episode Lengths",
        labels={'x': 'Episode Length (steps)', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Reward distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Episode Reward Distribution")
        episode_rewards = [sum(step.reward for step in episode) for episode in episodes]
        
        fig = px.histogram(
            x=episode_rewards,
            nbins=30,
            title="Total Reward per Episode"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Step Reward Distribution")
        step_rewards = [step.reward for episode in episodes for step in episode]
        
        fig = px.histogram(
            x=step_rewards,
            nbins=50,
            title="Reward per Step"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Action distribution
    st.subheader("Action Distribution")
    action_counts = Counter()
    for episode in episodes:
        for step in episode:
            action_counts[step.action] += 1
    
    action_names = [BalatroActions.get_action_name(action) for action in action_counts.keys()]
    action_values = list(action_counts.values())
    
    fig = px.bar(
        x=action_names,
        y=action_values,
        title="Distribution of Actions Taken"
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metadata
    if metadata:
        st.subheader("Dataset Metadata")
        metadata_df = pd.DataFrame([
            {"Property": k, "Value": str(v)} for k, v in metadata.items()
        ])
        st.dataframe(metadata_df, hide_index=True)

def render_episode_browser_page(episodes):
    """Render episode browser page"""
    st.header("🔍 Episode Browser")
    
    if not episodes:
        st.warning("No episodes found in dataset.")
        return
    
    # Episode selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        episode_idx = st.selectbox(
            "Select Episode",
            range(len(episodes)),
            format_func=lambda x: f"Episode {x} ({len(episodes[x])} steps, {sum(s.reward for s in episodes[x]):.1f} total reward)"
        )
    
    episode = episodes[episode_idx]
    
    with col2:
        # Strategy filter
        all_tags = set()
        for step in episode:
            all_tags.update(step.strategy_tags)
        
        strategy_filter = st.multiselect(
            "Filter by Strategy Tags",
            sorted(all_tags),
            key=f"strategy_filter_{episode_idx}"
        )
    
    with col3:
        # Step range
        max_step = len(episode) - 1
        step_range = st.slider(
            "Step Range",
            0, max_step,
            (0, min(50, max_step)),
            key=f"step_range_{episode_idx}"
        )
    
    # Filter steps
    filtered_steps = []
    for i, step in enumerate(episode[step_range[0]:step_range[1]+1]):
        if not strategy_filter or any(tag in step.strategy_tags for tag in strategy_filter):
            filtered_steps.append((step_range[0] + i, step))
    
    if not filtered_steps:
        st.warning("No steps match the current filters.")
        return
    
    # Step selection
    step_options = [f"Step {idx}: {BalatroActions.get_action_name(step.action)}" for idx, step in filtered_steps]
    selected_step_idx = st.selectbox(
        "Select Step",
        range(len(filtered_steps)),
        format_func=lambda x: step_options[x]
    )
    
    step_idx, step = filtered_steps[selected_step_idx]
    
    # Render step details
    render_step_analysis(step, step_idx)
    
    # Game state
    st.subheader("🎮 Game State")
    render_game_state(step.state)

def render_strategy_analysis_page(episodes):
    """Render strategy analysis page"""
    st.header("🎯 Strategy Analysis")
    
    # Compute strategy statistics
    with st.spinner("Computing strategy statistics..."):
        strategy_counts, avg_strategy_rewards, strategy_transitions = get_strategy_stats(episodes)
    
    # Strategy frequency
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strategy Tag Frequency")
        
        if strategy_counts:
            # Top 15 most common strategies
            top_strategies = dict(strategy_counts.most_common(15))
            
            fig = px.bar(
                x=list(top_strategies.values()),
                y=list(top_strategies.keys()),
                orientation='h',
                title="Most Common Strategy Tags"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average Reward by Strategy")
        
        if avg_strategy_rewards:
            # Sort by reward
            sorted_rewards = sorted(avg_strategy_rewards.items(), key=lambda x: x[1], reverse=True)[:15]
            
            fig = px.bar(
                x=[reward for _, reward in sorted_rewards],
                y=[strategy for strategy, _ in sorted_rewards],
                orientation='h',
                title="Highest Reward Strategy Tags"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Strategy transitions (if we have enough data)
    if strategy_transitions:
        st.subheader("Strategy Transition Analysis")
        
        # Select a strategy to analyze
        available_strategies = list(strategy_transitions.keys())
        if available_strategies:
            selected_strategy = st.selectbox(
                "Select strategy to analyze transitions from:",
                available_strategies[:20]  # Limit to first 20 for performance
            )
            
            if selected_strategy in strategy_transitions:
                transitions = strategy_transitions[selected_strategy]
                
                if transitions:
                    # Show top transitions
                    top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    transition_df = pd.DataFrame(
                        top_transitions,
                        columns=['Next Strategy', 'Count']
                    )
                    
                    fig = px.bar(
                        transition_df,
                        x='Count',
                        y='Next Strategy',
                        orientation='h',
                        title=f"Most Common Transitions from '{selected_strategy}'"
                    )
                    st.plotly_chart(fig, use_container_width=True)

def render_state_inspector_page(episodes):
    """Render detailed state inspector"""
    st.header("🔬 State Inspector")
    
    if not episodes:
        st.warning("No episodes found.")
        return
    
    # Random state selector
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🎲 Random State", type="primary"):
            # Pick random episode and step
            random_episode_idx = np.random.randint(0, len(episodes))
            random_step_idx = np.random.randint(0, len(episodes[random_episode_idx]))
            
            st.session_state.inspect_episode = random_episode_idx
            st.session_state.inspect_step = random_step_idx
    
    with col2:
        # Manual selection
        episode_idx = st.number_input(
            "Episode Index",
            min_value=0,
            max_value=len(episodes)-1,
            value=st.session_state.get('inspect_episode', 0)
        )
        
        step_idx = st.number_input(
            "Step Index", 
            min_value=0,
            max_value=len(episodes[episode_idx])-1,
            value=st.session_state.get('inspect_step', 0)
        )
    
    # Show selected state
    step = episodes[episode_idx][step_idx]
    
    st.subheader(f"State from Episode {episode_idx}, Step {step_idx}")
    
    # Raw state data
    with st.expander("🔍 Raw State Data", expanded=False):
        st.json(step.state)
    
    # Rendered state
    render_game_state(step.state)
    
    # Step analysis
    render_step_analysis(step, step_idx)

if __name__ == "__main__":
    main()
