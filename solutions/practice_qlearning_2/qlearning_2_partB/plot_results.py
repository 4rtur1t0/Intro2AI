import plotly.graph_objects as go
import json
import glob

# Create an interactive figure
fig = go.Figure()
# Load all experiment JSON files in directory
for file in glob.glob("results/*.json"):
    with open(file, "r") as f:
        data = json.load(f)
    
    # Add a curve for the 50-episode moving average reward
    fig.add_trace(go.Scatter(
        x=data["episodes"],
        y=data["avg_rewards"],
        mode='lines',
        name=data["experiment_filename"],
        hovertemplate="Episode %{x}<br>Avg Reward: %{y:.2f}"
    ))

# Customize interactive layout
fig.update_layout(
    title="Lunar Lander DQN Hyperparameter Comparison",
    xaxis_title="Episode",
    yaxis_title="Average Reward (Last 50)",
    hovermode="x unified",
    legend=dict(
        title="Click to Toggle Runs:",
        itemclick="toggle",      # Single click turns line on/off
        itemdoubleclick="toggleothers" # Double click isolates line
    )
)
# Show inside Jupyter or open in browser
fig.show()
# Optional: Save as an interactive HTML webpage
fig.write_html("lunar_lander_experiments.html")