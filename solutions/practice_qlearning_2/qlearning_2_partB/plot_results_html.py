import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import json

def build_results_table(fig):
    time_strings = []
    gammas = []
    hidden_layers = []
    total_episodes = []
    mean_rewards = []
    mean_variances = []
    freqs = []
    batch_sizes = []
    file_paths = sorted(glob.glob(f"results/test/*.json"))
    for file in file_paths:
        with open(file, "r") as f:
            data = json.load(f)
        # runs.append(data['experiment_name'] )
        time_strings.append(data['time_string'])
        total_episodes.append(data['params']['n_episodes_train'])
        mean_rewards.append(data['mean_reward'])
        mean_variances.append(data['mean_variance'])
        hidden_layers.append(data['params']['hidden_layer_sizes'])
        gammas.append(data['params']['gamma'])
        freqs.append(data['params']['target_update_freq'])
        batch_sizes.append(data['params']['batch_size'])

    # Add the Table to the first row
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Time string</b>",
                        "<b>Gamma</b>",
                        "<b>Target update freq.</b>",
                        "<b>Batch size</b>",
                        "<b>Hidden layers</b>",
                        "<b>N episodes train</b>",
                        "<b>Mean reward</b>",
                        "<b>Mean variance</b>"],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[time_strings, gammas, freqs, batch_sizes, hidden_layers, total_episodes, mean_rewards, mean_variances],
                fill_color='lavender',
                align='left'
            )
        ),
        row=1, col=1
    )


def build_results_graph(fig):
    file_paths = sorted(glob.glob(f"results/test/*.json"))
    # Load all experiment JSON files in directory
    for file in file_paths:
        with open(file, "r") as f:
            data = json.load(f)
        # Add a curve for the 50-episode moving average reward
        fig.add_trace(go.Scatter(
            x=list(range(len(data["rewards"]))),
            y=data["rewards"],
            mode='lines',
            name= data['time_string'],
            hovertemplate="Episode %{x}<br> Reward: %{y:.2f}"
        ))

    # Adjust size and axis titles in update_layout
    fig.update_layout(
        title=f"Test at different episodes",
        # Set Axis Titles
        xaxis_title="Episodes",
        yaxis_title="Reward",
        # Make the graph bigger (dimensions in pixels)
        width=2000,
        height=1200,
        # Optional: Increase font sizes for better readability on larger charts
        font=dict(size=14)
    )

def build_results_html():
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.01,  # Reduced spacing between rows (3% instead of 10%)
        row_heights=[0.2, 0.8],  # 20% of vertical height for table, 80% for graph
        specs=[[{"type": "table"}],
               [{"type": "scatter"}]]
    )
    # add a table with the hyper parameters
    build_results_table(fig)
    # add the graphs
    build_results_graph(fig)
    fig.write_html(f"results/lunar_lander_DQN_experiments_train_test.html")
    fig.show()


if __name__ == "__main__":
    build_results_html()

