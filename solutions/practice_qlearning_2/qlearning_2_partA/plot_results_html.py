import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import json

def build_results_table(fig, type_result):
    runs = []
    time_strings = []
    gammas = []
    epsilon_mins = []
    epsilon_maxs = []
    epsilon_percentages = []
    alphas = []
    total_episodes = []
    global_mean_rewards = []
    global_mean_variances = []
    file_paths = sorted(glob.glob(f"results/{type_result}/*.json"))
    for file in file_paths:
        with open(file, "r") as f:
            data = json.load(f)
        runs.append(data['experiment_name'] )
        time_strings.append(data['time_string'])
        total_episodes.append(data['episodes'][-1])
        global_mean_rewards.append(data['global_mean_reward'])
        global_mean_variances.append(data['global_mean_variance'])
        if type_result == 'train':
            gammas.append(data['params']['gamma'])
            epsilon_mins.append(data['params']['epsilon_min'])
            epsilon_maxs.append(data['params']['epsilon_max'])
            epsilon_percentages.append(data['params']['epsilon_percentage'])
            alphas.append(data['params']['alpha'])

    # Add the Table to the first row
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Run ID</b>",
                        "<b>Time string</b>",
                        "<b>Gamma</b>",
                        "<b>Epsilon max</b>",
                        "<b>Epsilon min</b>",
                        "<b>Epsilon percentage</b>",
                        "<b>Alpha</b>",
                        "<b>Total episodes</b>",
                        "<b>Global mean reward</b>",
                        "<b>Global mean variance</b>"],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[runs, time_strings, gammas, epsilon_maxs,
                        epsilon_mins, epsilon_percentages,
                        alphas, total_episodes, global_mean_rewards, global_mean_variances],
                fill_color='lavender',
                align='left'
            )
        ),
        row=1, col=1
    )


def build_results_graph(fig, type_result):
    file_paths = sorted(glob.glob(f"results/{type_result}/*.json"))
    # Load all experiment JSON files in directory
    for file in file_paths:
        with open(file, "r") as f:
            data = json.load(f)

        # Add a curve for the 50-episode moving average reward
        fig.add_trace(go.Scatter(
            x=data["episodes"],
            y=data["rewards"],
            mode='lines',
            name=data["experiment_name"] + ' ' + data['time_string'],
            hovertemplate="Episode %{x}<br> Reward: %{y:.2f}"
        ))
        # in case one needs to add the total reward at each episode
        # fig.add_trace(go.Scatter(
        #     x=data["episodes"],
        #     y=data["rewards"],
        #     mode='lines',
        #     name=data["experiment_name"] + ' ' + data['time_string'],
        #     hovertemplate="Episode %{x}<br> Reward: %{y:.2f}"
        # ))
    # Adjust size and axis titles in update_layout
    fig.update_layout(
        title=f"{type_result} reward over time (TRAIN: inline test. TEST: total reward at each episode)",
        # Set Axis Titles
        xaxis_title="Episodes",
        yaxis_title="Reward",
        # Make the graph bigger (dimensions in pixels)
        width=2000,
        height=1200,
        # Optional: Increase font sizes for better readability on larger charts
        font=dict(size=14)
    )

def build_results_html(type_result):
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.01,  # Reduced spacing between rows (3% instead of 10%)
        row_heights=[0.2, 0.8],  # 20% of vertical height for table, 80% for graph
        specs=[[{"type": "table"}],
               [{"type": "scatter"}]]
    )
    # add a table with the hyper parameters
    build_results_table(fig, type_result)
    # add the graphs
    build_results_graph(fig, type_result)
    fig.write_html(f"results/lunar_lander_DQN_experiments_{type_result}.html")
    fig.show()


if __name__ == "__main__":
    type_result='train'
    build_results_html(type_result=type_result)
    type_result='test'
    build_results_html(type_result=type_result)
