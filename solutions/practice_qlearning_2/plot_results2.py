import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# 1. Define data (Plotly tables prefer columns rather than rows)
runs = ["Exp_1", "Exp_2", "Exp_3"]
gammas = [0.99, 0.95, 0.99]
episodes = [5000, 5000, 10000]
learning_rates = [0.001, 0.005, 0.001]

# 2. Create a subplot figure: 2 rows, 1 column.
# The top row is for the table, the bottom row is for the chart.
fig = make_subplots(
    rows=2, cols=1,
    vertical_spacing=0.1,
    specs=[[{"type": "table"}],
           [{"type": "scatter"}]]
)

# 3. Add the Table to the first row
fig.add_trace(
    go.Table(
        header=dict(
            values=["<b>Run ID</b>", "<b>Gamma</b>", "<b>Episodes</b>", "<b>Learning Rate</b>"],
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[runs, gammas, episodes, learning_rates],
            fill_color='lavender',
            align='left'
        )
    ),
    row=1, col=1
)

# 4. Add the Scatter plots to the second row
fig.add_trace(go.Scatter(y=[10, 50, 150], name="Exp_1"), row=2, col=1)
fig.add_trace(go.Scatter(y=[5, 40, 120], name="Exp_2"), row=2, col=1)
fig.add_trace(go.Scatter(y=[15, 80, 210], name="Exp_3"), row=2, col=1)

# 5. Update layout and save
fig.update_layout(
    height=800, # Make the figure tall enough to fit both nicely
    title_text="Multi-Experiment Tracking",
)

time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fig.write_html(f"plotly_table_report_{time_string}.html")