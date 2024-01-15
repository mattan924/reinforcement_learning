import optuna
import plotly.graph_objects as go

study = optuna.create_study(storage="sqlite:///tuning_DB/high_low.db", study_name="client20_fix20", load_if_exists=True)

fig = go.Figure()

fig = optuna.visualization.plot_param_importances(study)
fig.write_image('./image/high_capacity_low_cycle_client20_fix20_importance.png')

fig = optuna.visualization.plot_optimization_history(study)
fig.write_image('./image/high_capacity_low_cycle_clietn20_fix_20_history.png')