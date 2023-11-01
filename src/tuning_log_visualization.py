import optuna
import plotly.graph_objects as go

study = optuna.create_study(storage="sqlite:///easy_tuning.db", study_name="easy_parallel_importance", load_if_exists=True)

fig = go.Figure()

fig = optuna.visualization.plot_param_importances(study)
fig.write_image('./image/test_importance.png')

fig = optuna.visualization.plot_optimization_history(study)
fig.write_image('./image/test_history.png')