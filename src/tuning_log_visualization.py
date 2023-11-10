import optuna
import plotly.graph_objects as go

study = optuna.create_study(storage="sqlite:///easy_hight_load_multi_tuning.db", study_name="tuning", load_if_exists=True)

fig = go.Figure()

fig = optuna.visualization.plot_param_importances(study)
fig.write_image('./image/easy_hight_load_importance.png')

fig = optuna.visualization.plot_optimization_history(study)
fig.write_image('./image/easy_hight_load_history.png')