import os
import pickle
import click
import mlflow
import optuna

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000") 
"""
# as the server is already running so the uri is take from the stacktrac after running the command `mlflow ui --backend-store-uri sqlite:///mlflow.db`
[2023-05-28 15:11:25 +0530] [6944] [INFO] Starting gunicorn 20.1.0
[2023-05-28 15:11:25 +0530] [6944] [INFO] Listening at: http://127.0.0.1:5000 (6944)
[2023-05-28 15:11:25 +0530] [6944] [INFO] Using worker: sync
[2023-05-28 15:11:25 +0530] [6945] [INFO] Booting worker with pid: 6945
[2023-05-28 15:11:25 +0530] [6946] [INFO] Booting worker with pid: 6946
[2023-05-28 15:11:25 +0530] [6947] [INFO] Booting worker with pid: 6947
[2023-05-28 15:11:25 +0530] [6948] [INFO] Booting worker with pid: 6948
"""
mlflow.set_experiment("random-forest-hyperopt")

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=10,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(trial):

        with mlflow.start_run():
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
                'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
                'random_state': 42,
                'n_jobs': -1
            }

            mlflow.log_params(params)

            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)

            mlflow.log_metric("rmse", rmse) 

        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)
    """
    here in the above code we are just taking a study with "sampler=TPESampler" and 
    trying to optimizing the objective func
    "objective" func is consisting 
    > of hyper-paramters provided to the Regressor, 
    > then fitting the model, 
    > then pridicting 
    > comparing the true values <> predicted values and returning a rsme
    
    """


if __name__ == '__main__':
    run_optimization()
