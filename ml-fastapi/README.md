# Iris Classification — SIS-3

Continuation of [Practical Task 6](https://github.com/mrglnnn/ml-fastapi-docker). The original Random Forest, Iris dataset, FastAPI request contract and Docker backend are retained. SIS-3 adds a Streamlit client and MLflow experiment tracking and model registry.

## Architecture

```text
Browser -> Streamlit :8501 -> FastAPI :8000 (Docker) -> model.joblib
Iris data -> train.py -> trained Random Forest -> model.joblib
                     -> MLflow :5000 -> SQLite metadata + artifacts + registry
```

The same training run exports the API's joblib file and logs/registers the trained estimator in MLflow. The API deliberately uses the local exported file rather than automatically fetching the latest registry version. Rebuild the Docker image after retraining to deploy the updated file.

## Run on macOS

Use Python 3.13 (as in Practical Task 6) and Docker Desktop. From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-sis3.txt
```

Terminal 1 (keep running):

```bash
source .venv/bin/activate
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Terminal 2 (in the same directory):

```bash
source .venv/bin/activate
python train.py
docker build -t iris-ml-api .
docker run --rm -p 8000:8000 --name iris-api iris-ml-api
```

If Docker Desktop is unavailable, run `python -m uvicorn main:app --port 8000` instead of the Docker commands (local API demonstration only).

Terminal 3 (same directory):

```bash
source .venv/bin/activate
python -m streamlit run frontend.py
```

Open http://localhost:8501 for the frontend, http://127.0.0.1:8000/docs for Swagger, and http://127.0.0.1:5000 for MLflow. Use port 5000 consistently; if macOS AirPlay Receiver already occupies it, disable AirPlay Receiver or use another port and set `MLFLOW_TRACKING_URI` to the new URL before training.

## MLflow evidence

- Experiment: `Iris Classification SIS-3`
- Run: `Random Forest - 100 trees`
- Parameters: actual Random Forest hyperparameters and split information.
- Metrics: held-out accuracy and macro F1-score.
- Run artifact: `saved_model/model.joblib`.
- Logged sklearn model: `iris_model`, including a signature and input example.
- Registry: `IrisRandomForest`; first successful registration in a fresh database creates version 1. A later training run creates the next version.

## API verification

In another terminal with the environment activated, run `python verify_api.py`. This checks the health endpoint, one prediction per class and invalid-input validation. In Streamlit, submit the default measurements (5.1, 3.5, 1.4, 0.2); expect Iris Setosa. Save screenshots of the real frontend, experiment metrics/parameters, run artifact and registry version for submission. Do not commit the local SQLite database or MLflow artifact directory. The experiment is reproducible by running `train.py` against the server.

## Submission

Commit the SIS-3 files and real evidence (screenshots), then push to the existing GitHub repository. The assignment does not specify a submission format; the repository link and screenshots are suggested evidence.
