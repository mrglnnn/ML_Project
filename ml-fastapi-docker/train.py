"""Train the same Iris model as Practical Task 6 and track it in MLflow."""
import os
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

MODEL_PATH = Path(__file__).with_name('model.joblib')
MODEL_NAME = 'IrisRandomForest'
EXPERIMENT = 'Iris Classification SIS-3'


def fit_model():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, predictions)),
        'f1_score': float(f1_score(y_test, predictions, average='macro')),
    }
    return model, X_train, X_test, metrics


def train():
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name='Random Forest - 100 trees') as run:
        model, X_train, X_test, metrics = fit_model()
        mlflow.log_params({**model.get_params(), 'test_size': 0.2, 'stratified': True,
                           'train_samples': len(X_train), 'test_samples': len(X_test)})
        mlflow.log_metrics(metrics)
        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path='saved_model')
        info = mlflow.sklearn.log_model(
             sk_model=model,
             name='iris_model',
             signature=infer_signature(X_train, model.predict(X_train)),
             input_example=X_train[:1],
               skops_trusted_types=["sklearn.tree._tree.Tree"],
)
        version = mlflow.register_model(info.model_uri, MODEL_NAME)
        print(f'Accuracy: {metrics["accuracy"]:.4f}')
        print(f'Macro F1-score: {metrics["f1_score"]:.4f}')
        print(f'Run ID: {run.info.run_id}')
        print(f'Registered model: {version.name}, version {version.version}')
        print(f'FastAPI model: {MODEL_PATH}')


if __name__ == '__main__':
    train()
