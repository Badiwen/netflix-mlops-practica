import mlflow
from mlflow.tracking import MlflowClient

NOMBRE_EXPERIMENTO = "clasificacion-netflix-rating-iteraciones"
NOMBRE_MODELO = "netflix-rating-classifier"
ALIAS = "champion"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Se buscan las runs del experimento ordenadas por accuracy descendente
runs = mlflow.search_runs(
    experiment_names=[NOMBRE_EXPERIMENTO],
    order_by=["metrics.accuracy DESC"],
)

if runs.empty:
    raise RuntimeError(
        f"No se encontraron runs en el experimento '{NOMBRE_EXPERIMENTO}'. "
        "Corre train.py antes de registrar un modelo."
    )

runs_finalizadas = runs[runs["status"] == "FINISHED"]
if len(runs_finalizadas) < 5:
    raise RuntimeError(
        f"Se esperaban al menos 5 runs FINISHED y se encontraron "
        f"{len(runs_finalizadas)}. Revisa que train.py haya corrido completo."
    )

mejor_run = runs_finalizadas.iloc[0]
run_id = mejor_run["run_id"]
accuracy = mejor_run["metrics.accuracy"]

print(f"Mejor run: {run_id} (accuracy={accuracy:.4f})")

# Se registra el modelo de esa run en el Model Registry
resultado = mlflow.register_model(
    model_uri=f"runs:/{run_id}/modelo",
    name=NOMBRE_MODELO,
)

# Se le asigna el alias que identifica la version desplegada en produccion
# (MLflow 3.x reemplazo los stages por aliases)
client.set_registered_model_alias(NOMBRE_MODELO, ALIAS, resultado.version)
client.set_model_version_tag(NOMBRE_MODELO, resultado.version, "accuracy", str(accuracy))
client.update_model_version(
    NOMBRE_MODELO,
    resultado.version,
    description=(
        f"Mejor run por accuracy ({accuracy:.4f}) entre 5 configuraciones "
        "de RandomForest. Version servida como @champion en produccion."
    ),
)

print(
    f"Modelo '{NOMBRE_MODELO}' version {resultado.version} "
    f"registrada con alias '@{ALIAS}'."
)

# Se exporta una copia plana y autocontenida del modelo ganador, ya que las
# rutas de artifacts del file-store son absolutas y no funcionan dentro de
# un contenedor Docker
ruta_exportada = mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{NOMBRE_MODELO}@{ALIAS}",
    dst_path="model_champion",
)

print(f"Modelo exportado a: {ruta_exportada}")
