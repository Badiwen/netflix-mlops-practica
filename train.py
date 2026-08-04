import os

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Todos los componentes usarán la misma base de datos local de MLflow.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

DATASET_PATH = "./data/netflix_titles.csv"
EXPERIMENT_NAME = "clasificacion-netflix-rating-iteraciones"


def cargar_y_limpiar_dataset() -> pd.DataFrame:
    """Carga el dataset y aplica el preprocesamiento básico."""
    datos = pd.read_csv(DATASET_PATH)

    # Se eliminan las filas sin variable objetivo.
    datos = datos.dropna(subset=["rating"]).copy()

    # Se usa únicamente el primer género y el primer país.
    datos["listed_in"] = (
        datos["listed_in"]
        .str.split(",")
        .str[0]
        .str.strip()
    )

    datos["country"] = (
        datos["country"]
        .fillna("Unknown")
        .str.split(",")
        .str[0]
        .str.strip()
    )

    return datos


class ClasificadorRatingNetflix(mlflow.pyfunc.PythonModel):
    """
    Empaqueta el Random Forest y sus encoders.

    El modelo recibe variables en formato original y devuelve el rating
    como texto, sin exigir que la API conozca la codificación interna.
    """

    def load_context(self, context):
        self.modelo = joblib.load(context.artifacts["modelo_rf"])
        self.encoder_type = joblib.load(context.artifacts["encoder_type"])
        self.encoder_listed_in = joblib.load(
            context.artifacts["encoder_listed_in"]
        )
        self.encoder_country = joblib.load(
            context.artifacts["encoder_country"]
        )
        self.encoder_rating = joblib.load(
            context.artifacts["encoder_rating"]
        )

    @staticmethod
    def _codificar_seguro(encoder, valor):
        """
        Devuelve -1 cuando llega una categoría no vista durante
        el entrenamiento.
        """
        clases = {
            clase: indice
            for indice, clase in enumerate(encoder.classes_)
        }
        return clases.get(valor, -1)

    def predict(self, context, model_input):
        filas = []

        for _, fila in model_input.iterrows():
            filas.append(
                [
                    self._codificar_seguro(
                        self.encoder_type,
                        fila["type"]
                    ),
                    fila["release_year"],
                    self._codificar_seguro(
                        self.encoder_listed_in,
                        fila["listed_in"]
                    ),
                    self._codificar_seguro(
                        self.encoder_country,
                        fila["country"]
                    ),
                ]
            )

        indices_predichos = self.modelo.predict(filas)

        return self.encoder_rating.inverse_transform(
            indices_predichos
        )


def preparar_datos(datos: pd.DataFrame):
    """Codifica variables y genera los conjuntos train/test."""

    columnas_features = [
        "type",
        "release_year",
        "listed_in",
        "country",
    ]

    X_crudo = datos[columnas_features].copy()

    encoder_type = LabelEncoder().fit(X_crudo["type"])
    encoder_listed_in = LabelEncoder().fit(X_crudo["listed_in"])
    encoder_country = LabelEncoder().fit(X_crudo["country"])
    encoder_rating = LabelEncoder().fit(datos["rating"])

    X = X_crudo.copy()

    X["type"] = encoder_type.transform(X["type"])
    X["listed_in"] = encoder_listed_in.transform(
        X["listed_in"]
    )
    X["country"] = encoder_country.transform(
        X["country"]
    )

    y = encoder_rating.transform(datos["rating"])

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_crudo_train,
        _,
    ) = train_test_split(
        X.values,
        y,
        X_crudo,
        test_size=0.20,
        random_state=42,
    )

    encoders = {
        "encoder_type": encoder_type,
        "encoder_listed_in": encoder_listed_in,
        "encoder_country": encoder_country,
        "encoder_rating": encoder_rating,
    }

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_crudo_train,
        encoders,
    )


def guardar_baseline_y_encoders(
    X_crudo_train: pd.DataFrame,
    encoders: dict,
) -> None:
    """Guarda los artefactos necesarios para monitoreo e inferencia."""

    os.makedirs("data", exist_ok=True)
    X_crudo_train.to_csv(
        "data/baseline_reference.csv",
        index=False,
    )

    os.makedirs("artifacts_encoders", exist_ok=True)

    for nombre, encoder in encoders.items():
        joblib.dump(
            encoder,
            f"artifacts_encoders/{nombre}.joblib",
        )


def entrenar_modelos(
    X_train,
    X_test,
    y_train,
    y_test,
    X_crudo_train,
) -> None:
    """Entrena y registra cinco configuraciones en MLflow."""

    mlflow.set_experiment(EXPERIMENT_NAME)

    configuraciones = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 150, "max_depth": 10},
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 250, "max_depth": 20},
    ]

    for config in configuraciones:
        n_estimators = config["n_estimators"]
        max_depth = config["max_depth"]

        nombre_run = (
            f"random-forest-estimators-{n_estimators}"
            f"-depth-{max_depth}"
        )

        with mlflow.start_run(run_name=nombre_run):
            modelo = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
            )

            modelo.fit(X_train, y_train)

            y_pred = modelo.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_param(
                "n_estimators",
                n_estimators,
            )
            mlflow.log_param(
                "max_depth",
                max_depth,
            )
            mlflow.log_metric(
                "accuracy",
                accuracy,
            )

            joblib.dump(
                modelo,
                "artifacts_encoders/modelo_rf.joblib",
            )

            mlflow.pyfunc.log_model(
                artifact_path="modelo",
                python_model=ClasificadorRatingNetflix(),
                artifacts={
                    "modelo_rf": (
                        "artifacts_encoders/modelo_rf.joblib"
                    ),
                    "encoder_type": (
                        "artifacts_encoders/"
                        "encoder_type.joblib"
                    ),
                    "encoder_listed_in": (
                        "artifacts_encoders/"
                        "encoder_listed_in.joblib"
                    ),
                    "encoder_country": (
                        "artifacts_encoders/"
                        "encoder_country.joblib"
                    ),
                    "encoder_rating": (
                        "artifacts_encoders/"
                        "encoder_rating.joblib"
                    ),
                },
                input_example=X_crudo_train.head(1),
            )

            print(
                f"Árboles: {n_estimators} | "
                f"Profundidad: {max_depth} | "
                f"Accuracy: {accuracy:.4f}"
            )


def main():
    datos = cargar_y_limpiar_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_crudo_train,
        encoders,
    ) = preparar_datos(datos)

    guardar_baseline_y_encoders(
        X_crudo_train,
        encoders,
    )

    entrenar_modelos(
        X_train,
        X_test,
        y_train,
        y_test,
        X_crudo_train,
    )


if __name__ == "__main__":
    main()