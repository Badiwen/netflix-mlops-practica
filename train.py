import pandas as pd

DATASET_PATH = "data/netflix_titles.csv"


def cargar_dataset(ruta: str) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV."""
    return pd.read_csv(ruta)


def mostrar_informacion(df: pd.DataFrame) -> None:
    """Muestra información básica del dataset."""
    print("=" * 50)
    print("DATASET CARGADO CORRECTAMENTE")
    print("=" * 50)

    print(f"\nNúmero de registros: {len(df)}")
    print(f"Número de columnas: {len(df.columns)}")

    print("\nColumnas disponibles:")
    print(df.columns.tolist())

    print("\nPrimeras cinco filas:")
    print(df.head())
    
def analizar_dataset(df: pd.DataFrame) -> None:
    """Realiza un análisis exploratorio básico del dataset."""

    print("\n" + "=" * 50)
    print("ANÁLISIS EXPLORATORIO")
    print("=" * 50)

    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    print("\nDistribución de la variable objetivo (rating):")
    print(df["rating"].value_counts())

    print("\nValores únicos de 'type':")
    print(df["type"].unique())

    print(f"\nCantidad de países distintos: {df['country'].nunique()}")

    print(f"Cantidad de géneros distintos: {df['listed_in'].nunique()}")

def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza la limpieza inicial del dataset."""

    print("\n" + "=" * 50)
    print("LIMPIEZA DEL DATASET")
    print("=" * 50)

    registros_originales = len(df)

    # Eliminar registros sin variable objetivo
    df = df.dropna(subset=["rating"])

    # Mantener únicamente el primer país
    df["country"] = (
        df["country"]
        .fillna("Unknown")
        .str.split(",")
        .str[0]
        .str.strip()
    )

    # Mantener únicamente el primer género
    df["listed_in"] = (
        df["listed_in"]
        .str.split(",")
        .str[0]
        .str.strip()
    )

    print(f"Registros antes de limpiar : {registros_originales}")
    print(f"Registros después          : {len(df)}")

    return df


def main():
    df = cargar_dataset(DATASET_PATH)

    mostrar_informacion(df)

    analizar_dataset(df)

    df = limpiar_dataset(df)


if __name__ == "__main__":
    main()