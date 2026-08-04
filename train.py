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


def main():
    df = cargar_dataset(DATASET_PATH)
    mostrar_informacion(df)


if __name__ == "__main__":
    main()