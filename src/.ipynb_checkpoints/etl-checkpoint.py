# etl.py 

# Script con funciones auxiliares para el proceso ETL que se aplica a los datos. 
# Incluye las cargas y guardados de archivos en diferentes formatos y limpieza de
# los mismos. 

### Librerias necesarias en estas funciones 

import pandas as pd
import numpy as np
import warnings
import unicodedata

from pyspark.sql import functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType
import os


from pyspark.sql.types import (
    StringType, IntegerType, FloatType, DoubleType,
    BooleanType, DateType, TimestampType, StructType, StructField
)

### --------------------------------------------------------------------------
### CARGA Y ALMACENAMIENTO DE ARCHIVOS---------------------------------------- 

def cargar_data_csv(file_path: str) -> pd.DataFrame:
    """
    Carga un archivo CSV y devuelve las primeras 5 filas.

    Parámetros:
    ----------
    file_path : str
        Ruta al archivo CSV a cargar.

    Regresa:
    -------
    pd.DataFrame
        Primeras 5 filas del DataFrame si el archivo se encuentra,
        de lo contrario imprime un mensaje de error.
    """
    try:
        # Intenta leer el archivo CSV
        df = pd.read_csv(file_path)
        return df.head(5)
    except FileNotFoundError:
        # Manejo de error si el archivo no existe en la ruta especificada
        print(f"El archivo no está en este path: {file_path}")
        print("Busca en otro lugar; por lo pronto no podemos proceder.")


def cargar_data_excel(file_path: str) -> pd.DataFrame:
    """
    Carga un archivo EXCEL y devuelve las primeras 5 filas.

    Parámetros:
    ----------
    file_path : str
        Ruta al archivo EXCEL a cargar.

    Regresa:
    -------
    pd.DataFrame
        Primeras 5 filas del DataFrame si el archivo se encuentra,
        de lo contrario imprime un mensaje de error.
    """
    try:
        # Intenta leer el archivo EXCEL
        df = pd.read_excel(file_path)
        return df.head(5)
    except FileNotFoundError:
        # Manejo de error si el archivo no existe en la ruta especificada
        print(f"El archivo no está en este path: {file_path}")
        print("Busca en otro lugar; por lo pronto no podemos proceder.")

def resumen_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen por columna de un DataFrame, incluyendo:
    tipo de dato, cantidad de valores únicos, nulos y total de registros.

    Parámetros:
    ----------
    df : pd.DataFrame
        DataFrame de entrada que se desea analizar.

    Regresa:
    -------
    pd.DataFrame
        DataFrame con resumen por columna.
    """
    try:
        # Validación de tipo
        if not isinstance(df, pd.DataFrame):
            raise TypeError("La entrada debe ser un DataFrame de pandas.")

        # Construcción del resumen por columna
        resumen = pd.DataFrame({
            'tipo_dato': df.dtypes,
            'valores_unicos': df.nunique(),
            'nulos': df.isnull().sum(),
            'total': len(df)
        })

        # Asegurarse de que los tipos de datos se conviertan a string 
        resumen['tipo_dato'] = resumen['tipo_dato'].astype(str)

        return resumen

    except Exception as e:
        print("Ocurrió un error al generar el resumen del DataFrame:")
        print(str(e))
        return pd.DataFrame()  # Regresa un DataFrame vacío en caso de error



def generar_schema_spark(df_esquema: pd.DataFrame, tolerante: bool = True) -> str:
    """
    Genera un esquema Spark (StructType) a partir de un DataFrame con nombres
    y tipos de columnas.
    Soporta todos los tipos comunes de pandas y puede operar en modo tolerante.

    Parámetros:
    ----------
    df_esquema : pd.DataFrame
        DataFrame con las columnas:
            - 'columna': nombre de la columna
            - 'tipo': dtype pandas o string representando el tipo

    tolerante : bool, default=True
        Si es True, convierte tipos desconocidos a StringType con advertencia.
        Si es False, lanza error si encuentra tipos no soportados.

    Regresa:
    -------
    str
        Texto del esquema StructType de PySpark.
    """
    try:
        if not isinstance(df_esquema, pd.DataFrame):
            raise TypeError("La entrada debe ser un DataFrame de pandas.")
        if not {'columna', 'tipo'}.issubset(df_esquema.columns):
            raise ValueError("El DataFrame debe contener las columnas 'columna' y 'tipo'.")

        # Mapeo extendido de tipos pandas a tipos Spark
        tipo_spark = {
            'object': 'StringType()',
            'string': 'StringType()',
            'str': 'StringType()',
            'category': 'StringType()',
            'int64': 'IntegerType()',
            'int32': 'IntegerType()',
            'int': 'IntegerType()',
            'int16': 'IntegerType()',
            'int8': 'IntegerType()',
            'float64': 'DoubleType()',
            'float32': 'FloatType()',
            'float': 'DoubleType()',
            'bool': 'BooleanType()',
            'boolean': 'BooleanType()',
            'datetime64[ns]': 'TimestampType()',
            'datetime64': 'TimestampType()',
            'timedelta[ns]': 'StringType()', 
            'date': 'DateType()',
            'timestamp': 'TimestampType()'
        }

        lineas = []
        for _, row in df_esquema.iterrows():
            nombre = str(row['columna']).strip()
            tipo_raw = row['tipo']

            # Normaliza el tipo a texto en minúsculas
            if isinstance(tipo_raw, (np.dtype, pd.api.extensions.ExtensionDtype)):
                tipo_normalizado = str(tipo_raw).lower()
            else:
                tipo_normalizado = str(tipo_raw).strip().lower()

            tipo_spark_str = tipo_spark.get(tipo_normalizado)

            if not tipo_spark_str:
                if tolerante:
                    tipo_spark_str = 'StringType()'
                    warnings.warn(
                        f"Tipo de dato no reconocido '{tipo_raw}' en columna '{nombre}', asignado como StringType()."
                    )
                else:
                    raise ValueError(f"Tipo de dato no reconocido: {tipo_raw} en columna {nombre}")

            lineas.append(f"    StructField('{nombre}', {tipo_spark_str}, True)")

        schema_str = "StructType([\n" + ",\n".join(lineas) + "\n])"
        return schema_str

    except Exception as e:
        print("Ocurrió un error al generar el esquema para Spark:")
        print(str(e))
        return ""


def guardar_data_csv(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """
    Guarda un DataFrame en un archivo CSV en la ruta especificada.

    Parámetros:
    ----------
    df : pd.DataFrame
        DataFrame que se desea guardar.
    
    file_path : str
        Ruta de destino para el archivo CSV (incluye nombre y extensión).
    
    index : bool, default=False
        Si se desea incluir el índice del DataFrame en el archivo CSV.
    
    Regresa:
    -------
    None
        Solo guarda el archivo o imprime un error si falla.
    """
    try:
        # Intenta guardar el DataFrame en CSV
        df.to_csv(file_path, index=index)
        print(f"Archivo guardado exitosamente en: {file_path}")

    except PermissionError:
        # Error si no hay permisos de escritura
        print(f"No tienes permisos para escribir en: {file_path}")
    
    except Exception as e:
        # Captura cualquier otro error inesperado
        print(f"Ocurrió un error al guardar el archivo CSV: {e}")


### --------------------------------------------------------------------------
### EXPLORACION Y LIMPIEZA BASICA DE LOS DATOS -------------------------------


import pandas as pd
import logging
from datetime import datetime


# Configura el logger para escribir en un archivo
logging.basicConfig(
    filename='../gsalinas_caso/logs/calidad_datos.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def log_result(column_name: str, test_name: str, passed: bool, details: str = "") -> None:
    """
    Registra en el log el resultado de una prueba de calidad.

    Parámetros
    ----------
    column_name : str
        Nombre de la columna evaluada.
    test_name : str
        Nombre de la prueba realizada.
    passed : bool
        Indica si la prueba fue superada.
    details : str, optional
        Información adicional del resultado.

    Regresa
    -------
    None
    """
    status = "PASÓ" if passed else "FALLÓ"
    message = f"[{column_name}] - {test_name}: {status}. {details}"
    if passed:
        logging.info(message)
    else:
        logging.warning(message)


## TESTS PARA FECHAS 
        
def test_datetime_type(df: pd.DataFrame, column: str) -> bool:
    """
    Verifica si la columna es de tipo datetime64 o similar.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.
    column : str
        Nombre de la columna a validar.

    Regresa
    -------
    bool
        True si la columna tiene tipo datetime válido, False en caso contrario.
    """
    passed = pd.api.types.is_datetime64_any_dtype(df[column])
    log_result(column, "Tipo datetime64[ns]", passed)
    return passed


def test_nulls_in_datetime(df: pd.DataFrame, column: str) -> bool:
    """
    Verifica si existen valores nulos en la columna.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.
    column : str
        Nombre de la columna a validar.

    Regresa
    -------
    bool
        True si no hay valores nulos, False si existen.
    """
    nulls = df[column].isnull().sum()
    passed = nulls == 0
    log_result(column, "Valores nulos", passed, f"Nulos encontrados: {nulls}")
    return passed


def test_future_dates(df: pd.DataFrame, column: str) -> bool:
    """
    Verifica si hay fechas futuras en la columna.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.
    column : str
        Nombre de la columna de fechas.

    Regresa
    -------
    bool
        True si no hay fechas futuras, False si las hay.
    """
    now = pd.Timestamp(datetime.now())
    future_dates = df[df[column] > now]
    passed = future_dates.empty
    log_result(column, "Fechas futuras", passed, f"{len(future_dates)} registros futuros encontrados")
    return passed


def test_min_date(df: pd.DataFrame, column: str, min_date: str = "2000-01-01") -> bool:
    """
    Verifica que no existan fechas anteriores a un umbral mínimo.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.
    column : str
        Nombre de la columna de fechas.
    min_date : str, default="2000-01-01"
        Umbral mínimo aceptable para las fechas.

    Regresa
    -------
    bool
        True si todas las fechas están por encima del mínimo, False en caso contrario.
    """
    min_limit = pd.to_datetime(min_date)
    old_dates = df[df[column] < min_limit]
    passed = old_dates.empty
    log_result(column, f"Fechas antes de {min_date}", passed, f"{len(old_dates)} registros antiguos encontrados")
    return passed


def test_monotonic_increasing(df: pd.DataFrame, column: str) -> bool:
    """
    Verifica que las fechas estén ordenadas en forma creciente.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.
    column : str
        Nombre de la columna de fechas.

    Regresa
    -------
    bool
        True si la columna es monótonamente creciente, False en caso contrario.
    """
    passed = df[column].is_monotonic_increasing
    log_result(column, "Orden creciente (monotonicidad)", passed)
    return passed


def run_datetime_quality_tests(df: pd.DataFrame, column: str) -> None:
    """
    Ejecuta todas las pruebas de calidad sobre una columna datetime.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos.
    column : str
        Nombre de la columna datetime a evaluar.

    Regresa
    -------
    None
    """
    logging.info(f"Iniciando pruebas de calidad para columna '{column}'")
    try:
        if test_datetime_type(df, column):
            test_nulls_in_datetime(df, column)
            test_future_dates(df, column)
            test_min_date(df, column)
            test_monotonic_increasing(df, column)
        else:
            log_result(column, "Pruebas adicionales omitidas", False, "Tipo de dato no válido para datetime")
    except Exception as e:
        logging.error(f"Error crítico al ejecutar pruebas de calidad sobre '{column}': {str(e)}")

## TESTS PARA TIPO OBJECT 

def test_object_type(df: pd.DataFrame, column: str) -> bool:
    """
    Verifica si la columna es de tipo object o string.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.
    column : str
        Nombre de la columna a validar.

    Regresa
    -------
    bool
        True si es tipo object, False en caso contrario.
    """
    passed = df[column].dtype == object or pd.api.types.is_string_dtype(df[column])
    log_result(column, "Tipo object/string", passed)
    return passed


def test_nulls_in_object(df: pd.DataFrame, column: str) -> bool:
    """
    Verifica si existen valores nulos en columnas tipo object.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.
    column : str
        Columna tipo string a validar.

    Regresa
    -------
    bool
        True si no hay nulos, False si existen.
    """
    nulls = df[column].isnull().sum()
    passed = nulls == 0
    log_result(column, "Valores nulos", passed, f"Nulos encontrados: {nulls}")
    return passed


def test_object_cardinality(df: pd.DataFrame, column: str, max_unique: int = 100) -> bool:
    """
    Verifica que el número de valores únicos en una columna object no supere un umbral.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.
    column : str
        Columna de texto a validar.
    max_unique : int, default=100
        Máximo número esperado de valores únicos.

    Regresa
    -------
    bool
        True si está dentro del límite, False si lo supera.
    """
    uniques = df[column].nunique(dropna=True)
    passed = uniques <= max_unique
    log_result(column, f"Cardinalidad máxima ({max_unique})", passed, f"Únicos encontrados: {uniques}")
    return passed


def run_object_quality_tests(df: pd.DataFrame, column: str) -> None:
    """
    Ejecuta pruebas de calidad para columnas tipo object o string.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos.
    column : str
        Columna tipo object a evaluar.

    Regresa
    -------
    None
    """
    logging.info(f"Iniciando pruebas de calidad para columna '{column}' (tipo object)")
    try:
        if test_object_type(df, column):
            test_nulls_in_object(df, column)
            test_object_cardinality(df, column)
        else:
            log_result(column, "Pruebas adicionales omitidas", False, "Tipo de dato no válido para object/string")
    except Exception as e:
        logging.error(f"Error crítico al ejecutar pruebas de calidad sobre '{column}': {str(e)}")


## TEST PARA TIPO INT

def test_integer_type(df: pd.DataFrame, column: str) -> bool:
    """
    Verifica si una columna es de tipo entero (int32, int64 o nullable Int).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.
    column : str
        Columna a validar.

    Regresa
    -------
    bool
        True si la columna es de tipo entero, False si no lo es.
    """
    passed = pd.api.types.is_integer_dtype(df[column])
    log_result(column, "Tipo entero (int)", passed)
    return passed


def test_nulls_in_integer(df: pd.DataFrame, column: str) -> bool:
    """
    Verifica si hay nulos en columnas numéricas enteras.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.
    column : str
        Columna tipo entero a validar.

    Regresa
    -------
    bool
        True si no hay nulos, False si existen.
    """
    nulls = df[column].isnull().sum()
    passed = nulls == 0
    log_result(column, "Valores nulos", passed, f"Nulos encontrados: {nulls}")
    return passed


def test_integer_range(df: pd.DataFrame, column: str, min_val: int = 0, max_val: int = 10_000_000) -> bool:
    """
    Verifica que los valores estén dentro de un rango válido.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.
    column : str
        Columna tipo entero a validar.
    min_val : int, default=0
        Valor mínimo permitido.
    max_val : int, default=1_000_000
        Valor máximo permitido.

    Regresa
    -------
    bool
        True si todos los valores están en el rango, False si hay valores fuera de rango.
    """
    out_of_range = df[~df[column].between(min_val, max_val)]
    passed = out_of_range.empty
    log_result(
        column,
        f"Rango entre {min_val} y {max_val}",
        passed,
        f"{out_of_range.shape[0]} registros fuera de rango"
    )
    return passed


def run_integer_quality_tests(df: pd.DataFrame, column: str) -> None:
    """
    Ejecuta un conjunto de pruebas de calidad para columnas tipo entero.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos.
    column : str
        Columna tipo entero a evaluar.

    Regresa
    -------
    None
    """
    logging.info(f"Iniciando pruebas de calidad para columna '{column}' (tipo entero)")
    try:
        if test_integer_type(df, column):
            test_nulls_in_integer(df, column)
            test_integer_range(df, column)
        else:
            log_result(column, "Pruebas adicionales omitidas", False, "Tipo de dato no válido para entero")
    except Exception as e:
        logging.error(f"Error crítico al ejecutar pruebas de calidad sobre '{column}': {str(e)}")


### TRANSFORMACIONES Y REGLAS EXPERTAS

def formatear_columnas_string(df, columnas=None):
    """
    Limpia y formatea columnas de texto para compatibilidad con Spark.

    Operaciones aplicadas:
    - Convierte a string
    - Elimina acentos (NFKD + ascii)
    - Elimina espacios extra (leading/trailing)
    - Reemplaza espacios internos por guiones bajos
    - Convierte a minúsculas

    Parámetros:
    ------------
    df : pd.DataFrame
        El DataFrame a procesar.

    columnas : list or None
        Lista de columnas a limpiar. Si es None, se aplicará solo a columnas de tipo object o string.

    Regresa:
    --------
    pd.DataFrame
        DataFrame con las columnas de texto formateadas.
    """
    df = df.copy()

    # Detectar columnas de texto si no se especifican
    if columnas is None:
        columnas = df.select_dtypes(include='object').columns.tolist()

    for col in columnas:
        df[col] = (
            df[col]
            .astype(str)
            .apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii'))  # quita acentos
            .str.strip()             # elimina espacios al inicio/final
            .str.replace(r'\s+', '_', regex=True)  # reemplaza espacios por guiones bajos
            .str.replace(r"[^\w\d\-]", "", regex = True) # elimina caracteres no seguros 
            .str.lower()             # todo a minúsculas
        )

    return df


def rellenar_nulos_numericos(df: pd.DataFrame, columna: str, metodo: str = 'media') -> pd.DataFrame:
    """
    Rellena los valores nulos de una columna numérica usando media o mediana.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene los datos.

    columna : str
        Nombre de la columna donde se desea aplicar el relleno.

    metodo : str, default='media'
        Método de imputación: 'media' para promedio, 'mediana' para valor central.

    Regresa:
    -------
    pd.DataFrame
        DataFrame con los valores nulos de la columna reemplazados.
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[columna]):
        raise TypeError(f"La columna '{columna}' debe ser numérica.")

    if metodo == 'media':
        valor = df[columna].mean()
    elif metodo == 'mediana':
        valor = df[columna].median()
    else:
        raise ValueError("El parámetro 'metodo' debe ser 'media' o 'mediana'.")

    df[columna] = df[columna].fillna(valor)
    return df


def agregar_columnas_temporales(df: DataFrame, columna_fecha: str) -> DataFrame:
    """
    Agrega columnas derivadas de una columna tipo fecha: año, mes, día y año-mes.

    Parámetros
    ----------
    df : pyspark.sql.DataFrame
        DataFrame de entrada que contiene la columna de fecha.

    columna_fecha : str
        Nombre de la columna tipo date o timestamp.

    Regresa
    -------
    pyspark.sql.DataFrame
        DataFrame con las nuevas columnas: 'anio', 'mes', 'dia', 'anio_mes'.
    """
    return (df
        .withColumn("anio", F.year(F.col(columna_fecha))) 
        .withColumn("mes", F.month(F.col(columna_fecha))) 
        .withColumn("dia", F.dayofmonth(F.col(columna_fecha))) 
        .withColumn("anio_mes", F.date_format(F.col(columna_fecha), "yyyy-MM")))



def cargar_csv_con_schema(spark: SparkSession,
                          ruta_archivo: str,
                          schema: StructType,
                          formato: str = "csv") -> DataFrame:
    """
    Carga un archivo CSV en un DataFrame de Spark usando un esquema definido.

    Parámetros
    ----------
    spark : SparkSession
        Sesión activa de Spark.

    ruta_archivo : str
        Ruta completa del archivo CSV a cargar.

    schema : StructType
        Esquema de datos definido para la lectura del archivo.

    formato : str, default='csv'
        Formato del archivo. Por defecto se asume CSV.

    Regresa
    -------
    DataFrame
        DataFrame de Spark con los datos cargados.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta especificada.

    Exception
        Si ocurre otro tipo de error al intentar leer el archivo.
    """
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"El archivo no está en este path: {ruta_archivo}")

    try:
        df = (spark.read.format(formato)
              .option("header", "true")
              .option("inferSchema", "false")
              .schema(schema)
              .load(ruta_archivo))
        print(f"Archivo cargado con éxito: {os.path.basename(ruta_archivo)}")
        return df

    except Exception as e:
        print("Ocurrió un error al cargar el archivo.")
        print(str(e))
        raise
