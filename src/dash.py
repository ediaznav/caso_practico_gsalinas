import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import seaborn as sns
import pandas as pd
import numpy as np


def generar_figura_comparativa(df, columna_fecha, columna_puntos1, columna_puntos2):
    """
        Genera una figura de matplotlib con:

        - Un gráfico de dispersión entre dos columnas numéricas con ajuste LOWESS.
        - Dos gráficos de líneas mostrando la evolución temporal de cada columna.

        Pensada para usarse con Streamlit.

        Parámetros:
        ----------
            df (pd.DataFrame): DataFrame que contiene los datos.
            columna_fecha (str): Nombre de la columna con valores de fecha (de tipo datetime).
            columna_puntos1 (str): Nombre de la primera columna numérica a graficar.
            columna_puntos2 (str): Nombre de la segunda columna numérica a graficar.

        Returns:
            matplotlib.figure.Figure: Figura con los tres gráficos embebidos.
"""


    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1.3, 1, 1], height_ratios=[1, 1],
        wspace=0.35, hspace=0.4
    )

    # Dispersión con ajuste
    ax_dispersion = fig.add_subplot(gs[:, 0])
    sns.regplot(
        x=columna_puntos1,
        y=columna_puntos2,
        data=df,
        ax=ax_dispersion,
        scatter_kws={'alpha': 0.6},
        line_kws={'color': 'black'},
        lowess=True
    )
    ax_dispersion.set_title(f"Dispersión: {columna_puntos1} vs {columna_puntos2}")
    ax_dispersion.set_xlabel(columna_puntos1)
    ax_dispersion.set_ylabel(columna_puntos2)

    # Línea temporal para puntos1
    ax_top = fig.add_subplot(gs[0, 1:])
    ax_top.plot(df[columna_fecha], df[columna_puntos1], color='tab:blue')
    ax_top.set_title(f"Evolución temporal: {columna_puntos1}")
    ax_top.set_ylabel(columna_puntos1)
    ax_top.tick_params(axis='x', rotation=45)

    # Línea temporal para puntos2
    ax_bottom = fig.add_subplot(gs[1, 1:])
    ax_bottom.plot(df[columna_fecha], df[columna_puntos2], color='tab:green')
    ax_bottom.set_title(f"Evolución temporal: {columna_puntos2}")
    ax_bottom.set_ylabel(columna_puntos2)
    ax_bottom.set_xlabel("Fecha")
    ax_bottom.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig



def graficar_arpc_por_region(
    df: pd.DataFrame,
    col_fecha: str,
    col_region: str,
    col_ventas: str,
    colores: list[str] = None
) -> plt.Figure:
    """
    Genera una gráfica de líneas del ARPC promedio por región a lo largo del tiempo.

    Parámetros
    ----------
    df : pd.DataFrame
        Datos de entrada.
    col_fecha : str
        Nombre de la columna de fecha.
    col_region : str
        Nombre de la columna de región.
    col_ventas : str
        Nombre de la columna de ventas por cliente.
    colores : list[str], opcional
        Lista de colores para cada línea.

    Retorna
    -------
    matplotlib.figure.Figure
        Figura con la visualización.
    """
    # Validación y preparación
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha])

    # Crear tabla dinámica
    tabla = df.pivot_table(
        index=col_fecha,
        columns=col_region,
        values=col_ventas,
        aggfunc=np.mean
    ).sort_index()

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 5))
    tabla.plot(ax=ax, color=colores)
    ax.set_title("ARPC promedio por región a lo largo del tiempo")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Ventas por cliente (ARPC)")
    ax.legend(title=col_region)
    ax.grid(True)

    return fig



def graficar_histograma_con_percentiles(df: pd.DataFrame,
                                        columna: str,
                                        bins: int = 20,
                                        color: str = "steelblue") -> tuple[plt.Figure, float, float]:
    """
    Genera un histograma de una variable numérica con líneas para los percentiles 10 y 90.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene los datos.
    columna : str
        Nombre de la columna numérica.
    bins : int, opcional
        Número de barras en el histograma (default: 20).
    color : str, opcional
        Color del histograma (default: "darkblue").
    
    Retorna
    -------
    fig : matplotlib.figure.Figure
        Figura del histograma.
    p10 : float
        Valor del percentil 10.
    p90 : float
        Valor del percentil 90.
    """

    # Calcular percentiles
    p10 = df[columna].quantile(0.10)
    p90 = df[columna].quantile(0.90)

    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 4))
    df[columna].hist(bins=bins, color=color, alpha=0.7, ax=ax)

    # Agregar líneas de percentil
    ax.axvline(p10, color='red', linestyle='--', label=f'P10: {p10:.2f}')
    ax.axvline(p90, color='green', linestyle='--', label=f'P90: {p90:.2f}')

    # Etiquetas y leyenda
    ax.set_title(f"Distribución de {columna}")
    ax.set_xlabel(columna)
    ax.set_ylabel("Frecuencia")
    ax.legend()

    return fig


def graficar_yoy_mensual(df: pd.DataFrame, periods: int = 12, label: str="YoY") -> plt.Figure:
    """
    Genera un gráfico de barras del cambio porcentual YoY mensual para ventas y clientes.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas: 'fecha', 'ventas', 'clientes'.
        Puede contener múltiples registros por mes.

    Retorna
    -------
    matplotlib.figure.Figure
        Figura de barras con % de cambio YoY por mes.
    """
    # Asegurar tipos
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Agrupar por mes y sumar valores
    df_mensual = df.groupby(pd.Grouper(key='fecha', freq='MS'))[['ventas', 'clientes']].sum()

    # Calcular variación YoY (12 meses atrás)
    df_yoy = df_mensual.pct_change(periods=periods) * 100
    df_yoy = df_yoy.dropna().round(2)

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 15  # días para separación visual

    ax.bar(df_yoy.index - pd.Timedelta(days=width),
           df_yoy['ventas'],
           width=10, label='Ventas YoY (%)',
            align='edge', color='tab:blue')

    ax.bar(df_yoy.index + pd.Timedelta(days=width),
           df_yoy['clientes'],
           width=-10, label='Clientes YoY (%)', 
           align='edge', color='tab:green')

    # Formato
    ax.set_title(f"Crecimiento porcentual {label} mensual: Ventas vs Clientes")
    ax.set_ylabel("Variación porcentual (%)")
    ax.set_xlabel("Mes")
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Etiquetas de mes
    ax.set_xticks(df_yoy.index)
    ax.set_xticklabels([d.strftime('%Y-%m') for d in df_yoy.index], rotation=45)

    return fig


def pronosticar_ventas_hw(df: pd.DataFrame,
                          columna_fecha: str = "fecha",
                          columna_valor: str = "ventas",
                          pasos: int = 3) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Realiza un pronóstico usando el modelo Holt-Winters (ETS) para los siguientes períodos.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas de fecha y valores numéricos.
    columna_fecha : str
        Nombre de la columna con fechas (datetime o convertible).
    columna_valor : str
        Nombre de la columna con los valores a pronosticar.
    pasos : int
        Número de períodos a pronosticar (default: 3).

    Retorna
    -------
    fig : matplotlib.figure.Figure
        Figura con datos históricos, pronóstico y bandas de confianza.
    df_forecast : pd.DataFrame
        DataFrame con pronósticos y bandas de confianza (al 90%).
    """
    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])
    df = df.set_index(columna_fecha).sort_index()

    # Asegurar frecuencia mensual
    serie = df[columna_valor].asfreq('MS')

    # Ajustar modelo Holt-Winters
    modelo = ExponentialSmoothing(
        serie,
        trend='add',
        seasonal='add',
        initialization_method='estimated'
    ).fit()

    # Pronóstico
    predicciones = modelo.forecast(steps=pasos)

    # Estimación básica de intervalos de confianza (suponiendo varianza constante normal)
    error_std = np.std(modelo.resid, ddof=1)
    z_90 = 1.645  # para 90% CI

    lower = predicciones - z_90 * error_std
    upper = predicciones + z_90 * error_std

    # Construir DataFrame de salida
    df_forecast = pd.DataFrame({
        "pronóstico": predicciones,
        "confianza_90_inf": lower,
        "confianza_90_sup": upper
    })

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 5))
    serie.plot(ax=ax, label='Histórico', color='blue')
    predicciones.plot(ax=ax, label='Pronóstico', color='green')
    ax.fill_between(df_forecast.index, lower, upper,
                    color='lightgreen', alpha=0.4, label='Intervalo 90%')
    ax.set_title(f"Pronóstico Holt-Winters para '{columna_valor}'")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(columna_valor.capitalize())
    ax.grid(True)
    ax.legend()

    return fig, df_forecast.clip(0)
