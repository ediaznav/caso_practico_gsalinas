# IMPORTAMOS LAS LIBRERIAS NECESARIAS

# UI
import streamlit as st

# Manejo de datos y analitica
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib
import os
from src.dash import *

# ----------------------------
# CONFIGURACIÓN GENERAL Y PARAMETROS
# ----------------------------
st.set_page_config(
    page_title="Global Retail Corp.",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_CLEAN_DIRECTORY = "./data/clean/"

VENTAS_FILE = "data_agg_ventas.csv"
CLIENTES_FILE = "data_agg_clientes.csv"
TOP_FILE = "data_top_productos.csv"
CLEAN_FILE = "data_clean.csv"

# ------------------------------
# Cargar archivo CSV desde backend

@st.cache_data
def cargar_dataset(nombre_archivo: str) -> pd.DataFrame:
    ruta = os.path.join( nombre_archivo)
    return pd.read_csv(ruta)

# ------------------------------
# Cargar modelo desde backend

@st.cache_resource
def cargar_modelo(nombre_modelo: str):
    ruta = os.path.join("models", nombre_modelo)
    return joblib.load(ruta)

# ----------------------------
# ESTILOS PERSONALIZADOS

st.markdown("""
    <style>
        body {
            background-color: #0f1117;
            color: #c7d4e4;
        }
        .main {
            background-color: #0f1117;
        }
        .block-container {
            padding: 2rem;
        }
        header, .st-bq, .st-bq div {
            background-color: #0f1117 !important;
            color: #c7d4e4 !important;
        }
        .stSidebar {
            background-color: #10141f;
        }
    </style>
""", unsafe_allow_html=True)

plt.style.use("ggplot")

# ----------------------------
# MENÚ LATERAL

seccion = st.sidebar.selectbox(
     "Selecciona sección",
    ["Análisis Exploratorios", "Analítica Predictiva", "Gobierno de Datos"]
)

# ----------------------------
# ENCABEZADO 

st.markdown("""
    <div style='background-color: #0f1117; padding: 1rem; text-align: center; border-radius: 0.25rem'>
        <h1 style='color: #4c8bf5; font-family: sans-serif; margin: 0;'>🌐 Global Retail Corp. </h1>
        <p style='color: #c7d4e4; margin-top: 4px;'>Dashboard con solucions analíticas para Ventas mesnuales.</p>
    </div>
""", unsafe_allow_html=True)


# ----------------------------
# EDA SECTION
# ----------------------------
if seccion == "Análisis Exploratorios":
    st.title("Análisis Exploratorios de Datos (EDA)")

    st.markdown("### Estado actual de las tiendas")

    df = cargar_dataset(DATA_CLEAN_DIRECTORY + CLEAN_FILE)

    # Filtrar por una categoría (ej. columna de texto o etiquetas)
    opciones_region = df['region'].unique()
    opciones_producto = df['producto'].unique()
    
    cat_reg_selec = st.multiselect("Selecciona la región",
                                   placeholder = "Regiones",
                                    options = opciones_region)
    cat_prod_selec = st.multiselect("Selecciona la producto",
                                    placeholder = "Productos",
                                     options = opciones_producto)

    df = (df.filter(["fecha", "region", "producto", 
                            "ventas","clientes"]))
    
    if cat_reg_selec:
        df = df[df.region.isin(cat_reg_selec)]

    if cat_prod_selec:
        df = df[df.producto.isin(cat_prod_selec)]

    #df.columns = [x.title() for x in df.columns.tolist()]

    st.markdown("En la siguiente tabla podemos ver los datos correspondientes a los filtros seleccionados. ")
    
    st.dataframe((df.filter(["fecha", "region", "producto", 
                            "ventas","clientes"])
                    
                ), height = 200, hide_index = True
                )

    st.markdown("Se presentan las tendencias para los clientes y las ventas en las regiones y productos seleccionados.")

    auxiliar_pivot_df = (df.pivot_table(index = "fecha",
                    values = ["ventas", "clientes"],
                    aggfunc = np.sum).reset_index().rename(
                        columns = {"index":"fecha"}
                    ))
    
    #auxiliar_pivot_df.columns = [x.title() for x in auxiliar_pivot_df.columns.tolist()]

    # Figura con tendencias 
    figura = generar_figura_comparativa(auxiliar_pivot_df
                     , "fecha", "clientes", "ventas")
    st.pyplot(figura)


    ### Dominio de productos y tiendas ------

    st.markdown("Como referencia de desempeño, vemos cuales son los productos que están teniendo mayor impacto en las ventas de cada región. ")
    df_top = cargar_dataset(DATA_CLEAN_DIRECTORY + TOP_FILE)
    df_top.rename(
        columns = {"ventas_totales":"ventas 2024"}, 
        inplace = True
    )

    df_ventas = pd.read_csv( DATA_CLEAN_DIRECTORY + VENTAS_FILE)


    # Crear dos columnas: izquierda (dataframe) y derecha (plot)
    col1, col2 = st.columns([1, 1]) 

    with col1:
        st.subheader("Productos Lideres")
        st.markdown("Principal producto en términos de ventas anuales para el 2024 por región. ")
        st.dataframe((df_top
                ),  hide_index = True
                )

    with col2:
        df_ventas = (df_ventas.pivot_table(index = "region", values = "ventas_totales", 
                      aggfunc= np.sum, columns = "producto")/1e6).round(3)

        st.subheader("Ventas por producto y región")
        st.markdown("Unidades vendidas durante el 2024 (en Millones) por región y producto.")
        styled_df = df_ventas.style.background_gradient(cmap='Blues')

        st.write(styled_df)


    st.markdown("Es importante, con ayuda de la sigueinte tabla, entender el porcentaje de los clientes que tiene cada zona y cada producto para saber las preferencias del consumidor y entender que productos/zonas estan ganando/perdiendo relevancia general en terminos de dominio de las ventas internas. ")

    df_clientes = pd.read_csv( DATA_CLEAN_DIRECTORY + CLIENTES_FILE)


    pivote_clientes_df = (df_clientes.pivot_table(index = "region", columns = "producto", 
                        values = "clientes_cuota", aggfunc = np.sum)*100).round(2)

    # Agregar columna "Total" por fila (suma horizontal)
    pivote_clientes_df["Total"] = pivote_clientes_df.sum(axis=1)

    # Agregar fila "Total" por columna (suma vertical)
    pivote_clientes_df.loc["Total"] = pivote_clientes_df.sum(axis=0)

    columnas_sin_total = pivote_clientes_df.columns.difference(["Total"])
    filas_sin_total = pivote_clientes_df.index.difference(["Total"])

    styled_df = pivote_clientes_df.style.background_gradient(
        cmap='Blues',
        subset=pd.IndexSlice[filas_sin_total, columnas_sin_total]
    )

    st.write(styled_df)
    
    

    st.markdown("### Comportamiento Histórico y KPIs")

    st.markdown("Entendiendo el comportamiento del cambio YoY tanto de los clientes como de las ventas nos permite entender en que momentos se esta teniendo un cambio con respecto a la misma temporada en el año previo y entender en donde se esta teniendo un crecimiento o pérdida.")
    fig = graficar_yoy_mensual(df.filter(["fecha","ventas","clientes"]))
    st.pyplot(fig)

    # Crear dos columnas: izquierda (dataframe) y derecha (plot)
    col1, col2 = st.columns([1, 1]) 

    with col1:
        st.markdown("Las ventas por cliente, también conocidas como Average Return Per Cliente (AVPC), son un indicador de que tan compradores son nuestros clientes en una determinada región, o para algún producto. ")

        df["ventas_p_cliente"] = df["ventas"]/df["clientes"]

    
        fig = graficar_arpc_por_region(df,
                "fecha","region","ventas",
                colores = ["darkblue", "green", "black", "lightblue"]
            )
        
        st.pyplot(fig)

    with col2:

        st.markdown("La distribución de ventas por cliente nos permite conocer el volumen de compra que tienen los clientes observados, y también, nos permite identificar la existencia de 'super-clientes'. ")
           
        fig = graficar_histograma_con_percentiles(df,"ventas_p_cliente")
        st.pyplot(fig)


    fig = graficar_yoy_mensual(df.filter(["fecha","ventas","clientes"]),1, "MoM")
    st.pyplot(fig)





# ----------------------------
# PREDICTIVE ANALYTICS SECTION
# ----------------------------
elif seccion == "Analítica Predictiva":
    st.title("Analítica Predictiva")

    st.markdown("### Predicción de Ventas y Clientes")
    st.markdown("Selecciona los productos y regiones de interés para generar un pronóstico para los siguientes 3 meses. ")

    df = cargar_dataset(DATA_CLEAN_DIRECTORY + CLEAN_FILE)

    # Filtrar por una categoría (ej. columna de texto o etiquetas)
    opciones_region = df['region'].unique()
    opciones_producto = df['producto'].unique()
    
    cat_reg_selec = st.multiselect("Selecciona la región",
                                   placeholder = "Regiones",
                                    options = opciones_region)
    cat_prod_selec = st.multiselect("Selecciona la producto",
                                    placeholder = "Productos",
                                     options = opciones_producto)

    df = (df.filter(["fecha", "region", "producto", 
                            "ventas","clientes"]))
    
    if cat_reg_selec:
        df = df[df.region.isin(cat_reg_selec)]

    if cat_prod_selec:
        df = df[df.producto.isin(cat_prod_selec)]

    df_ventas = df.pivot_table(index = "fecha", values = "ventas", 
                         aggfunc = np.sum).reset_index()
    
    df_clientes = df.pivot_table(index = "fecha", values = "clientes", 
                         aggfunc = np.sum).reset_index()
    
    #st.markdown("")

    col1, col2 = st.columns([1, 1]) 
    
    with col1:
        fig, df_pred = pronosticar_ventas_hw(df_ventas,
                        columna_fecha="fecha", columna_valor="ventas")
        
        st.dataframe(df_pred.astype("int").clip(0),  hide_index = True)
        st.pyplot(fig)

    with col2:
        fig, df_pred = pronosticar_ventas_hw(df_clientes,
                        columna_fecha="fecha", columna_valor="clientes")
        
        st.dataframe(df_pred.astype("int").clip(0),  hide_index = True)
        st.pyplot(fig)


    st.markdown("### Clusterización de Actividad")
    st.markdown("Se presentan las observaciones de regiones y productos a lo largo del tiempo y el cluster de desempeño (en términos de clientes y ventas) al que pertenecen en cada mes. ")

    col1, col2 = st.columns([1, 1]) 

    cluster_df = (pd.read_csv(DATA_CLEAN_DIRECTORY + "clusters.csv"))
    
    if cat_reg_selec:
        cluster_df = cluster_df[cluster_df.region.isin(cat_reg_selec)]

    if cat_prod_selec:
        cluster_df = cluster_df[cluster_df.producto.isin(cat_prod_selec)]
    
    with col1:
        st.dataframe(cluster_df.filter(["anio_mes", "region", "producto", "cluster"]),
                    height = 360, hide_index = True)

    with col2:
        fig,ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=cluster_df, x="ventas",y= "clientes",
                             hue="cluster", palette="Blues")
        ax.set_title(f"KMeans Clustering")                    
        st.pyplot(fig)


# Simulación de predicción

    # Selección de modelo
    #modelo_nombre = st.selectbox("Selecciona modelo", ["modelo_rf.pkl"])
    #modelo = cargar_modelo(modelo_nombre)


# ----------------------------
# DATA GOVERNANCE
# ----------------------------
elif seccion == "Gobierno de Datos":
    st.title("Gobierno de Datos")
    
    st.markdown("Registro básico de gobernanza de datos con respecto a su calidad y completez. Así como el linaje y origen de los mismos.")

    resumen_data = pd.read_csv(DATA_CLEAN_DIRECTORY + "resumen_datos.csv")

    def background_por_benchmark(val):
        color = 'lightgreen' if val < 0.05 else 'salmon'
        return f'background-color: {color}'

    resumen_data= resumen_data.style.applymap(background_por_benchmark,
                                             subset=['pc_nulos'])


    st.write(resumen_data)
        

    df_gobierno_datos = pd.DataFrame({
        "Columna": ["Fecha", "Región", "Producto", "Ventas", "Clientes"],
        "Clasificación Atlas": [
            "TimeStamp, Operational_Metadata",
            "Geographic_Data, Location_Code",
            "Product_Entity",
            "Financial_Measure, Revenue_KPI",
            "Customer_Count, Indirect_PII"
        ],
        "Sensibilidad": ["Baja", "Media", "Baja", "Alta", "Media-Alta"],
        "Usuarios permitidos": [
            "Todos los usuarios con rol: data_analyst, data_engineer, viewer.",
            "regional_manager, data_analyst, acceso restringido por ubicación.",
            "Todos los usuarios autorizados.",
            "Solo finance_analyst, executive, compliance_officer.",
            "marketing_lead, privacy_officer, acceso anonimizado o agregado."
        ]
    })

    st.markdown("### Política de Gobierno de Datos")
    st.markdown("La siguiente tabla representa la política de acceso y criterios de clasificación de la sensibilidad de los datos tratados en este tablero. ")

    st.dataframe(df_gobierno_datos, hide_index = True)
    


    st.markdown("### Linaje de Datos")
    st.markdown("La siguiente tabla representa la política de acceso y criterios de clasificación de la sensibilidad de los datos tratados en este tablero. ")        
    
    st.dataframe(pd.DataFrame(
            {
               "Campo":["Fecha","Region","Producto","Ventas","Clientes"], 
               "Origen": ["'ventas_clientes.xlsx'"]*5,
               "Procesos":["'eda.ipynb' , 'seccion_practica.ipynb'"]*5,
               "Destino": ["Dashboard Ventas"]*5

            }
        ),
                    hide_index = True)


