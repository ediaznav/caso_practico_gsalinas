# Prueba Teórica – Gerente Especializado en Cloudera y Spark

Fecha de generación: 2025-05-07

## Cloudera Machine Learning y CDP

Las preguntas a detalle en esta sección y sus respuestas se encuentran en el documento "./docs/Prueba Teórica – Gerente Especializado en Cloudera y Spark.pdf"


## Sección Práctica – Ejercicios Técnicos

Pipeline de Procesamiento de Datos con PySpark (CML)Desarrolle un pipeline que procese el dataset ventas_clientes.csv con PySpark. 

ncluya:- Lectura del archivo CSV con esquema definido- Filtrado de datos para el año 2024- Agregación de ventas por Región y Producto- Identificación del producto con mayor venta en cada región- Exportación del resultado a CSV o visualizaciónAplique buenas prácticas: código modular, nombres descriptivos, comentarios y estilo PEP8.

La respuesta a este inciso se puede encontrar en: "./docs/seccion_practica.ipynb".


### Consultas SQL Avanzadas (Oracle/SQL)

Usando la tabla VENTAS (con las columnas del CSV), escriba:


- Consulta del total de ventas y clientes por región en 2024

**Total de ventas y clientes por región en 2024**

Query: 

``` SQL
-- Consulta para sumar clientes y ventas por región en el año 2024
SELECT
    region,
    SUM(clientes) AS clientes_total,
    SUM(ventas) AS ventas_total
FROM VENTAS
WHERE anio = 2023
GROUP BY region
```

- Consulta de los 3 productos con mayores ventas globales en 2024

**Top 3 productos con mayores ventas globales en 2024**

Query: 

``` SQL
-- Consulta  top 3 productos más vendidos en 2024
SELECT
    producto,
    SUM(ventas) AS ventas_total
FROM VENTAS
WHERE anio = 2023
GROUP BY producto
ORDER BY ventas_total DESC
LIMIT 3

```


- Consulta del porcentaje de crecimiento de ventas por producto entre 2023 y 2024


**Consulta del porcentaje de crecimiento de ventas por producto entre 2023 y 2024**

Query: 

``` SQL
-- Consulta del crecimiento porcentual de ventas por producto entre 2023 y 2024

-- CTE: Ventas por producto en 2023
WITH agg_ventas_2023 AS (
    SELECT
        producto,
        SUM(ventas) AS ventas_2023
    FROM VENTAS
    WHERE anio = 2023
    GROUP BY producto
),

-- CTE: Ventas por producto en 2024
agg_ventas_2024 AS (
    SELECT
        producto,
        SUM(ventas) AS ventas_2024
    FROM VENTAS
    WHERE anio = 2024
    GROUP BY producto
)

-- Join externo para comparar ventas año contra año
SELECT
    COALESCE(v24.producto, v23.producto) AS producto,
    COALESCE(ventas_2024, 0) AS ventas_2024,
    COALESCE(ventas_2023, 0) AS ventas_2023,
    -- Variación relativa de ventas: (2024 - 2023) / 2023
    CASE
        WHEN ventas_2023 > 0 THEN (ventas_2024 / ventas_2023) - 1
        WHEN ventas_2024 > 0 THEN 1.0
        ELSE 0.0
    END AS crecimiento_ventas
FROM agg_ventas_2023 v23
FULL OUTER JOIN agg_ventas_2024 v24
    ON v23.producto = v24.producto
```



Buenas Prácticas: Aplique principios de modularidad, legibilidad y documentación en su código. Se evaluará claridad, reutilización, y estilo.


# Caso Práctico Ejecutivo – Escenario de Negocio

La parte documental y el reporte ejecutivo correspondientes a esta parte se encuentran en el siguiente documento: "./docs/reporte_ejecutivo.pdf". 

El análisis y la puesta en producción de las visualizaciones se pueden encontrar en: [Dashboard](https://casopracticogsalinas-edgardaniel.streamlit.app/). 
