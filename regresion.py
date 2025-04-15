# regresion.py (Simplificado)
from sklearn.linear_model import LinearRegression
import plotly.express as px
import pandas as pd
import numpy as np

# Nombres de columna
COL_X = "Variables_independiente"
COL_Y = "Variables_dependientes"
COL_PRED = "Predicciones"

def correlacion_pearson(X: list, Y: list) -> tuple[float | None, str | None]:
    """Calcula la correlacion de Pearson."""
    try:
        X_data = np.array(X, dtype=float)
        Y_data = np.array(Y, dtype=float)

        if len(X_data) < 2:
            return None, "Se necesitan al menos 2 puntos."
        # np.corrcoef maneja std=0 devolviendo NaN o con warnings, que podemos chequear después
        correlacion = np.corrcoef(X_data, Y_data)[0, 1]

        if np.isnan(correlacion):
             return None, "Correlación no definida (posiblemente datos constantes)."

        return correlacion, None
    except ValueError:
        return None, "Datos inválidos para correlación (no numéricos)."
    except Exception as e:
        return None, f"Error en correlación: {e}"

def validar_datos(X: list, Y: list) -> tuple[pd.DataFrame | None, str | None]:
    """Valida y convierte datos a DataFrame."""
    if not X or not Y:
        return None, "Las listas X e Y no pueden estar vacías."
    if len(X) != len(Y):
        return None, f"X e Y deben tener la misma longitud ({len(X)} vs {len(Y)})."

    try:
        X_data = np.array(X, dtype=float)
        Y_data = np.array(Y, dtype=float)

        # Check for NaNs introduced by conversion
        if np.isnan(X_data).any() or np.isnan(Y_data).any():
            return None, "Se encontraron valores no numéricos en los datos."

        if len(X_data) < 2:
             return None, "Se necesitan al menos 2 puntos de datos válidos."

        datos = {COL_X: X_data, COL_Y: Y_data}
        df = pd.DataFrame(datos)
        return df, None

    except ValueError:
         return None, "Error al convertir datos a números. Verifica que todos sean válidos."
    except Exception as e:
        return None, f"Error inesperado al validar datos: {e}"

def regresion_lineal(df: pd.DataFrame) -> tuple:
    """Calcula la regresion lineal."""
    if df is None or df.empty:
         return None, None, None, None, None, "DataFrame inválido."
    if len(df) < 2:
        return None, None, None, None, None, "Se necesitan al menos 2 puntos."

    try:
        x_vals = df[[COL_X]].values
        y_vals = df[COL_Y].values

        modelo = LinearRegression()
        modelo.fit(x_vals, y_vals)

        y_pred = modelo.predict(x_vals)
        r2 = modelo.score(x_vals, y_vals)
        intercepcion = modelo.intercept_
        pendiente = modelo.coef_[0]

        # Chequeo básico de NaN post-cálculo (sklearn es robusto, pero por si acaso)
        if not all(np.isfinite([pendiente, intercepcion, r2])):
             return None, None, None, None, None, "Cálculo de regresión resultó en valores no finitos."

        return y_pred, pendiente, r2, intercepcion, modelo, None
    except Exception as e:
        return None, None, None, None, None, f"Error en cálculo de regresión: {e}"

def predecir_valor(modelo, nuevo_x) -> tuple[float | None, str | None]:
    """Realiza una predicción."""
    if modelo is None:
        return None, "Modelo no disponible."
    try:
        nuevo_x_float = float(nuevo_x) # Intenta convertir a float
        nuevo_x_array = np.array([[nuevo_x_float]])
        prediccion = modelo.predict(nuevo_x_array)

        if not np.isfinite(prediccion[0]):
             return None, "Predicción resultó en valor no finito."

        return prediccion[0], None
    except (ValueError, TypeError):
        return None, f"Valor de entrada '{nuevo_x}' no es un número válido para predecir."
    except Exception as e:
        return None, f"Error al predecir: {e}"

def graficar_regresion(df: pd.DataFrame, predicciones, r2: float):
    """Genera la figura de Plotly."""
    if df is None or predicciones is None or r2 is None or not np.isfinite(r2):
        return None # No graficar si faltan datos o r2 es inválido

    df_copy = df.copy()
    df_copy[COL_PRED] = predicciones
    try:
        fig = px.scatter(df_copy, x=COL_X, y=COL_Y,
                         title=f"Regresión Lineal (R² = {r2:.4f})",
                         labels={COL_X: 'Variable Independiente (X)', COL_Y: 'Variable Dependiente (Y)'})
        fig.add_scatter(x=df_copy[COL_X], y=df_copy[COL_PRED],
                        mode="lines", name="Línea de regresión", line=dict(color='red'))
        fig.update_layout(showlegend=True)
        return fig
    except Exception as e:
        print(f"Error al graficar: {e}")
        return None