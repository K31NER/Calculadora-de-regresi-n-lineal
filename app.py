import streamlit as st
import pandas as pd
import numpy as np
import io

# --- Importar funciones desde regresion.py ---
try:
    from regresion import (
        validar_datos,
        regresion_lineal,
        correlacion_pearson,
        graficar_regresion,
        predecir_valor,
        COL_X, COL_Y, COL_PRED
    )
    REGRESION_PY_IMPORTED = True
except ImportError as e:
    st.error(f"**Error Crítico:** No se pudo importar `regresion.py`. Detalles: {e}")
    REGRESION_PY_IMPORTED = False
    st.stop()

# --- Configuración de la Página e Inicialización del Estado ---
st.set_page_config(page_title="Calculadora Regresión Lineal", page_icon="https://cdn-icons-png.flaticon.com/128/1998/1998646.png", layout="wide")

default_session_state = {
    'calculado': False, 'df': None, 'modelo': None, 'pendiente': None,
    'intercepcion': None, 'r2': None, 'correlacion': None, 'fig': None,
    'predicciones_modelo': None, 'error_calculo': None,
    'csv_x_col': None, 'csv_y_col': None
}
for key, default_value in default_session_state.items():
    if key not in st.session_state: st.session_state[key] = default_value

# --- Estilos CSS ---
st.markdown("""
<style>
    /* ... (otros estilos) ... */
    .main-header { font-size: 2.2rem; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.4rem; font-weight: bold; color: #26A69A; margin-top: 1.5rem; margin-bottom: 0.8rem; border-bottom: 2px solid #eee; padding-bottom: 5px; }

    /* --- AJUSTES PARA MÉTRICAS --- */
    .metric-block { margin-bottom: 12px; } /* Espacio entre métricas */
    .metric-label-small { /* Etiqueta más pequeña */
        font-weight: bold;
        color: #666; /* Un gris más suave */
        font-size: 0.85em; /* Tamaño más pequeño */
        display: block;
        margin-bottom: 1px;
    }
    .metric-value-small { /* Valor numérico más pequeño */
        font-size: 1.2em; /* Más pequeño que st.metric, pero legible. Ajusta si es necesario. */
        font-family: monospace;
        color: #eee; /* Blanco/Casi blanco para tema oscuro */
        font-weight: bold; /* Mantenerlo en negrita */
        display: block;
    }
    /* Fin de ajustes para métricas */

    .stButton>button { width: 100%; background-color: #1E88E5; color: white; border-radius: 5px; border: none; padding: 10px;}
    .stButton>button:hover { background-color: #1565C0; color: white; }
    .stButton[key="calculate_button"]>button { background-color: #28a745; }
    .stButton[key="calculate_button"]>button:hover { background-color: #218838; }
    div[data-testid="stVerticalBlock"]:has(>div>span.metric-label:contains("Predicción")) { background-color: #e3f2fd; border-left: 5px solid #1e88e5; padding: 10px; border-radius: 5px; }

    /* --- ESTILO PARA LA ECUACIÓN (MÁS GRANDE) --- */
    .latex-equation-container .stLatex {
        font-size: 2.0em !important; /* Aumentado a 2.0em. Ajusta al gusto. */
        text-align: center;
        display: block;
        margin-top: 10px; /* Más espacio arriba */
        margin-bottom: 15px; /* Más espacio abajo */
        padding: 10px; /* Añadir algo de padding */
        /* background-color: rgba(255, 255, 255, 0.05); /* Fondo sutil opcional */
        /* border-radius: 5px; */
    }

</style>
""", unsafe_allow_html=True)

# --- Título Principal ---
st.markdown("<div class='main-header'>Calculadora de Regresión Lineal Simple 📈</div>", unsafe_allow_html=True)
st.caption("Ingresa datos, calcula la regresión, visualiza y predice.")
st.divider()

# --- Contenido en Pestañas (2 Pestañas) ---
tab1, tab2 = st.tabs(["📊 Ingresar Datos y Resultados", "📋 Guía Rápida"])

# =====================================
# PESTAÑA 1: INGRESAR DATOS Y RESULTADOS
# =====================================
with tab1:
    st.markdown("<div class='sub-header'>1. Ingreso de Datos</div>", unsafe_allow_html=True)
    input_option = st.radio("Método:", ["Manual (comas)", "Subir CSV"], key="input_method", horizontal=True, label_visibility="collapsed")

    datos_x_list = []
    datos_y_list = []

    # --- Entrada Manual (comas) ---
    if input_option == "Manual (comas)":
        col1, col2 = st.columns(2)
        with col1:
            x_input_text = st.text_area("Valores de X (separados por coma):", "15,14,17,16,15,16,15,13,17,16,16", height=150, key="x_manual", help="Ej: 1, 2.5, 3, ...")
        with col2:
            y_input_text = st.text_area("Valores de Y (separados por coma):", "2,0,3,4,3,4,3,1,4,3,5", height=150, key="y_manual", help="Misma cantidad que X.")
        if x_input_text: datos_x_list = [val.strip() for val in x_input_text.strip().split(',') if val.strip()]
        if y_input_text: datos_y_list = [val.strip() for val in y_input_text.strip().split(',') if val.strip()]

    # --- Subir Archivo CSV ---
    else: # Subir CSV
        uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"], key="csv_uploader")
        if uploaded_file is not None:
            # ... (código para cargar CSV sin cambios) ...
            try:
                @st.cache_data
                def load_csv(file):
                    try: return pd.read_csv(file)
                    except Exception: file.seek(0); return pd.read_csv(file, sep=';')
                    except Exception: file.seek(0); return pd.read_csv(file, sep='\t')

                df_input_preview = load_csv(uploaded_file)
                st.success("CSV cargado.")
                st.dataframe(df_input_preview.head(), use_container_width=True)
                available_columns = df_input_preview.columns.tolist()
                if len(available_columns) < 2: st.error("CSV debe tener al menos 2 columnas.")
                else:
                    col1_csv, col2_csv = st.columns(2)
                    with col1_csv:
                        idx_x = available_columns.index(st.session_state.csv_x_col) if st.session_state.csv_x_col in available_columns else 0
                        st.session_state.csv_x_col = st.selectbox("Columna X:", available_columns, index=idx_x, key="x_col_select")
                    with col2_csv:
                        valid_y_options = [c for c in available_columns if c != st.session_state.csv_x_col]
                        default_y_col = st.session_state.csv_y_col if st.session_state.csv_y_col in valid_y_options else (valid_y_options[0] if valid_y_options else available_columns[0])
                        default_y_index = available_columns.index(default_y_col)
                        st.session_state.csv_y_col = st.selectbox("Columna Y:", available_columns, index=default_y_index, key="y_col_select")

                    if st.session_state.csv_x_col and st.session_state.csv_y_col:
                        if st.session_state.csv_x_col == st.session_state.csv_y_col: st.warning("⚠️ X e Y son la misma columna.")
                        try:
                            datos_x_list = df_input_preview[st.session_state.csv_x_col].astype(str).tolist()
                            datos_y_list = df_input_preview[st.session_state.csv_y_col].astype(str).tolist()
                        except Exception as e_csv:
                             st.error(f"❌ Error al extraer datos del CSV: {e_csv}")
                             datos_x_list, datos_y_list = [], []
            except Exception as e:
                st.error(f"❌ Error al procesar CSV: {e}")


    st.divider()

    # --- Botón para Calcular ---
    if st.button("🚀 Calcular Regresión Lineal", key="calculate_button", use_container_width=True):
        # ... (código de resetear estado y validación/cálculo sin cambios) ...
        for key in default_session_state:
            if key not in ['csv_x_col', 'csv_y_col']: st.session_state[key] = default_session_state[key]
        st.session_state.error_calculo = None

        if not datos_x_list or not datos_y_list:
            st.warning("⚠️ Ingresa o carga datos válidos para X e Y.")
            st.session_state.error_calculo = "Datos insuficientes."
        else:
            df_validado, error_val = validar_datos(datos_x_list, datos_y_list)
            if error_val:
                st.error(f"❌ Validación Fallida: {error_val}")
                st.session_state.error_calculo = error_val
            else:
                st.session_state.df = df_validado
                corr, error_corr = correlacion_pearson(st.session_state.df[COL_X].tolist(), st.session_state.df[COL_Y].tolist())
                if error_corr: st.warning(f"⚠️ Correlación: {error_corr}")
                st.session_state.correlacion = corr

                preds, pend, r2_val, intercep, mod, error_reg = regresion_lineal(st.session_state.df)
                if error_reg:
                    st.error(f"❌ Regresión Fallida: {error_reg}")
                    st.session_state.error_calculo = error_reg
                else:
                    st.session_state.predicciones_modelo = preds
                    st.session_state.pendiente = pend
                    st.session_state.r2 = r2_val
                    st.session_state.intercepcion = intercep
                    st.session_state.modelo = mod
                    figura = graficar_regresion(st.session_state.df, preds, r2_val)
                    st.session_state.fig = figura
                    st.session_state.calculado = True


    st.divider()

    # --- SECCIÓN DE RESULTADOS ---
    st.markdown("<div class='sub-header'>Resultados del Análisis</div>", unsafe_allow_html=True)

    if st.session_state.get('calculado', False):
        col_res1, col_res2 = st.columns([1, 1.5]) # Ajustar ratio si es necesario

        with col_res1: # Métricas y Predicción
            st.markdown("##### 📊 Métricas")
            with st.container(border=True):

                # --- CAMBIO: Mostrar métricas con Markdown y CSS ---
                if st.session_state.pendiente is not None:
                    st.markdown(f"""<div class='metric-block'>
                                <span class='metric-label-small'>Pendiente (m):</span>
                                <span class='metric-value-small'>{st.session_state.pendiente:.4f}</span>
                                </div>""", unsafe_allow_html=True)
                if st.session_state.intercepcion is not None:
                     st.markdown(f"""<div class='metric-block'>
                                 <span class='metric-label-small'>Intercepto (b):</span>
                                 <span class='metric-value-small'>{st.session_state.intercepcion:.4f}</span>
                                 </div>""", unsafe_allow_html=True)
                if st.session_state.r2 is not None:
                     st.markdown(f"""<div class='metric-block'>
                                 <span class='metric-label-small'>R²:</span>
                                 <span class='metric-value-small'>{st.session_state.r2:.4f}</span>
                                 </div>""", unsafe_allow_html=True)
                if st.session_state.correlacion is not None:
                     st.markdown(f"""<div class='metric-block'>
                                 <span class='metric-label-small'>Correlación (r):</span>
                                 <span class='metric-value-small'>{st.session_state.correlacion:.4f}</span>
                                 </div>""", unsafe_allow_html=True)
                # --- FIN DEL CAMBIO ---

                # Ecuación en LaTeX (más grande)
                if st.session_state.pendiente is not None and st.session_state.intercepcion is not None:
                    m = st.session_state.pendiente
                    b = st.session_state.intercepcion
                    signo_b = "+" if b >= 0 else "-"
                    abs_b = abs(b)
                    latex_eq = rf"Y \approx {m:.4f}X {signo_b} {abs_b:.4f}"
                    # st.markdown("**Ecuación de Regresión:**") # Título opcional antes de la ecuación
                    st.markdown(f"<div class='latex-equation-container'>", unsafe_allow_html=True)
                    st.latex(latex_eq)
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("##### 🔮 Predicción")
            # ... (código de predicción sin cambios) ...
            with st.container(border=True):
                nuevo_valor_x = st.number_input("Valor de X para predecir:", value=None, step=1.0, format="%.4f", placeholder="Escribe un número...", label_visibility="collapsed")
                if nuevo_valor_x is not None:
                    prediccion, error_pred = predecir_valor(st.session_state.modelo, nuevo_valor_x)
                    if error_pred: st.error(f"❌ {error_pred}")
                    elif prediccion is not None:
                         st.success(f"Para X = {nuevo_valor_x:.4f}, Y predicho ≈ **{prediccion:.4f}**")
                         st.session_state.last_prediction = {'x': nuevo_valor_x, 'y': prediccion}
                else:
                    st.caption("Ingresa un valor de X para predecir.")


        with col_res2: # Gráfico y Datos
            # ... (código del gráfico y tabla sin cambios) ...
            st.markdown("##### 📈 Gráfico")
            if st.session_state.fig:
                fig_display = st.session_state.fig
                if 'last_prediction' in st.session_state and st.session_state.last_prediction:
                     try:
                         pred_info = st.session_state.last_prediction
                         st.session_state.fig.add_scatter(
                             x=[pred_info['x']], y=[pred_info['y']], mode='markers',
                             marker=dict(color='purple', size=12, symbol='star'), name=f'Predicción ({pred_info["x"]:.2f})' )
                         fig_display = st.session_state.fig
                     except Exception as e: print(f"Warn: No se pudo añadir punto pred: {e}")
                st.plotly_chart(fig_display, use_container_width=True)
                if 'last_prediction' in st.session_state: del st.session_state.last_prediction
            else: st.warning("⚠️ Gráfico no disponible.")

            st.markdown("##### 💾 Datos")
            if st.session_state.df is not None:
                df_display = st.session_state.df.copy()
                if st.session_state.predicciones_modelo is not None:
                     df_display[COL_PRED] = st.session_state.predicciones_modelo
                try:
                    for col in df_display.select_dtypes(include=np.number).columns: df_display[col] = df_display[col].map('{:.4f}'.format)
                except Exception: pass
                st.dataframe(df_display, use_container_width=True, height=200)
            else: st.caption("No hay datos procesados.")


    elif st.session_state.get('error_calculo'):
        st.error(f"❌ El cálculo falló: {st.session_state.error_calculo}")
        st.info("ℹ️ Corrige los datos de entrada o el error indicado e intenta de nuevo.")
    else:
         st.info("ℹ️ Ingresa datos y haz clic en 'Calcular Regresión Lineal' para ver los resultados.")

# ==========================
# PESTAÑA 2: GUÍA RÁPIDA
# ==========================
with tab2:
    # ... (código de la guía sin cambios) ...
    st.markdown("<div class='sub-header'>Guía de Uso y Conceptos Clave</div>", unsafe_allow_html=True)
    st.markdown(f"""
    ### 📚 Regresión Lineal Simple
    Modela relación entre 2 variables numéricas: **X** (`{COL_X}`) y **Y** (`{COL_Y}`). Busca la mejor **línea recta**.
    ### 🧮 Ecuación: `Y = mX + b`
    *   **`m` (Pendiente):** Cambio en Y por unidad de X.
    *   **`b` (Intercepto):** Valor de Y cuando X=0.
    ### 📊 Interpretación
    *   **R²:** (0 a 1) % de varianza de Y explicada por X. > Cercano a 1 = mejor ajuste.
    *   **r (Correlación):** (-1 a +1) Fuerza y dirección lineal. > Cercano a +/-1 = fuerte.
    ### ⚙️ Pasos
    1.  Ve a "📊 Ingresar Datos y Resultados".
    2.  Elige método: **Manual (comas)** o **Subir CSV**.
    3.  Proporciona los datos.
    4.  Clic en **"🚀 Calcular Regresión Lineal"**.
    5.  Resultados aparecen **debajo del botón**.
    ### ⚠️ Importante
    *   Asume relación **lineal**.
    *   **Correlación ≠ Causalidad.**
    *   **Outliers** pueden afectar.
    """)
