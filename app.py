import streamlit as st
import pandas as pd
from regresion import validar_datos, regresion_lineal, predecir_valor, graficar_regresion, correlacion_pearson

# Configuración de la página
st.set_page_config(
    page_title="Calculadora de Regresión Lineal",
    page_icon="https://cdn-icons-png.flaticon.com/128/891/891175.png",
    layout="wide",
)

st.title("🔢 Análisis de Regresión Lineal")
st.markdown("Introduce tus datos para visualizar la correlación, calcular la regresión lineal y hacer predicciones.")

# ================== Entradas ==================
st.subheader("📥 Ingreso de Datos")

col1, col2 = st.columns(2)
with col1:
    x_values = st.text_area("Valores de X (separados por coma)", "15,14,17,16,15,16,15,13,17,16,16")
with col2:
    y_values = st.text_area("Valores de Y (separados por coma)", "2,0,3,4,3,4,3,1,4,3,5")

# ================== Botón de cálculo ==================
if st.button("⚙️ Calcular Regresión"):
    try:
        X = list(map(float, x_values.split(',')))
        Y = list(map(float, y_values.split(',')))
        df = validar_datos(X, Y)

        if df is None:
            st.error("❌ Error en los datos. Asegúrate de que las listas no estén vacías y tengan igual longitud.")
        else:
            correlacion = correlacion_pearson(X, Y)
            y_pred, pendiente, r2, intercepcion, modelo = regresion_lineal(df)

            if modelo:
                # Almacenamos el modelo y resultados en la sesión para persistirlos
                st.session_state.modelo = modelo
                st.session_state.df = df
                st.session_state.y_pred = y_pred
                st.session_state.pendiente = pendiente
                st.session_state.intercepcion = intercepcion
                st.session_state.r2 = r2
                st.session_state.corr = correlacion

                st.success(f"📈 Correlación de Pearson: **{correlacion:.2f}**")
            else:
                st.error("⚠️ No se pudo entrenar el modelo. Verifica tus datos.")
    except Exception as e:
        st.error(f"💥 Error al procesar los datos: {e}")

# ================== Mostrar resultados persistentes ==================
if 'modelo' in st.session_state:
    st.markdown("### 🧮 Ecuación de la recta")
    # La ecuación se muestra en formato LaTeX
    st.latex(f"y = {st.session_state.pendiente:.2f}x + {st.session_state.intercepcion:.2f}")
    
    st.subheader("📊 Estadísticas del Modelo")
    met1, met2, met3, met4 = st.columns(4)
    met1.metric("📐 Pendiente (m)", f"{st.session_state.pendiente:.2f}")
    met2.metric("🎯 Intercepto (b)", f"{st.session_state.intercepcion:.2f}")
    met3.metric("📈 R²", f"{st.session_state.r2:.2f}")
    met4.metric("🔗 Correlación (r)", f"{st.session_state.corr:.2f}")
    
    st.subheader("📈 Gráfica de Regresión")
    fig = graficar_regresion(st.session_state.df, st.session_state.y_pred, st.session_state.r2)
    st.plotly_chart(fig, use_container_width=True)

    # ================== Predicción personalizada (sin borrar contenido anterior) ==================
    st.subheader("🔮 Hacer una predicción")
    nuevo_valor = st.number_input("Introduce un nuevo valor de X:", value=20.0, step=1.0)
    # Usamos un botón separado para predecir
    if st.button("📌 Predecir Nuevo Valor"):
        prediccion = predecir_valor(st.session_state.modelo, [nuevo_valor])[0]
        st.success(f"➡️ Para X = `{nuevo_valor}`, la predicción de Y es: **{prediccion:.2f}**")
