import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from io import StringIO, BytesIO
import numpy as np
from scipy.interpolate import lagrange, interp1d, CubicSpline, BarycentricInterpolator
from scipy.optimize import curve_fit

# Función para generar enlace de descarga de Excel
def generar_enlace_descarga_excel(df):
    towrite = BytesIO()
    df.to_excel(towrite, index=False, header=True)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="data_descargada.xlsx">📥 Descargar archivo Excel 📊</a>'
    return st.markdown(href, unsafe_allow_html=True)

# Función para generar enlace de descarga del gráfico en HTML
def generar_enlace_descarga_html(fig):
    towrite = StringIO()
    fig.write_html(towrite, include_plotlyjs="cdn")
    towrite = BytesIO(towrite.getvalue().encode())
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:text/html;charset=utf-8;base64,{b64}" download="grafico.html">📥 Descargar gráfico 📉</a>'
    return st.markdown(href, unsafe_allow_html=True)

# Función para calcular la interpolación de Newton
def interpolacion_newton(x, y, x_new):
    n = len(x)
    coeficientes = np.zeros((n, n))
    coeficientes[:, 0] = y

    # Calcular coeficientes de diferencias divididas
    for j in range(1, n):
        for i in range(n - j):
            coeficientes[i, j] = (coeficientes[i + 1, j - 1] - coeficientes[i, j - 1]) / (x[i + j] - x[i])
    
    # Evaluar el polinomio de Newton en los nuevos puntos
    def newton_polynomial(x_val):
        result = coeficientes[0, 0]
        term = 1.0
        for i in range(1, n):
            term *= (x_val - x[i - 1])
            result += coeficientes[0, i] * term
        return result
    
    return np.array([newton_polynomial(xi) for xi in x_new])

# Configuración de la página
st.set_page_config(page_title='Graficador Excel con Interpolación', layout='wide', initial_sidebar_state='expanded')

# Barra lateral
st.sidebar.title("Acerca de 📘")
st.sidebar.info("Sube un archivo Excel, visualiza sus datos y realiza múltiples interpolaciones. 🚀")

# Contenido principal
st.title('Graficador Excel con Interpolación 📈')
st.subheader('¡Sube tu archivo Excel y que comience la magia! ✨')

uploaded_file = st.file_uploader('Elige un archivo XLSX 📁', type='xlsx')
if uploaded_file:
    st.markdown('---')
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    st.dataframe(df.style.highlight_max(axis=0))

    st.markdown('### Análisis de datos e Interpolación 🔍')

    # Listar todas las columnas del DataFrame
    all_columns = df.columns.tolist()

    # Selección de columna para la interpolación
    columna_interpolacion = st.selectbox('Selecciona una columna para la interpolación 📊:', all_columns)
    
    # Elegir la columna para el eje X
    columna_x = st.selectbox('Selecciona la columna para el eje X (ej. FECHA):', all_columns)

    # Seleccionar métodos de interpolación
    metodos_seleccionados = st.multiselect(
        'Selecciona los métodos de interpolación a utilizar:',
        ['Mínimos Cuadrados Polinomico', 'Levenberg-Marquardt', 'Spline', 'Interpolación Lineal', 'Lagrange', 'Newton', 'Barycentrica'],
        default=['Lagrange']
    )

    try:
        # Agrupar los datos y calcular la media para los valores duplicados
        grouped_df = df.groupby(columna_x).agg({columna_interpolacion: 'mean'}).reset_index()

        # Reconstruir x_values y y_values
        x_values = pd.to_datetime(grouped_df[columna_x], errors='coerce')
        y_values = pd.to_numeric(grouped_df[columna_interpolacion], errors='coerce')

        # Eliminar NaN después de la agrupación
        mask = x_values.notna() & y_values.notna()
        x_values = x_values[mask]
        y_values = y_values[mask]

        # Manejar ceros en los datos
        y_values = y_values.replace(0, 0.1)

        # Verifica si hay suficientes puntos
        if len(x_values) > 1:
            # Convertir x_values a timestamps (números flotantes)
            x_values_numeric = x_values.astype(np.int64) // 10**9  # Convertir a segundos
            
            # Escalar los valores de x restando el valor mínimo
            x_min = x_values_numeric.min()
            x_values_scaled = x_values_numeric - x_min
            x_new = np.linspace(x_values_scaled.min(), x_values_scaled.max(), 500)
            x_new_dates = pd.to_datetime((x_new + x_min) * 10**9)  # Convertir de segundos a datetime

            # Crear el gráfico
            fig = px.scatter(x=x_values, y=y_values, labels={'x': columna_x, 'y': columna_interpolacion},
                             title=f'Interpolación de {columna_interpolacion} vs {columna_x}')

            # Mínimos Cuadrados Polinomicos
            if 'Mínimos Cuadrados Polinomico' in metodos_seleccionados:
                grado_polinomio = st.slider('Selecciona el grado del polinomio para Mínimos Cuadrados:', min_value=1, max_value=10, value=3)
                coeficientes = np.polyfit(x_values_scaled, y_values, grado_polinomio)
                poly = np.poly1d(coeficientes)
                y_new = poly(x_new)
                fig.add_scatter(x=x_new_dates, y=y_new, mode='lines', name='Mínimos Cuadrados Polinomico', line=dict(color='yellow'))

            # Levenberg-Marquardt
            if 'Levenberg-Marquardt' in metodos_seleccionados:
                def model(x, a, b, c):
                    return a * x**b + c  # Modelo polinómico
                try:
                    popt, _ = curve_fit(model, x_values_scaled, y_values, p0=(1, 1, 1))  # Ajuste usando Levenberg-Marquardt
                    y_lm = model(x_new, *popt)
                    fig.add_scatter(x=x_new_dates, y=y_lm, mode='lines', name='Levenberg-Marquardt', line=dict(color='orange'))
                except Exception as e:
                    st.error(f"Error en el ajuste de Levenberg-Marquardt: {e}")

            # Spline
            if 'Spline' in metodos_seleccionados:
                spline = CubicSpline(x_values_scaled, y_values)
                y_spline = spline(x_new)
                fig.add_scatter(x=x_new_dates, y=y_spline, mode='lines', name='Spline', line=dict(color='blue'))

            # Interpolación Lineal
            if 'Interpolación Lineal' in metodos_seleccionados:
                linear_interp = interp1d(x_values_scaled, y_values, kind='linear')
                y_linear = linear_interp(x_new)
                fig.add_scatter(x=x_new_dates, y=y_linear, mode='lines', name='Interpolación Lineal', line=dict(color='green'))

            # Lagrange
            if 'Lagrange' in metodos_seleccionados:
                try:
                    poly_lagrange = lagrange(x_values_scaled, y_values)
                    y_lagrange = poly_lagrange(x_new)
                    fig.add_scatter(x=x_new_dates, y=y_lagrange, mode='lines', name='Lagrange', line=dict(color='red'))
                except Exception as e:
                    st.error(f"Error al calcular Lagrange: {e}")

            # Interpolación de Newton
            if 'Newton' in metodos_seleccionados:
                y_newton = interpolacion_newton(x_values_scaled, y_values, x_new)
                fig.add_scatter(x=x_new_dates, y=y_newton, mode='lines', name='Interpolación de Newton', line=dict(color='purple'))

            # Interpolación Barycentrica (con escalado y visualización parcial)
            if 'Barycentrica' in metodos_seleccionados:
                try:
                    # Asegúrate de que x_values_scaled esté ordenado
                    sorted_indices = np.argsort(x_values_scaled)
                    x_sorted = x_values_scaled[sorted_indices]
                    y_sorted = y_values[sorted_indices]

                    P = BarycentricInterpolator(x_sorted, y_sorted)
                    y_barycentric = P(x_new)

                    # Mostrar una sección más pequeña de los datos
                    fig.add_scatter(x=x_new_dates[:100], y=y_barycentric[:100], mode='lines', name='Barycentrica (pequeña sección)', line=dict(color='cyan'))

                    # Mostrar el gráfico completo
                    fig.add_scatter(x=x_new_dates, y=y_barycentric, mode='lines', name='Barycentrica (completa)', line=dict(color='cyan', dash='dash'))

                except Exception as e:
                    st.error(f"Error al calcular la Interpolación Barycentrica: {e}")

            # Mostrar el gráfico
            st.plotly_chart(fig)

            # Sección de descargas
            st.markdown('### Descargas 📥')
            generar_enlace_descarga_excel(df)
            generar_enlace_descarga_html(fig)

        else:
            st.error("No hay suficientes puntos para realizar la interpolación.")
    except Exception as e:
        st.error(f"Error en el procesamiento del archivo: {e}")
else:
    st.warning('Por favor, sube un archivo XLSX para continuar.')
