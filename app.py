import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from io import StringIO, BytesIO
import numpy as np
from scipy.interpolate import lagrange, interp1d, CubicSpline, BarycentricInterpolator, splprep, splev
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error,mean_absolute_error,mean_squared_error

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

# Método de Máxima Verosimilitud
def metodo_maxima_verosimilitud(x, y):
    def modelo(x, mu, sigma, amplitude):
        return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    p0 = [np.mean(x), np.std(x), np.max(y)]
    try:
        popt, _ = curve_fit(modelo, x, y, p0=p0)
        return modelo(x, *popt), popt
    except Exception as e:
        st.error(f"Error en el ajuste de Máxima Verosimilitud: {e}")
        return np.zeros_like(x), None

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

    # Selección de columnas para la interpolación
    columnas_interpolacion = st.multiselect('Selecciona las columnas para la interpolación 📊:', all_columns)
    
    # Elegir la columna para el eje X
    columna_x = st.selectbox('Selecciona la columna para el eje X (ej. FECHA):', all_columns)

    # Seleccionar métodos de interpolación
    metodos_interpolacion = st.multiselect(
        'Selecciona los métodos de interpolación:',
        ['Interpolación Lineal', 'Lagrange', 'Newton', 'Barycentrica', 'Spline Cubico', 'B-Spline'],
        default=['Lagrange']
    )

    # Seleccionar métodos de ajuste de curvas
    metodos_ajuste = st.multiselect(
        'Selecciona los métodos de ajuste de curvas:',
        ['Mínimos Cuadrados Polinomico', 'Máxima Verosimilitud', 'Ridge', 'Lasso','Levenberg-Marquardt']
    )
    # Almacena los errores en un DataFrame
    errores_df = pd.DataFrame(columns=['Método', 'Error'])
    try:
        # Agrupar los datos y calcular la media para los valores duplicados
        grouped_df = df.groupby(columna_x).agg({columna: 'mean' for columna in columnas_interpolacion}).reset_index()

        # Reconstruir x_values y y_values
        x_values = pd.to_datetime(grouped_df[columna_x], errors='coerce')
        y_values = pd.concat([pd.to_numeric(grouped_df[col], errors='coerce') for col in columnas_interpolacion], axis=1).mean(axis=1)

        # Eliminar NaN después de la agrupación
        mask = x_values.notna() & y_values.notna()
        x_values = x_values[mask]
        y_values = y_values[mask]
        # Manejar ceros en los datos
        y_values = y_values.replace(0, 0.1)

        if len(x_values) > 1:
            #Conversión de fechas a segundos
            x_values_numeric = x_values.astype(np.int64) // 10**9  # Convertir a segundos
            #Cálculo del mínimo
            x_min = x_values_numeric.min()
            #Escalamiento de los valores
            x_values_scaled = x_values_numeric - x_min
            #Generación de nuevos valores
            x_new = np.linspace(x_values_scaled.min(), x_values_scaled.max(), 1095)
            #Conversión de nuevo a fechas
            x_new_dates = pd.to_datetime((x_new + x_min) * 10**9)

            # Crear el gráfico
            fig = px.scatter(x=x_values, y=y_values, labels={'x': columna_x, 'y': 'Media de ' + ', '.join(columnas_interpolacion)},
                             title=f'Interpolación de Media de {", ".join(columnas_interpolacion)} vs {columna_x}')

            # Métodos de interpolación
            errores = {}
            resultados_error = []
            # Diccionario para almacenar métricas
            metricas_resultados = []
            # Inicializar un diccionario para almacenar las métricas
            metrics = {}    

            if 'Interpolación Lineal' in metodos_interpolacion:
                try:
                    linear_interp = interp1d(x_values_scaled, y_values, kind='linear')
                    y_linear = linear_interp(x_new)
                    fig.add_scatter(x=x_new_dates, y=y_linear, mode='lines', name='Interpolación Lineal')
                    # Asegúrate de que ambos arrays tengan la misma longitud
                    

                    mse = np.mean((y_values[:len(x_new)] - y_linear) ** 2)
                    error_relativo = np.mean(
                    np.abs((y_values[:len(x_new)] - y_linear) / np.where(y_values[:len(x_new)] != 0, y_values[:len(x_new)], np.nan))) * 100
                    resultados_error.append({'Método': 'Interpolación Lineal', 'MSE': mse, 'Error Relativo (%)': error_relativo})
            

                except Exception as e:
                     st.error(f"Error en Interpolación Lineal: {e}")
            # Mostrar tabla de errores
            if resultados_error:
                st.subheader("Errores de Interpolación")
                st.table(resultados_error)

            if 'Lagrange' in metodos_interpolacion:
                try:
                    poly_lagrange = lagrange(x_values_scaled, y_values)
                    y_lagrange = poly_lagrange(x_new)
                    fig.add_scatter(x=x_new_dates, y=y_lagrange, mode='lines', name='Lagrange', line=dict(color='red'))
                except Exception as e:
                    errores['Lagrange'] = str(e)

            if 'Newton' in metodos_interpolacion:
                # Implementación de la interpolación de Newton
                def interpolacion_newton(x, y, x_new):
                    n = len(x)
                    coeficientes = np.zeros((n, n))
                    coeficientes[:, 0] = y

                    for j in range(1, n):
                        for i in range(n - j):
                            coeficientes[i, j] = (coeficientes[i + 1, j - 1] - coeficientes[i, j - 1]) / (x[i + j] - x[i])
                    
                    def newton_polynomial(x_val):
                        result = coeficientes[0, 0]
                        term = 1.0
                        for i in range(1, n):
                            term *= (x_val - x[i - 1])
                            result += coeficientes[0, i] * term
                        return result
                    
                    return np.array([newton_polynomial(xi) for xi in x_new])

                try:
                    y_newton = interpolacion_newton(x_values_scaled, y_values, x_new)
                    fig.add_scatter(x=x_new_dates, y=y_newton, mode='lines', name='Interpolación de Newton', line=dict(color='purple'))
                except Exception as e:
                    errores['Interpolación de Newton'] = str(e)

            if 'Barycentrica' in metodos_interpolacion:
                try:
                    sorted_indices = np.argsort(x_values_scaled)
                    x_sorted = x_values_scaled[sorted_indices]
                    y_sorted = y_values[sorted_indices]

                    P = BarycentricInterpolator(x_sorted, y_sorted)
                    y_barycentric = P(x_new)
                    fig.add_scatter(x=x_new_dates, y=y_barycentric, mode='lines', name='Barycentrica', line=dict(color='cyan'))
                except Exception as e:
                    errores['Barycentrica'] = str(e)

            if 'Spline Cubico' in metodos_interpolacion:
                try:
                    
                    # Utilizar todos los datos como están
                    spline = CubicSpline(x_values_scaled, y_values, extrapolate=True)
                    y_spline = spline(x_new)
                    fig.add_scatter(x=x_new_dates, y=y_spline, mode='lines', name='Spline Cubico', line=dict(color='blue'))
                    # Asegurarte de que las longitudes coincidan
                    y_actual = y_values.values  # Los valores originales
                    min_length = min(len(y_actual), len(y_spline))
                    y_actual = y_actual[:min_length]
                    y_spline = y_spline[:min_length]
                    # Calcular errores para Spline Cúbico
                    mse_spline = np.mean((y_spline - y_actual[:len(y_spline)]) ** 2)
                    error_relativo_spline = np.mean(np.abs((y_spline - y_actual[:len(y_spline)]) / y_actual[:len(y_spline)])) * 100
                    # Guardar resultados en la lista
                    resultados_error.append({
                        'Método': 'Spline Cúbico',
                        'Error Cuadrático Medio (MSE)': mse_spline,
                        'Error Relativo (%)': error_relativo_spline
                    })
                except Exception as e:
                    errores['Spline Cubico'] = str(e)
            
                
            if 'B-Spline' in metodos_interpolacion:
                try:
                    y_actual = y_values.values  # Los valores originales
                    tck, u = splprep([x_values_scaled, y_values], k=5, s=0)
                    u_new = np.linspace(0, 1, 1096)
                    out = splev(u_new, tck)
                    # La salida es una lista, por lo que tomamos out[1] como el valor y
                    y_bspline = out[1]
                    # Asegurarse de que las longitudes coincidan
                    min_length = min(len(y_actual), len(y_bspline))
                    y_actual = y_actual[:min_length]
                    y_bspline = y_bspline[:min_length]
                    fig.add_scatter(x=x_new_dates[:min_length], y=y_bspline, mode='lines', name='B-Spline', line=dict(color='orange'))
                    # Calcular errores para B-Spline
                    mse_bspline = np.mean((y_bspline - y_actual) ** 2)
                    error_relativo_bspline = np.mean(np.abs((y_bspline - y_actual) / y_actual)) * 100 if np.any(y_actual) else float('inf')
                    # Guardar resultados en la lista
                    resultados_error.append({
                        'Método': 'B-Spline',
                        'Error Cuadrático Medio (MSE)': mse_bspline,
                        'Error Relativo (%)': error_relativo_bspline
                    })
                except Exception as e:
                    errores['B-Spline'] = str(e)
            
            # Crear un DataFrame para almacenar los resultados
            df_interpolado = pd.DataFrame({'Fecha': x_new_dates})
   


            # Ajuste de Mínimos Cuadrados Polinomico
            if 'Mínimos Cuadrados Polinomico' in metodos_ajuste:
                # Seleccionar grado del polinomio para Mínimos Cuadrados
                grado_polinomio = st.slider('Selecciona el grado del polinomio para Mínimos Cuadrados:', min_value=1, max_value=10, value=3)
                poly_coeff = np.polyfit(x_values_scaled, y_values, deg=grado_polinomio)
                poly_func = np.poly1d(poly_coeff)
                y_poly = poly_func(x_new)
                fig.add_scatter(x=x_new_dates, y=y_poly, mode='lines', name='Mínimos Cuadrados Polinómico', line=dict(color='magenta'))
                df_interpolado['Mínimos Cuadrados Polinómico'] = y_poly
                # Mostrar la función polinómica
                func_str = "g(x) = " + " + ".join([f"{coef:.3f}*x^{deg}" for deg, coef in enumerate(poly_coeff[::-1])])
                st.markdown(f"### Función Polinómica Encontrada:")
                st.markdown(func_str)
                
            if 'Máxima Verosimilitud' in metodos_ajuste:
                try:
                    y_mvl, params_mvl = metodo_maxima_verosimilitud(x_values_scaled, y_values)
                    fig.add_scatter(x=x_new_dates, y=y_mvl, mode='lines', name='Máxima Verosimilitud', line=dict(color='orange'))
                    df_interpolado['Máxima Verosimilitud'] = y_mvl
                    mse_mvl = np.mean((y_mvl - y_values[:len(y_mvl)]) ** 2)
                    # Calcular el R² (coeficiente de determinación)
                    ss_total = np.sum((y_values[:len(y_mvl)] - np.mean(y_values[:len(y_mvl)])) ** 2)
                    ss_residual = np.sum((y_values[:len(y_mvl)] - y_mvl) ** 2)
                    r_squared = 1 - (ss_residual / ss_total)
                    # Mostrar los parámetros del modelo
                    st.markdown("### Parámetros del Modelo de Máxima Verosimilitud:")
                    st.write(f"Media (mu): {params_mvl[0]:.2f}")
                    st.write(f"Desviación Estándar (sigma): {params_mvl[1]:.2f}")
                    st.write(f"Amplitud: {params_mvl[2]:.2f}")
                    # Mostrar el MSE
                    st.write(f"Error Cuadrático Medio (MSE): {mse_mvl:.4f}")
                    # Mostrar el R²
                    st.write(f"Coeficiente de Determinación (R²): {r_squared:.4f}")
                    #Graficar residuos
                    residuos = y_values[:len(y_mvl)] - y_mvl  # Asegúrate de que ambas longitudes coincidan
                    fig_residuos = px.scatter(x=x_new_dates, y=residuos, labels={'x': 'Fecha', 'y': 'Residuos'}, title='Residuos de Máxima Verosimilitud')
                    fig_residuos.add_scatter(x=x_new_dates, y=np.zeros_like(residuos), mode='lines', line=dict(color='red', dash='dash'), name='Cero')
                    # Mostrar el gráfico de residuos
                    st.plotly_chart(fig_residuos)
                except Exception as e:
                    st.error(f"Error en el ajuste de Máxima Verosimilitud: {e}")
                
                
            # Ajuste de Ridge
            if 'Ridge' in metodos_ajuste:
                ridge_model = Ridge(alpha=1.0)
                x_values_reshaped = x_values_scaled.values.reshape(-1, 1)  # Convertir a array y luego reshaping
                ridge_model.fit(x_values_reshaped, y_values)
                y_ridge = ridge_model.predict(x_new.reshape(-1, 1))
                fig.add_scatter(x=x_new_dates, y=y_ridge, mode='lines', name='Ridge', line=dict(color='brown'))
                df_interpolado['Ridge'] = y_ridge
                # Cálculo de métricas
                mse_ridge = np.mean((y_values - ridge_model.predict(x_values_reshaped)) ** 2)
                r2_ridge = r2_score(y_values, ridge_model.predict(x_values_reshaped))
                mae_ridge = mean_absolute_error(y_values, ridge_model.predict(x_values_reshaped))
                # Almacenar los resultados
                metricas_resultados.append({"Método": "Ridge", "MSE": mse_ridge, "R²": r2_ridge, "MAE": mae_ridge})
                # Mostrar tabla de resultados en Streamlit
                df_metricas = pd.DataFrame(metricas_resultados)
                st.table(df_metricas)
            # Ajuste de Lasso
            if 'Lasso' in metodos_ajuste:
                lasso_model = Lasso(alpha=1.0)
                x_values_reshaped = x_values_scaled.values.reshape(-1, 1)  # Convertir a array y luego reshaping
                lasso_model.fit(x_values_reshaped, y_values)
                y_lasso = lasso_model.predict(x_new.reshape(-1, 1))
                fig.add_scatter(x=x_new_dates, y=y_lasso, mode='lines', name='Lasso', line=dict(color='pink'))
                df_interpolado['Lasso'] = y_lasso
                # Calcular métricas
                mse_lasso = mean_squared_error(y_values, y_lasso)
                r2_lasso = r2_score(y_values, y_lasso)
                mae_lasso = mean_absolute_error(y_values, y_lasso)
                # Almacenar las métricas en el diccionario
                metrics['Lasso'] = {'MSE': mse_lasso, 'R²': r2_lasso, 'MAE': mae_lasso}
                # Crear la tabla de resultados
                results_df = pd.DataFrame(metrics).T.reset_index()
                results_df.columns = ['Método', 'MSE', 'R²', 'MAE']
                # Mostrar la tabla en Streamlit
                st.write("### Tabla de Resultados")
                st.table(results_df)
            # Ajuste de Levenberg-Marquardt
            if 'Levenberg-Marquardt' in metodos_ajuste:
                def model(x, a, b, c):
                    return a * x**b + c
                
                try:
                    x_values_numeric = x_values.astype(np.int64) // 10**9
                    popt, _ = curve_fit(model, x_values_numeric, y_values, p0=(1, 1, 1))
                    y_lm = model(x_new_dates.astype(np.int64) // 10**9, *popt)
                    # Mostrar los parámetros del modelo
                    st.markdown("### Parámetros del Modelo de Levenberg-Marquardt:")
                    # Calcular MSE y R²
                    mse_lm = np.mean((y_lm - y_values[:len(y_lm)]) ** 2)
                    ss_total = np.sum((y_values[:len(y_lm)] - np.mean(y_values[:len(y_lm)])) ** 2)
                    ss_residual = np.sum((y_values[:len(y_lm)] - y_lm) ** 2)
                    r_squared = 1 - (ss_residual / ss_total)
                    # Mostrar errores
                    st.write(f"Error Cuadrático Medio (MSE): {mse_lm:.4f}")
                    st.write(f"Coeficiente de Determinación (R²): {r_squared:.4f}")
                    fig.add_scatter(x=x_new_dates, y=y_lm, mode='lines', name='Levenberg-Marquardt', line=dict(color='orange'))
                    df_interpolado['Levenberg-Marquardt'] = y_lm
                except Exception as e:
                    st.error(f"Error en el ajuste de Levenberg-Marquardt: {e}")
            # Mostrar el gráfico
            st.plotly_chart(fig)

            if errores:
                st.warning("Se produjeron algunos errores en los métodos de interpolación:")
                for metodo, error in errores.items():
                    st.write(f"{metodo}: {error}")

            # Opción para descargar los datos
            st.markdown('---')
            st.subheader('Descarga de Datos 📥')
            st.write("Puedes descargar los datos de la interpolación en un archivo Excel.")
            generar_enlace_descarga_excel(df_interpolado)

            st.markdown('---')
            st.subheader('Descarga de Gráfico 📊')
            generar_enlace_descarga_html(fig)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
