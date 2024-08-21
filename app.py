from dash import Dash, html, dcc, Output, Input  # pip install dash
import pandas as pd  # pip install pandas
import random
import plotly.express as px
from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo
import scipy  # pip install scipy
import os
# Carga de datos
drug_reviews_drugs_com = fetch_ucirepo(id=462)
data = drug_reviews_drugs_com.data.features
# Filtrado de datos
categorias = ['Depression', 'Pain', 'Anxiety', 'Insomnia', 'High Blood Pressure', 'Migraine']
data = data[data['condition'].isin(categorias)].copy()
data['date'] = pd.to_datetime(data['date'], format='%d-%b-%y')

# Función para borrar datos por porcentaje - Laboratorio: Datos Faltantes
def drop_values(data, percent):
    # Create a copy of the dataframe to avoid modifying the original data.
    data_copy = data.copy()

    # Iterate through each column in the dataframe.
    for column in data_copy.columns:
        # Generate a list of random indices for the column.
        indices = list(range(len(data_copy[column])))
        random.shuffle(indices)

        # Calculate the number of values to drop based on the percentage.
        num_values_to_drop = int(len(data_copy[column]) * percent)

        # Make a copy of the column to avoid SettingWithCopyWarning
        column_copy = data_copy[column].copy()

        # Drop the specified number of values at the random indices.
        column_copy.iloc[indices[:num_values_to_drop]] = None

        # Assign back to the original dataframe
        data_copy[column] = column_copy

    return data_copy

app = Dash(__name__)
# Definición del layout de la aplicación
app.layout = html.Div(
    children=[
        html.H1(children='DRUGS REVIEW:',
                style={'font-family': 'Courier New',
                       'color': 'white', 'font-size': '40px',
                       'text-align': 'center',
                       'letter-spacing': '2px'}),
        html.P(children='Aceptación y Efectividad Basada en Experiencias de Usuarios',
               style={'font-family': 'Courier New',
                      'color': 'white', 'font-size': '20px',
                      'text-align': 'center', 'padding': '0px 0px 1px'}),
        html.H2(children='Rango de selección de data',
                style={'color': 'white', 'font-size': '25px', 'text-align': 'center'}),
        # Primer controlador
        dcc.RangeSlider(
            id='controlador1',
            min=2008,
            max=2017,
            step=1,
            value=[2008, 2017],
            marks={year: str(year) for year in range(2008, 2018)},
        ),
        # Primeras 3 Gráficas, modificadas con el controlador
        dcc.Graph(id='grafica1', style={'padding': '10px 50px 20px'}),
        dcc.Graph(id='grafica2', style={'padding': '10px 50px 20px'}),
        dcc.Graph(id='grafica3', style={'padding': '10px 50px 20px'}),
        html.H2(children='Ingrese el porcentaje para la eliminación de datos: 1-80%',
                style={'color': 'white', 'font-size': '25px', 'text-align': 'center'}),
        # Segundo controlador
        html.Div(
            children=[
                dcc.Input(
                    id="controlador2", type="number", placeholder="%",
                    min=1, max=80, step=1),
            ],
            style={'padding': '0 50%'}
        ),
        # Definición de grafica 4
        dcc.Graph(id='grafica4', style={'padding': '10px 50px 20px'}),
        html.H2(id='porcentaje', style={'color': 'white', 'font-size': '25px', 'text-align': 'center'}),
        html.P(children='Este estudio examina la valoración que los pacientes otorgan a las ' +
                        'terapias medicinales a través del tiempo, permitiendo obtener un diagnóstico sobre la efectividad ' +
                        'y los riesgos potenciales de los medicamentos en diferentes condiciones de salud según su categoría. ' +
                        'Para salvaguardar la salud de los pacientes, asegurando el promedio de utilidad y seguridad de los ' +
                        'medicamentos en el mercado.',
               style={'font-family': 'Courier New',
                      'color': 'white', 'font-size': '20px',
                      'text-align': 'justify'}),
    ],
    style={'background': '#111111', 'border-radius': '15px'},
)

# Definición de la función de callback para actualizar las gráficas
@app.callback(
    Output('grafica1', 'figure'),
    Output('grafica2', 'figure'),
    Output('grafica3', 'figure'),
    Input('controlador1', 'value')
)
def update_output(value):
    # Filtración de los datos por la fecha seleccionada
    rango_fecha = range(value[0], value[1] + 1)
    df = data[data['date'].dt.year.isin(rango_fecha)]

    # Gráfico 1: Resumen por condiciones seleccionadas
    conteo_nombres = df['condition'].value_counts()
    df_1 = pd.DataFrame({'condition': conteo_nombres.index, 'value': conteo_nombres.values})
    figure1 = px.pie(df_1, values='value', names='condition', template='plotly_dark', title='Data Disponible por categoría')

    # Gráfico 2: Top 3 Medicamentos Más Usados por Categoría
    # Estos sse agrupan por condicion obteniendo un conteo de las drogas utilizadas por condicion, posteriormente se seleccionan las 3 primeras de cada una
    df_2 = df.groupby('condition')['drugName'].value_counts().groupby(level=0).nlargest(3).reset_index(level=0, drop=True)
    control = {'condition': [], 'drugName': [], 'count': []}
    top_drogas = []
    # De la serie obtenida como df_2, se transforma a un dataframe en un for con control
    for (cond, drug), count in df_2.items():
        control['condition'].append(cond)
        control['drugName'].append(drug)
        control['count'].append(count)
        top_drogas.append(drug)  # Esta lista nos servira para la grafica 3
    figure2 = px.bar(control, x='condition', y='count', color='drugName', text='drugName',
                     title='Top 3 Medicamentos Más Usados por Categoría', template='plotly_dark')
    figure2.update_layout(showlegend=False, xaxis_title='Medicamentos por Categoria', yaxis_title='Uso contabilizado')

    # Grafico 3 - Blox Plot del rating por condicion solo tomando en cuenta las 3 mejores drogas por categoria
    df_3 = df[df['drugName'].isin(top_drogas)]  # Filtramos por la lista obtenida anteriormente
    figure3 = px.box(df_3, x='condition', y='rating', template='plotly_dark',
                     title='Rating por condición tomando en cuenta las 3 mejores drogas',
                     points='all', color='condition',
                     )
    return figure1, figure2, figure3

# Definicion del segundo controlador
@app.callback(
    Output('grafica4', 'figure'),
    Output('porcentaje', 'children'),
    Input('controlador2', 'value')
)
def update_output(value):
    # Obtiene un promedio de utilidad por rating
    df_4 = data.groupby('rating')['usefulCount'].mean().reset_index()
    df_4.columns = ['rating', 'usefulCount']  # Pasa la serie df_4 a ser un dataframe
    val = int(value) / 100 if value is not None else 1  # Este condicional funciona para la primera vez que se carga el Dash, no obtener un valor nulo del controlador
    # Llamado de la imputacion por el val obtenido del controlador
    X_scaled_df_NAN = drop_values(df_4, val)
    data_imputada = pd.DataFrame()
    # Obtencion de la data imputada con el metodo de interpolacion spline, agregando el llenado hacia adelante para asegurar que no quede ningun valor nulo
    data_imputada = X_scaled_df_NAN.interpolate(method="spline", order=1).ffill().bfill()
    # Creación del grafico
    figure4 = px.line(template='plotly_dark')
    # Se agregan dos lineas para cada data frame
    figure4.add_scatter(x=df_4['rating'], y=df_4['usefulCount'], mode='lines', name='DataFrame Original')
    figure4.add_scatter(x=data_imputada['rating'], y=data_imputada['usefulCount'], mode='lines', name='DataFrame Imputado')
    figure4.update_layout(title='Promedio de utilidad de reseñas por el rating dado',
                          xaxis_title='Rating',
                          yaxis_title='UsefulCount')
    if val == 1:  # Manejo de errores para la primera carga del Dash
        val = 0
    text = 'Porcentaje de datos borrados: ' + str(int(val * 100)) + '%'
    return figure4, text


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)
