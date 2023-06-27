import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from PIL import Image

#to make graphs
import matplotlib.pyplot as plt
import seaborn as sns

#to make the plotly graphs
import plotly.graph_objs as go
import plotly.express as px

#to make maps
import geopandas as gpd
from branca.colormap import LinearColormap
import folium

# warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Modelo
from pycaret.regression import load_model, predict_model

# ------- CONFIGURACION DE LA PAGINA---------------------------------------#
st.set_page_config(page_title="Insideairbnb_Estambul",page_icon="📍",layout= 'wide')
# """(para que no nos muestre (los waring) lo que cabia de streamlist y nos muestre solo lo que hagamos)"""
st.set_option('deprecation.showPyplotGlobalUse', False) 

# ------- COSAS QUE VAMOS A USAR EN TODA LA APP----------------------------#
df = pd.read_csv('df_limpio.csv')
colors = ['#AF1D56', '#FFDE59', '#CB6CE6', '#FF914D']

# ------- TITULO-----------------------------------------------------------#
# image = Image.open('info/estambul.png')
# st.image(image, caption='',width=300)
"""

"""
st.title ("**Inisde Airbnb**")





# ------- SIDE BAR----------------------------------------------------------#
image = Image.open('info/estambul.png')
st.sidebar.image(image, caption='',width=300)
st.sidebar.title ('Inisde Airbnb')

# ------- MENÚ SIDE BAR-----------------------------------------------------#
with st.sidebar:
    selected = option_menu(
        menu_title= "Menú" ,
        options= ['Introducción','Limpieza de Datos','Análisis Exploratorio','Modelo Predictivo','Conclusión'], 
        )
if selected == 'Introducción':
    st.subheader('Introducción')

    col1, col2 = st.columns(2)
    with col1:
        image2 = Image.open('info/foto_estambul.jpg')
        st.image(image2, caption='',width=550)
    with col2:

        image3 = Image.open('info/mapa.png')
        st.image(image3, caption='',width=600)
    
    st.subheader('')
    """
    "**Inside Airbnb**" es un sitio web que proporciona información sobre los alojamientos disponibles para alquilar en la plataforma de Airbnb. El objetivo de este sitio web es mostrar el impacto que tiene esta plataforma en el mercado de la vivienda en cada ciudad. 
    Para este análisis, hemos seleccionado la ciudad de Estambul, conocida por su rica historia, cultura y su gastronomía. A menudo se le llama la "Ciudad de las Mil Mezquitas". 
    
    ¿Sabías que Estambul es la única ciudad en el mundo ubicada entre dos continentes, Asia y Europa?

    **Objetivo**:
    El objetivo de este análisis es explorar la oferta de alojamientos en Estambul. Examinaremos varios aspectos, como la distribución de los alojamientos, las tendencias de precios, los tipos de alojamientos disponibles y las características de los anfitriones.

    Las preguntas clave que buscamos responder son:
    * ¿Cómo se distribuyen los alojamientos en Estambul en general?
    * ¿Cómo varían los precios en diferentes tipos de alojamientos?
    * ¿Cuáles son los tipos de alojamientos más comunes en Estambul?
    * ¿Qué podemos aprender sobre los anfitriones de estos alojamientos?

    Al abordar estas preguntas, buscamos obtener información sobre la dinámica del mercado de Airbnb en Estambul y comprender mejor las características del panorama de alojamientos en la ciudad

    """



if selected == 'Limpieza de Datos':
    st.subheader('Limpieza de Datos')

    """ En el proceso de análisis, se utilizaron 3 de los 7 archivos obtenidos de la web:
    * listings.cvs 
    * listings_details.cvs
    * calendar.csv
    * reviews.csv
    * reviews_details.csv
    * neighbourhoods.csv
    * neighbourhoods.geojson

    Se utilizaron los archivos "listings.csv", "listings_details.csv" y "neighbourhoods.geojson" para realizar el análisis, construir el modelo predictivo y crear los mapas.
    
    A continuación, se describen los pasos realizados en el proceso de limpieza de datos:

    **1 - Selección de columnas**:
    Dado que este dataframe "listings_details" tiene 74 columnas, se selccionaron las columnas que serán necesarias para el análisis:
    """
    code = '''columna =["property_type", "accommodates", "first_review", "review_scores_value", "review_scores_cleanliness", "review_scores_location", "review_scores_accuracy", "review_scores_communication", "review_scores_checkin", "review_scores_rating", "maximum_nights", "listing_url", "host_is_superhost", "host_about", "host_response_time", "host_response_rate", "host_listings_count","number_of_reviews_ltm","reviews_per_month"]'''
    st.code(code,language='python')

    """**2 -	Merge: listings & listings_details**:
    Se realizó proceso de fusión (merge) entre los conjuntos de datos "listings" y "listings_details". Esta fusión se llevó a cabo con el objetivo de combinar la información relevante de ambos conjuntos en un solo dataframe, para facilitar el análisis posterior. 
    """
    code = '''df = pd.merge(listings, listings_details[columns], on='id', how='left')'''
    st.code(code,language='python')

    """
    **3 - Identificación de duplicados**: No se identificaron valores duplicados.

    **4 - Identificación de valores nulos**: Los valores nulos obtenidos se presentarán en la siguiente tabla:
    """
    df_null = pd.read_csv('df.csv')
    df_null_percentage = pd.isnull(df_null).sum()/len(df_null)*100 #Calculamos el % del los datos faltantes en cada columna
    nulos_totales= df_null_percentage.sort_values(ascending = False).round(2) #Ordenamos de mayor a menor
    nulos_totales = pd.DataFrame(nulos_totales,columns=["% nulos"])

    # df_null1 = df_null.isnull().sum().sort_values(ascending=False).reset_index(name='Count')
    st.write(nulos_totales,height=100,width=550 )

    """
    * Se eliminan las columnas:
        * *neighbourhood_group* y *license* debido a la falta de datos.
        * *host_about* ya que contiene la descripción proporcionada por los anfitriones sobre sí mismos, la cual no se consideró relevante para el análisis
    * En la columna *name*, se aplicó la función *fillna()* para reemplazar los valores nulos por la palabra "no name".
    * En cuanto a las demás columnas que presentan valores nulos y están relacionadas con las reviews, se optó por mantenerlas y reemplazar los valores nulos, ya que representan métricas opcionales y no absolutas.


    **5 - Transformación de columnas**
    * **host_response_rate**: se realizó la conversión a tipo numérico utilizando la función *to_numeric()*. 
    Esta columna representa el porcentaje de respuestas del anfitrión, y al convertirla a tipo numérico, nos permitirá realizar cálculos y análisis más precisos.
    * **price_euro**: se creó una nueva variable que representa el precio en euros. 
    Esta transformación se realizó a partir de la columna "price" que se encuentra en la moneda local (Lira turca). 
    Para realizar esta conversión, se utilizó el tipo de cambio proporcionado por la web de la Unión Europea, específicamente el tipo de cambio para 
    la fecha 26/06/2023, donde 1 TRY (Lira turca) equivale a 0.04564 EUR (euros). De esta manera, obtenemos una mejor comprensión y percepción del precio.


    Datasets: http://insideairbnb.com/get-the-data.html 

    Tipos de cambio: https://commission.europa.eu/funding-tenders/procedures-guidelines-tenders/information-contractors-and-beneficiaries/exchange-rate-inforeuro_es

    """

if selected == 'Análisis Exploratorio':
    st.subheader('Análisis Exploratorio')

    """ Para responder a las preguntas planteadas al inicio, se dividió el análisis en cuatro secciones principales. 
    Además, se creó un tablero de Power BI para obtener una visión resumida de los datos. 
    
    A continuación, se detallan las secciones del análisis y la funcionalidad del tablero de Power BI."""

    tab1, tab2 , tab3, tab4, tab5= st.tabs(["Visualización de datos", "1. Distritos", "2. Precios","3. Alojamientos","4. Porpietarios"])

# TABLEREO POWER BI
    with tab1: 
        link = '<iframe title="Airbnb_Estambul" width="1440" height="841.25" src="https://app.fabric.microsoft.com/reportEmbed?reportId=d15b01e9-003c-41df-aeb4-a4728af89d08&autoAuth=true&ctid=8aebddb6-3418-43a1-a255-b964186ecc64" frameborder="0" allowFullScreen="true"></iframe>'
        st.markdown(link, unsafe_allow_html=True)  


# ANALISIS DISTRITOS

    with tab2:
        """ Estambul esta dividida por **39 distritos** en los cuales podemos ver según la gráfica y en el mapa que la mayoría de los hospedajes 
        se encuentran en los distritos de: Beyoglu, Sisli, Kadikoy y Fatih."""
        """
        * **Beyoglu**: es la zona donde las comunidades extranjeras establecieron las embajadas y las iglesias, y donde en el siglo XX se levantaron grandes hoteles y tiendas más lujosas.
        * **Sisli**: es el distrito de cines y lugares de ocio. Aquí se encuentran entre otras cosas salas de conciertos y teatros.
        * **Kadikoy**: es un distrito residencial y es conocido por su mercado de pescado y productos agrícolas y por lo general los turistas y locales aprovecha a comprar especias, tés o frutos secos.
        * **Fatih**: es el distrito donde se encuentra su barrio histórico, aquí se encuentra La mezquita de Fatih, que es una de las mas grandes de la ciudad, así como también el acueducto romano ente otras tantos edificios y monumentos históricos.

        """

# ------- COL-------------------------------------------------------------#
        col1, col2 = st.columns(2)
        with col1:
            neighbourhood=df['neighbourhood'].value_counts().sort_values(ascending=True)
            fig1 = px.bar(neighbourhood, orientation='h', 
                template= "plotly_dark",
                color_discrete_sequence = [colors[3]],
                height=700,
                width=600    
                )
            fig1.update_layout(
                title='Distribución de distritos',
                xaxis=dict(title='Cantidades'),
                yaxis=dict(title=''),
                title_font_size=20,
                showlegend=False
                )
            st.plotly_chart(fig1)
             
        with col2:
            dfneighbourhood = pd.DataFrame(neighbourhood)
            dfneighbourhood = dfneighbourhood.reset_index()
            adam = gpd.read_file("data/neighbourhoods.geojson")
            fig2 = px.choropleth_mapbox(dfneighbourhood, geojson=adam, featureidkey='properties.neighbourhood',locations ="neighbourhood",color = 'count', 
                                        color_continuous_scale='magma', title="Distritos de Estambul",zoom=10, hover_data = ['neighbourhood','count'],
                                        mapbox_style="carto-positron",width=700, height=700,center = {"lat": 41.0036, "lon": 28.9737})
            fig2.update(layout_coloraxis_showscale=True)
            fig2.update_layout( paper_bgcolor="#fff",font_color="#AF1D56",title_font_size=20, title_x = 0.2)
            st.plotly_chart(fig2)

# ANALISIS PRECIOS 
    with tab3:
        """
        Precios: Este gráfico se muestran los valores en euros para mejor entendimiento y dimensión. 
        El precio medio de alojamientos en Estambul es **TRY 2.007,34  (liras turcas)** que equivalen a **EUR 91,12 ** euros al tipo de cambio del 26/06/23.
        El distrito con mayor valor de media es el distrito de **Beylikduzu** con **€344,29**
        Para los distritos con mayor distribución son:
        * Beyoglu: EUR 88,11
        * Sisli: EUR 82,55
        * Kadikoy: EUR 108,3
        * Fatih: EUR 91,03

        """
        st.subheader(' ')
        price_mean = df.groupby('neighbourhood')['price_euro'].mean().round(2).sort_values(ascending=True)
        dfprecio = pd.DataFrame(price_mean)
        dfprecio = dfprecio.reset_index()
        #ahora graficamos con plotly
        fig3 = px.area(dfprecio, x="neighbourhood", y="price_euro",
                template= "plotly_dark",
                color_discrete_sequence = [colors[2]], 
                )

        fig3.update_layout(
        title='Precio promedio de alojamientos por distrito',
        xaxis=dict(title='distritos'),
        yaxis=dict(title='euros'),
        showlegend=False,
        width=1300,
                )
        st.plotly_chart(fig3)

# ANALISIS ALOJAMIENTOS
    with tab4:
        """
        Con este gráfico podemos ver los diferentes alojamientos ofrecidos.
        
        Por un lado, están diferentes ***tipos de habitación***: 
        * Entire home/apt - (**Casa/apartamento completo**)
        * Private room - (**Habitación privada**)
        * Hotel room - (**Habitación de hotel**)
        * Shared room - (**Habitación compartida**)

        Luego están los diferentes ***tipos de propiedad***: en los que se pueden diferenciar 109 diferentes tipos.

        Por lo que podemos tener: “Entire serviced apartment”, que corresponden a apartamento completo con servicio hasta “Private room in tent” que son Habitación privada en tienda de campaña.
        Lo que se puede ver que lo que mas se predomina es Casa/apartamento completo con mas de la mitad de los datos.
        """



        col1, col2 = st.columns(2)
        with col1:
            fig5 = px.histogram(df,'room_type',
             nbins = 20,
             template= "plotly_dark",
             color_discrete_sequence = [colors[1]])

            fig5.update_layout(
            title='Tipos de habitación',
            showlegend=False
            )
            st.plotly_chart(fig5)

        with col2:
            propertytype=df['property_type'].value_counts().sort_values(ascending=False).head(10)
            fig6 = px.bar(propertytype, 
                template= "plotly_dark",
                color_discrete_sequence = [colors[3]],
                #    height=800    
                )
            fig6.update_layout(
                title='Tipos de habitación (Top 10)',
                showlegend=False
                )
            st.plotly_chart(fig6)
        
        rooms = df.groupby(['neighbourhood', 'room_type','property_type','accommodates']).size().reset_index(name='Count')
        # Crear el gráfico de barras agrupadas
        fig4 = px.treemap(rooms, path=['room_type','property_type'], values='Count', 
                            color_discrete_sequence=colors, template='plotly_dark')

        fig4.update_layout(
                title='Distribución por tipo de habitación y tipo de propiedad',
                xaxis=dict(title='Edad'),
                yaxis=dict(title='Cantidad'),
                bargap=0.1,
                width=1300,
                showlegend=True
            )
        st.plotly_chart(fig4)

# ANALISIS PROPIETARIOS

    with tab5:

        fig7 = px.box(df, x="review_scores_value",y= 'host_is_superhost',
                    template="plotly_dark",
                    color_discrete_sequence = colors
                    )

        fig7.update_layout(
            title='Numero de Reseñas',
            xaxis=dict(title='Calificaciones'),
            yaxis=dict(title=''),
            bargap=0.1,
            showlegend=True 
            )
        st.plotly_chart(fig7)


 

if selected == 'Modelo Predictivo':
    st.subheader('Modelo Predictivo')
    """
    En la implementación de este modelo, utilizamos la biblioteca **pycaret**. Nuestro objetivo principal es predecir la variable **"precio_euro"**,
    por lo que la configuramos como nuestra variable objetivo. Para lograr esto, elegimos el modelo de regresión lineal..

    """
    model = load_model('ml_airbnb')

    st.title('Predicción de Precios de Airbnb en Estambul')

    neighbourhood = st.selectbox('Barrio', options=['Besiktas', 'Beyoglu', 'Sisli', 'Sariyer', 'Fatih', 'Uskudar',
       'Kadikoy', 'Kagithane', 'Basaksehir', 'Bagcilar', 'Maltepe',
       'Esenyurt', 'Beykoz', 'Cekmekoy', 'Sancaktepe', 'Atasehir',
       'Tuzla', 'Pendik', 'Bahcelievler', 'Kartal', 'Beylikduzu',
       'Bakirkoy', 'Adalar', 'Gaziosmanpasa', 'Zeytinburnu',
       'Kucukcekmece', 'Umraniye', 'Eyup', 'Gungoren', 'Avcilar', 'Sile',
       'Arnavutkoy', 'Buyukcekmece', 'Bayrampasa', 'Catalca', 'Esenler',
       'Silivri', 'Sultangazi', 'Sultanbeyli'
        ])

    property_type = st.selectbox('Tipo de Propiedad', options=['Entire rental unit', 'Private room in loft',
       'Entire serviced apartment', 'Private room in home',
       'Private room in rental unit', 'Entire home',
       'Room in serviced apartment', 'Entire loft',
       'Private room in villa', 'Private room in serviced apartment',
       'Entire condo', 'Private room in townhouse', 'Shared room in home',
       'Entire cabin', 'Camper/RV', 'Room in aparthotel',
       'Shared room in rental unit', 'Room in boutique hotel',
       'Private room in condo', 'Private room', 'Entire villa',
       'Private room in bed and breakfast', 'Entire bed and breakfast',
       'Entire townhouse', 'Entire chalet', 'Room in hotel',
       'Entire cottage', 'Shared room in loft', 'Shared room in hotel',
       'Tiny home', 'Private room in yurt', 'Room in hostel',
       'Room in bed and breakfast', 'Shared room in condo',
       'Private room in treehouse', 'Private room in hostel',
       'Entire vacation home', 'Earthen home', 'Tent',
       'Entire guest suite', 'Entire place', 'Entire hostel',
       'Private room in hut', 'Private room in guesthouse',
       'Private room in casa particular', 'Private room in nature lodge',
       'Boat', 'Shared room in boutique hotel', 'Shared room in barn',
       'Shared room in casa particular', 'Shared room in hostel',
       'Farm stay', 'Private room in guest suite',
       'Private room in tiny home', 'Island', 'Private room in castle',
       'Shared room in villa', 'Yurt', 'Private room in farm stay',
       'Private room in boat', 'Room in pension',
       'Shared room in tiny home', 'Private room in cottage',
       'Room in nature lodge', 'Entire guesthouse',
       'Shared room in townhouse', 'Shared room in bed and breakfast',
       'Casa particular', 'Castle', 'Shared room in guesthouse',
       'Room in heritage hotel', 'Shared room in serviced apartment',
       'Lighthouse', 'Shared room in earthen home',
       'Private room in chalet', 'Shared room in pension', 'Treehouse',
       'Shared room in guest suite', 'Entire bungalow',
       'Private room in bungalow', 'Shared room in aparthotel',
       'Private room in cave', 'Private room in vacation home',
       'Private room in camper/rv', 'Private room in pension',
       'Entire home/apt', 'Pension', 'Shared room', 'Tower',
       'Private room in cabin', 'Houseboat', 'Ice dome',
       'Private room in dome', 'Shared room in boat',
       'Private room in earthen home', 'Shared room in vacation home',
       'Shared room in camper/rv', 'Dome', 'Private room in resort',
       'Shared room in nature lodge', 'Shared room in plane', 'Campsite',
       'Shared room in minsu', 'Shared room in ryokan',
       'Shared room in hut', 'Barn', 'Private room in ryokan',
       'Private room in tent', 'Shipping container', 'Bus',
       'Shared room in farm stay'
        ])

    accommodates = st.slider('Número de Personas', min_value=1, max_value=17, value=1)

    room_type = st.selectbox('Tipo de Habitación', options=['Entire home/apt', 'Private room', 'Hotel room', 'Shared room'])

    maximum_nights = st.slider('Noches Máximas', min_value=1, max_value=100, value=1)

    minimum_nights = st.slider('Noches Mínimas', min_value=1, max_value=10, value=1)

    input_data = pd.DataFrame([[
        neighbourhood, property_type, accommodates, room_type,
        maximum_nights, minimum_nights
    ]], columns=['neighbourhood', 'property_type', 'accommodates', 'room_type', 'maximum_nights', 'minimum_nights'])

    if st.button('¡Descubre el precio!'):
        prediction = predict_model(model, data=input_data)
        st.write(str(prediction["prediction_label"].values[0].round(2)) + ' euros')




if selected == 'Conclusión':
        st.subheader('Conclusión')
        """
        Después de realizar el análisis del dataset, podemos llegar a las siguientes conclusiones sobre la oferta de alojamientos en la ciudad de Estambul:

        * En total, hay 41,500 alojamientos disponibles en la plataforma Airbnb en Estambul.
        * Los distritos con la mayor cantidad de ofertas de alojamiento son: Beyoglu, Sisli, Kadikoy y Fatih. Estos distritos destacan por tener una amplia variedad de opciones para los viajeros.
        * El precio medio por noche de alojamiento en Estambul es de €91.12. Este valor representa el promedio de los precios de todos los alojamientos disponibles en la ciudad.
        * El tipo de alojamiento más comúnmente ofrecido en Estambul es el ***"Casa/apartamento completo"***. Esto significa que la mayoría de los alojamientos disponibles en la plataforma son viviendas completas que los viajeros pueden reservar y disfrutar en su totalidad.

        Estas conclusiones nos brindan una idea general sobre la oferta de alojamientos en Estambul a través de Airbnb y nos ayudan a comprender mejor el mercado de alojamiento en la ciudad.

        """



