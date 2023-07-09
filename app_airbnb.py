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
df = pd.read_csv('doc_processed/df_limpio.csv')
colors = ['#AF1D56', '#FFDE59', '#CB6CE6', '#FF914D']

# ------- TITULO-----------------------------------------------------------#
# image = Image.open('info/estambul.png')
# st.image(image, caption='',width=300)
"""

"""
st.title ("**Inisde Airbnb - ESTAMBUL**")





# ------- SIDE BAR----------------------------------------------------------#
image = Image.open('Info/estambul.png')
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
        image2 = Image.open('Info/foto_estambul.jpg')
        st.image(image2, caption='',width=550)
    with col2:

        image3 = Image.open('Info/mapa.png')
        st.image(image3, caption='',width=600)
    
    st.subheader('')
    """
    "**Inside Airbnb**" es un sitio web que proporciona información sobre los alojamientos disponibles para alquilar en la plataforma de Airbnb. 
    El objetivo de este sitio web es mostrar el impacto que tiene esta plataforma en el mercado de la vivienda en cada ciudad.
    
    Para este análisis, hemos seleccionado la ciudad de Estambul, conocida por su rica historia, cultura y su gastronomía. A menudo se le llama la "Ciudad de las Mil Mezquitas". 
    
    ¿Sabías que Estambul es la única ciudad en el mundo ubicada entre dos continentes, Asia y Europa?

    **Objetivo**:
    El objetivo de este trabajo es utilizar los dataset de esta web y reaizar un análisis sobre la oferta de alojamientos en Estambul. Examinaremos varios aspectos, como la distribución de los alojamientos, las tendencias de precios, los tipos de alojamientos disponibles y las características de los anfitriones.

    Las preguntas clave que buscamos responder son:
    * ¿Cómo se distribuyen los alojamientos en Estambul en general?
    * ¿Cómo varían los precios en diferentes distritos?
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
    df_null = pd.read_csv('doc_processed/df.csv')
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


    **5 - Transformación y creación de columnas**
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

    tab1, tab2 , tab3, tab4, tab5= st.tabs(["**Visualización de datos**", "**1.Distritos**", "**2.Precios**","**3.Alojamientos**","**4.Anfitriones**"])

# TABLEREO POWER BI
    with tab1: 
        link = '<iframe title="Report Section" width="1340" height="641.5" src="https://app.powerbi.com/view?r=eyJrIjoiM2VkOTQzOTgtMDFlZi00NWY2LWE5MjQtODJkYjMwZWQ5NzNlIiwidCI6IjhhZWJkZGI2LTM0MTgtNDNhMS1hMjU1LWI5NjQxODZlY2M2NCIsImMiOjl9" frameborder="0" allowFullScreen="true"></iframe>'
        st.markdown(link, unsafe_allow_html=True)  


# ANALISIS DISTRITOS

    with tab2:
        """ Estambul está dividida por 39 distritos en los cuales podemos ver, según la gráfica y en el mapa, que la mayoría de los hospedajes se encuentran en los distritos de: Beyoglu, Sisli, Kadikoy y Fatih. 
        Estos distritos tienen características únicas que los hacen atractivos para los visitantes y locales."""
        """
        * **Beyoglu**: es reconocido por ser el lugar donde las comunidades extranjeras establecieron embajadas e iglesias en el pasado. En el siglo XX, este distrito fue testigo de la construcción de 
        grandes hoteles y tiendas de lujo, lo que lo convierte en un centro de elegancia y sofisticación.
        * **Sisli**: se destaca como el distrito de cines y lugares de ocio. Aquí se encuentran diversas salas de conciertos, teatros y otros espacios de entretenimiento. Es un lugar animado y vibrante que ofrece una amplia gama de actividades culturales para disfrutar.
        * **Kadikoy**: por otro lado, es un distrito residencial conocido mercado de pescado y productos agrícolas. Tanto los turistas como los locales aprovechan la oportunidad de explorar este mercado y adquirir especias, tés y frutos secos frescos. Además, Kadikoy cuenta con una gran cantidad de restaurantes, cafeterías y tiendas que reflejan la vida cotidiana de la ciudad.
        * **Fatih**: este distrito alberga el barrio histórico de Estambul. Aquí se encuentra la gran Mezquita de Fatih, una de las más grandes de la ciudad, así como el antiguo acueducto romano y otros impresionantes edificios y monumentos históricos. Fatih es un lugar de gran importancia cultural y atrae a los amantes de la historia y la arquitectura.

        """

# ------- COL-------------------------------------------------------------#
        col1, col2 = st.columns(2)
        with col1:
            neighbourhood_count=df['neighbourhood'].value_counts().sort_values(ascending=True)
            fig1 = px.bar(neighbourhood_count, orientation='h', 
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
            dfneighbourhood = pd.DataFrame(neighbourhood_count)
            dfneighbourhood = dfneighbourhood.reset_index()
            adam = gpd.read_file("data/neighbourhoods.geojson")
            fig2 = px.choropleth_mapbox(dfneighbourhood, geojson=adam, featureidkey='properties.neighbourhood',locations ="neighbourhood",color ='count', 
                                        color_continuous_scale='magma', title="Distritos de Estambul",zoom=10, hover_data =['neighbourhood','count'],
                                        mapbox_style="carto-positron",width=700, height=700,center = {"lat": 41.0036, "lon": 28.9737})
            fig2.update(layout_coloraxis_showscale=True)
            fig2.update_layout(paper_bgcolor="#fff",font_color="#AF1D56",title_font_size=20, title_x = 0.2)
            st.plotly_chart(fig2)

# ANALISIS PRECIOS 
    with tab3:
        """
        En el análisis de precios de los alojamientos en Estambul, se observa que el precio medio es de **TRY 2.007,34** (liras turcas), lo que equivale a 
        aproximadamente **EUR 91,12** al tipo de cambio del 26/06/23.
        
        El distrito de **Beylikduzu** se destaca como el distrito con el mayor precio medio de alojamientos por noche, alcanzando los EUR 344,29. 
        Esto contrasta con los distritos que presentan una mayor cantidad de hospedajes ofrecidos, donde los precios medios son más moderados:
        * Beyoglu: EUR 88,11
        * Sisli: EUR 82,55
        * Kadikoy: EUR 108,3
        * Fatih: EUR 91,03

        Esto puede deberse a que el distrito de Beylikduzu ha experimentado un rápido desarrollo en los últimos años en cuanto lo urbano y comodidades, 
        además presenta una gran demanda de viviendas de alta calidad.

        """
        st.subheader(' ')
        price_mean = df.groupby('neighbourhood')['price_euro'].mean().round(2).sort_values(ascending=True)
        dfprecio = pd.DataFrame(price_mean)
        dfprecio = dfprecio.reset_index()
        #ahora graficamos con plotly
        fig3 = px.scatter(dfprecio, x="neighbourhood", y="price_euro",
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


        dfprice_meand = pd.DataFrame(price_mean)
        dfneighbourhood2 = dfprice_meand.reset_index()
        adam = gpd.read_file("data/neighbourhoods.geojson")
        fig3 = px.choropleth_mapbox(dfneighbourhood2, geojson=adam, featureidkey='properties.neighbourhood',locations ="neighbourhood",color = 'price_euro', 
                                    color_continuous_scale='magma', title="Precio promedio de alojamientos por distrito",zoom=9, hover_data = ['neighbourhood','price_euro'],
                                    mapbox_style="carto-positron",width=1350, height=700,center = {"lat": 41.0035, "lon": 28.9737})
        fig3.update(layout_coloraxis_showscale=True)
        fig3.update_layout( paper_bgcolor="#fff",font_color="#AF1D56",title_font_size=30, title_x = 0.2)
        st.plotly_chart(fig3)





# ANALISIS ALOJAMIENTOS
    with tab4:
        """
        En el análisis de los alojamientos ofrecidos en Estambul, podemos observar a través de estos gráficos la diversidad de opciones disponibles. 
        Se pueden identificar diferentes tipos de habitaciones, así como una amplia variedad de tipos de propiedades.
        
        En cuanto a los **tipos de habitación**, encontramos las siguientes categorías: 
        * Entire home/apt - (**Casa/apartamento completo**)
        * Private room - (**Habitación privada**)
        * Hotel room - (**Habitación de hotel**)
        * Shared room - (**Habitación compartida**)

        Por otro lado, en cuanto a los **tipos de propiedad**, se identifican 109 categorías diferentes. 
        Estas van desde "Entire serviced apartment" (apartamento completo con servicio) hasta "Private room in tent" (habitación privada en tienda de campaña), 
        lo que demuestra la amplia gama de opciones disponibles para los visitantes. Por esta razón en el gráfico se han representado únicamente los primeros 10 tipos de propiedad 
        para proporcionar una muestra representativa de la variedad existente.

        Con el tercer gráfico podemos destacar que el tipo de propiedad que predomina en Estambul es Casa/apartamento completo y la unidad de alquiler completa ("Entire rental unit"), 
        que representa más de la mitad de los alojamientos registrados. 
        Esto indica que la mayoría de los viajeros prefieren tener un espacio privado y completo durante su estancia en la ciudad.
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
                title='Tipos de alojamientos (Top 10)',
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

# ANALISIS ANFITRIONES

    with tab5:
        """
        Se llevó el análisis en función de las calificaciones y reseñas que presentan los anfitriones.

        En primer lugar, se examinó si los huéspedes mostraban preferencia por los anfitriones con la calificación de "Superhost". Sin embargo, 
        se observó que los usuarios no presentaban preferencia por los Superhosts, ya que la mayoría de las reseñas dadas corresponde a los que no tienen 
        calificación Superhost. Dentro de este análisis podemos ver que predominan las calificaciones en el rango de 4 a 5 en ambas categorías.

        Posteriormente, se analizó la distribución de las respuestas de los anfitriones en función de la cantidad de reseñas recibidas. 
        Específicamente, se examinaron los diferentes tipos de respuestas que ofrecen los anfitriones en relación con su tiempo de respuesta. 
        Se observó que aquellos anfitriones que brindaban respuestas dentro de una hora tenían una mayor cantidad de reseñas en comparación con aquellos que 
        tenían tiempos de respuesta más prolongados
        """

        col1, col2 = st.columns(2)
        with col1:
            df['host_is_superhost'] = df['host_is_superhost'].replace({"t": "Superhost", "f": "No_superhost"})
            fig7 = px.box(df, x="review_scores_value",y= 'host_is_superhost',
                        template="plotly_dark",
                        color_discrete_sequence = colors
                        )

            fig7.update_layout(
                title='Distribución de numero de reseñas por calificación superhost',
                xaxis=dict(title='Calificaciones'),
                yaxis=dict(title=''),
                bargap=0.1,
                showlegend=True 
                )
            st.plotly_chart(fig7)
        
        with col2:
            response_time_rates = df.groupby(["host_response_time", 'review_scores_value'])["host_response_rate"].mean().reset_index()
            fig8 = px.histogram(response_time_rates,x='host_response_time' ,y="review_scores_value", 
                    template="plotly_dark", 
                    marginal="box", 
                    hover_data=response_time_rates.columns,
                        color_discrete_sequence = [colors[3]]
                        )
                    
            fig8.update_layout(
                        title='Distribución de puntuaciones de huéspedes por tiempo de respuesta del anfitrión',
                        xaxis=dict(title='Tiempo de respuesta'),
                        yaxis=dict(title='Puntuaciones de huéspedes'),
                        bargap=0.1,
                        showlegend=True 
                        )
            st.plotly_chart(fig8)


 

if selected == 'Modelo Predictivo':
    st.subheader('Modelo Predictivo')
    """
    Realizamos un modelo predictivo de **regresión lineal** para predecir el **precio** de los alojamientos en función 
    de las variables: distrito, tipo de propiedad, cantidad de personas, tipo de habitación, cantidad de noches máximas y mínimas. 
    
    En la implementación de este modelo, utilizamos la biblioteca **pycaret**.

    """
    model = load_model('doc_processed/ml_airbnb')

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
        Después analizar nuestras variables relacionada con la oferta de alojamientos en la ciudad de Estambul, podemos llegar a las siguientes conclusiones:

        * **Oferta de Alojamientos**: En Estambul, se encuentran disponibles un total de mas de 41.000 alojamientos en la plataforma Airbnb. Esta cifra refleja una amplia 
        variedad de opciones para los viajeros que visitan la ciudad.

        * **Distritos Populares**: Los distritos más destacados en términos de oferta de alojamientos son Beyoglu, Sisli, Kadikoy y Fatih. Estos distritos destacan por 
        tener una amplia variedad de opciones para los viajeros.
        
        * **Precio Medio**: El precio medio por noche de alojamiento en Estambul es de €91.12. Esta cifra proporciona una referencia útil para los viajeros que deseen 
        planificar su presupuesto de alojamiento en la ciudad.

        * **Tipo de Alojamiento**: El tipo de alojamiento más comúnmente ofrecido en Estambul es el "Casa/apartamento completo". Esto implica que la mayoría de los alojamientos 
        disponibles son unidades residenciales completas que brindan privacidad y comodidad a los huéspedes.
        
        * **Calificación y Respuestas de los Anfitriones**: Aunque los Superhosts reciben reconocimiento por su excelencia en el servicio, no se observa una preferencia marcada 
        por parte de los huéspedes hacia ellos. Sin embargo, se destaca la importancia de una pronta respuesta por parte de los anfitriones, ya que aquellos que responden rápidamente 
        suelen recibir más reseñas de los huéspedes.

        Estas conclusiones nos brindan una idea general sobre la oferta de alojamientos en Estambul a través de Airbnb y nos ayudan a comprender mejor el mercado de alojamiento en la ciudad.

        En cuanto a futuros análisis, se puede profundizar en el análisis de variables relacionadas con las reseñas:

        * Como el sentimiento de los comentarios y su relación con la calificación de los anfitriones. 
        * También es posible examinar las regulaciones y restricciones aplicables a los anfitriones en la ciudad, 
        lo cual podría proporcionar una perspectiva adicional sobre el mercado de alojamiento en Estambul.

        """

# python -m streamlit run app_airbnb.py



