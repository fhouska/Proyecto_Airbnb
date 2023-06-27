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
image = Image.open('info/estambul.png')
st.image(image, caption='',width=300)
"""

"""
st.title ("**Inisde Airbnb**")
"""
  
"""



# ------- SIDE BAR----------------------------------------------------------#
st.sidebar.title ('Inisde Airbnb')

# ------- MENÚ SIDE BAR-----------------------------------------------------#
with st.sidebar:
    selected = option_menu(
        menu_title= "Menú" ,
        options= ['Introducción','Limpieza de Datos','Análisis Exploratorio','Modelado','Conclusión'], 
        )
if selected == 'Introducción':
    st.subheader('Introducción')
    col1, col2 = st.columns(2)
    with col1:
        image2 = Image.open('info/foto_estambul.jpg')
        st.image(image2, caption='',width=600)
    with col2:
        """ **"Inside Airbnb"** es un sitio web que publica información sobre los alojamientos que se encuentran rentados bajo la plataforma de plataforma 
        Airbnb. Esta web busca mostrar el impacto que tiene esta plataforma en el mercado de la vivienda para cada ciudad.Hemos decidido seleccionar 
        la ciudad de Estambul. Se etrajeron 7 Dataset, 2 con información de las vivendas rentadas y sus propietarios, 2 con datos del vecindario , 
        2 con las reviews y uno con el calandario."""
    
    st.subheader('Correlation Matrix')
    df


if selected == 'Limpieza de Datos':
    st.subheader('aca vaLimpieza de Datos {selected}')

    st.markdown("""El df esta formado por:  41501 filas y 17 columnas 
    Cantidad de valores duplicados:  8
    Cantidad de valores nulos:  118101""", unsafe_allow_html=True,help=None)





    

if selected == 'Análisis Exploratorio':
    st.subheader('Análisis Exploratorio')

    """ aca va una breve intruduccion a como analizamos las variables"""

    tab1, tab2 , tab3, tab4, tab5= st.tabs(["Visualización de datos", "Distritos", "Precios","Alojamientos","Porpietarios"])

# TABLEREO POWER BI
    with tab1: 
        """ aca va el tablero BI"""

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
                height=800    
                )
            fig1.update_layout(
                title='Average price by neighborhoods',
                xaxis=dict(title='Average price'),
                yaxis=dict(title='rneighborhoods'),
                title_font_size=20,
                showlegend=False
                )
            st.plotly_chart(fig1)
             
        with col2:
            dfneighbourhood = pd.DataFrame(neighbourhood)
            dfneighbourhood = dfneighbourhood.reset_index()
            adam = gpd.read_file("data/neighbourhoods.geojson")
            fig2 = px.choropleth_mapbox(dfneighbourhood, geojson=adam, featureidkey='properties.neighbourhood',locations ="neighbourhood",color = 'count', 
                                        color_continuous_scale='magma', title="Districts of Istanbul",zoom=10, hover_data = ['neighbourhood','count'],
                                        mapbox_style="carto-positron",width=1000, height=800,center = {"lat": 41.0036, "lon": 28.9737})
            fig2.update(layout_coloraxis_showscale=True)
            fig2.update_layout( paper_bgcolor="#fff",font_color="#AF1D56",title_font_size=20, title_x = 0.2)
            st.plotly_chart(fig2)

# ANALISIS PRECIOS 
    with tab3:
        """
        Precios: Este gráfico se muestran los valores en euros para mejor entendimiento y dimensión. 
        El precio medio de alojamientos en Estambul es ₺2.007,34 (liras turcas) que equivalen a €91,12 euros al tipo de cambio del 26/06/23.
        El distrito con mayor valor de media es el distrito de **Beylikduzu** con **€344,29**
        Para los distritos con mayor distribución son:
        * Beyoglu: € 88,11
        * Sisli: € 82,55
        * Kadikoy: € 108,3
        * Fatih: € 91,03

        """
        st.subheader(' ')
        price_mean = df.groupby('neighbourhood')['price_euro'].mean().round(2).sort_values(ascending=True)
        dfprecio = pd.DataFrame(price_mean)
        dfprecio = dfprecio.reset_index()
        #ahora graficamos con plotly
        fig3 = px.area(dfprecio, x="neighbourhood", y="price_euro",
                template= "plotly_dark", 
                title = "Average daily price based on location in Amsterdam",
                color_discrete_sequence = [colors[2]], 
                )

        fig3.update_layout(
        title='Average price by neighborhoods',
        xaxis=dict(title='average price'),
        yaxis=dict(title='neighborhoods'),
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
            title='Room_type',
            showlegend=False
            )
            st.plotly_chart(fig5)

        with col2:
            propertytype=df['property_type'].value_counts().sort_values(ascending=False).head(15)
            fig6 = px.bar(propertytype, 
                template= "plotly_dark",
                color_discrete_sequence = [colors[3]],
                #    height=800    
                )
            fig6.update_layout(
                title='First 15 Property Types',
                showlegend=False
                )
            st.plotly_chart(fig6)
        
        rooms = df.groupby(['neighbourhood', 'room_type','property_type','accommodates']).size().reset_index(name='Count')
        # Crear el gráfico de barras agrupadas
        fig4 = px.treemap(rooms, path=['room_type','property_type'], values='Count', 
                            color_discrete_sequence=colors, template='plotly_dark')

        fig4.update_layout(
                title='Sobrevivientes por Clase',
                xaxis=dict(title='Edad'),
                yaxis=dict(title='Cantidad'),
                bargap=0.1,
                width=1300,
                showlegend=True
            )
        st.plotly_chart(fig4)

# ANALISIS PROPIETARIOS




 
    

if selected == 'Modelado':
    st.subheader('aca va Modelado {selected}')
    
    model = load_model('ml_airbnb')

    st.title('Predicción de Precios de Airbnb en Estambul')

    neighbourhood = st.selectbox('Barrio', options=[
        'De Baarsjes - Oud-West', 'De Pijp - Rivierenbuurt', 'Centrum-West', 'Centrum-Oost',
        'Westerpark', 'Zuid', 'Oud-Oost', 'Bos en Lommer', 'Oostelijk Havengebied - Indische Buurt',
        'Oud-Noord', 'Watergraafsmeer', 'IJburg - Zeeburgereiland', 'Slotervaart', 'Noord-West',
        'Buitenveldert - Zuidas', 'Noord-Oost', 'Geuzenveld - Slotermeer', 'Osdorp',
        'De Aker - Nieuw Sloten', 'Gaasperdam - Driemond', 'Bijlmer-Centrum', 'Bijlmer-Oost'
        ])

    property_type = st.selectbox('Tipo de Propiedad', options=[
        'Apartment', 'Townhouse', 'Houseboat', 'Bed and breakfast', 'Boat',
        'Guest suite', 'Loft', 'Serviced apartment', 'House',
        'Boutique hotel', 'Guesthouse', 'Other', 'Condominium', 'Chalet',
        'Nature lodge', 'Tiny house', 'Hotel', 'Villa', 'Cabin',
        'Lighthouse', 'Bungalow', 'Hostel', 'Cottage', 'Tent',
        'Earth house', 'Campsite', 'Castle', 'Camper/RV', 'Barn',
        'Casa particular (Cuba)', 'Aparthotel'
        ])

    accommodates = st.slider('Número de Personas', min_value=1, max_value=17, value=1)

    room_type = st.selectbox('Tipo de Habitación', options=['Private room', 'Entire home/apt', 'Shared room'])

    maximum_nights = st.slider('Noches Máximas', min_value=1, max_value=100, value=1)

    minimum_nights = st.slider('Noches Mínimas', min_value=1, max_value=10, value=1)

    input_data = pd.DataFrame([[
        neighbourhood, property_type, accommodates, room_type,
        maximum_nights, minimum_nights
    ]], columns=['neighbourhood', 'property_type', 'accommodates', 'room_type', 'maximum_nights', 'minimum_nights'])

    if st.button('¡Descubre el precio!'):
        prediction = predict_model(model, data=input_data)
        st.write(str(prediction["prediction_label"].values[0]) + ' euros')





if selected == 'Conclusión':
        st.subheader('aca va Conclusión {selected}')




