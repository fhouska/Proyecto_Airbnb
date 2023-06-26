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
st.image(image, caption='',width=400)

st.title ("**Inisde Airbnb: ESTABUL**")




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
    st.subheader('aca va Análisis Exploratorio {selected}')

    tab1, tab2 , tab3, tab4, tab5= st.tabs(["Procesamiento de Datos", "Correlación de las Variables", "Pasajeros","Clases y Lugar de Embarque","Conclusiones"])
    with tab1:
# ------- COL-------------------------------------------------------------#
        col1, col2 = st.columns(2)
        with col1:
            feq=df['neighbourhood'].value_counts().sort_values(ascending=True)
            fig1 = px.bar(feq, 
                orientation='h', 
                title = "Number of listings by neighbourhood", 
                template= "plotly_dark",
                color_discrete_sequence = colors)
            st.plotly_chart(fig1)
             
        with col2:
            price_mean = df.groupby('neighbourhood')['price_euro'].mean().round(2).sort_values(ascending=True)
            fig2 = px.bar(price_mean, orientation='h', title = "Number of listings by neighbourhood", template= "plotly_dark")
            st.plotly_chart(fig2)
    
    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            dfneighbourhood = pd.DataFrame(feq)
            dfneighbourhood = dfneighbourhood.reset_index()
            adam = gpd.read_file("data/neighbourhoods.geojson")
            fig3 = px.choropleth_mapbox(dfneighbourhood, geojson=adam, featureidkey='properties.neighbourhood',locations ="neighbourhood",color = 'count', 
                            color_continuous_scale="portland", title="Neighbourhoods in Istambul",zoom=10, hover_data = ['neighbourhood','count'],
                            mapbox_style="carto-positron",width=1000, height=750,center = {"lat": 41.0036, "lon": 28.5737})
            fig3.update(layout_coloraxis_showscale=True)
            fig3.update_layout( paper_bgcolor="#1f2630",font_color="white",title_font_size=20, title_x = 0.5)
            st.plotly_chart(fig3)


        with col4:
            """aca va algo"""

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




