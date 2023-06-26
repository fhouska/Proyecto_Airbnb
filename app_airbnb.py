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

# ------- CONFIGURACION DE LA PAGINA---------------------------------#
st.set_page_config(page_title="Insideairbnb_Estambul",page_icon="⚓",layout= 'wide')
# """(para que no nos muestre (los waring) lo que cabia de streamlist y nos muestre solo lo que hagamos)"""
st.set_option('deprecation.showPyplotGlobalUse', False) 

# ------- COSAS QUE VAMOS A USAR EN TODA LA APP----------------------#
df = pd.read_csv('df_limpio.csv')
colors = ['#AF1D56', '#FFDE59', '#CB6CE6', '#FF914D']

# ------- TITULO-----------------------------------------------------#
image = Image.open('estambul.png')
st.image(image, caption='',width=400)

st.title ("**Inisde Airbnb: ESTABUL**")




# ------- SIDE BAR-----------------------------------------------------#
st.sidebar.title ('Inisde Airbnb')

with st.sidebar:
    selected = option_menu(
        menu_title= "Menú" ,
        options= ['Introducción','Limpieza de Datos','Análisis Exploratorio','Modelado','Conclusión'], 
        )
if selected == 'Introducción':
    st.subheader('aca va Introducción {selected}')


if selected == 'Limpieza de Datos':
    st.subheader('aca vaLimpieza de Datos {selected}')

    

if selected == 'Análisis Exploratorio':
    st.subheader('aca va Análisis Exploratorio {selected}')

    tab1, tab2 , tab3, tab4, tab5= st.tabs(["Procesamiento de Datos", "Correlación de las Variables", "Pasajeros","Clases y Lugar de Embarque","Conclusiones"])
    with tab1:
# ------- COL-----------------------------------------------------#
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

if selected == 'Conclusión':
    st.subheader('aca va Conclusión {selected}')




