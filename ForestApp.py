# -*- coding: utf-8 -*-


import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

tree_model = pickle.load(open('tree_trained_model.sav','rb'))
rfo_model = pickle.load(open('rfo_trained_model.sav','rb'))

stdscaler = pickle.load(open('stdscaler.sav','rb'))

#creating function
def forest_prediction(input_data, classifier_name):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    print(input_data_reshaped)

    std_data = stdscaler.transform(input_data_reshaped)

    if (classifier_name=="Decision Tree"):
        prediction = tree_model.predict(std_data)
    elif (classifier_name=="Random Forest"):
        prediction = rfo_model.predict(std_data) 

    print(prediction)

    if (prediction[0] == 1):
      return 'Spruce/Fir Forest Type'
    elif (prediction[0] == 2):
      return 'Lodgepole Pine Forest Type'
    elif (prediction[0] == 3):
      return 'Ponderosa Pine Forest Type'
    elif (prediction[0] == 4):
      return 'Cottonwood/Willow Forest Type'
    elif (prediction[0] == 5):
      return 'Aspen Forest Type'
    elif (prediction[0] == 6):
      return 'Douglas-fir Forest Type'
    elif (prediction[0] == 7):
      return 'Krummholz Forest Type'
    
    return prediction
    

    
    
def main():
    
    st.title('Forest Cover Type Prediction')
    st.subheader('Use cartographic variables to classify forest categories')
    st.markdown('___')

    classifier_name = st.sidebar.selectbox('Select classifier',('Decision Tree', 'Random Forest'))

    Elevation=st.slider('Set the elevation in meters',min_value=1700.0, step=1.0, max_value=4000.0, format="%1f")
    Aspect=st.slider('Set the aspect in degrees azimuth',min_value=0.0, step=1.0, max_value=360.0, format="%1f")
    Slope=st.slider('Set the slope in degrees',min_value=0.0, step=1.0, max_value=70.0, format="%1f")
    Horizontal_Distance_To_Hydrology=st.slider('Set Horizontal Dist to nearest surface water features',min_value=0.0, step=1.0, max_value=1400.0, format="%1f")  
    Vertical_Distance_To_Hydrology=st.slider('Set Vertical Dist to nearest surface water features',min_value=-190.0, step=1.0, max_value=700.0, format="%1f")  
    Horizontal_Distance_To_Roadways=st.slider('Set Horizontal Dist to nearest roadway',min_value=0.0, step=1.0, max_value=7500.0, format="%1f")  
    Hillshade_9am=st.slider('Set Hillshade index at 9am, summer solstice',min_value=0, step=1, max_value=255, format="%f")  
    Hillshade_Noon=st.slider('Set Hillshade index at noon, summer solstice',min_value=0, step=1, max_value=255, format="%f")    
    Hillshade_3pm=st.slider('Set Hillshade index at 3pm, summer solstice',min_value=0, step=1, max_value=255, format="%f")    
    Horizontal_Distance_To_Fire_Points=st.slider('Horizontal Distance to nearest wildfire ignition points',min_value=0.0, step=1.0, max_value=7500.0, format="%1f") 
    Wilderness_Area=st.radio("Wilderness area designation", ('Rawah Wilderness Area','Neota Wilderness Area','Comanche Peak Wilderness Area','Cache la Poudre Wilderness Area'))   
    if (Wilderness_Area=="Rawah Wilderness Area"):
        Wilderness_Area_0 = 1
        Wilderness_Area_1 = 0
        Wilderness_Area_2 = 0
        Wilderness_Area_3 = 0
    elif (Wilderness_Area=="Neota Wilderness Area"):
        Wilderness_Area_0 = 0
        Wilderness_Area_1 = 1
        Wilderness_Area_2 = 0
        Wilderness_Area_3 = 0
    elif (Wilderness_Area=="Comanche Peak Wilderness Area"):
        Wilderness_Area_0 = 0
        Wilderness_Area_1 = 0
        Wilderness_Area_2 = 1
        Wilderness_Area_3 = 0
    elif (Wilderness_Area=="Cache la Poudre Wilderness Area"):
        Wilderness_Area_0 = 0
        Wilderness_Area_1 = 0
        Wilderness_Area_2 = 0
        Wilderness_Area_3 = 1

    Soil_Type = st.selectbox("Soil Type designation", ('0 Cathedral family - Rock outcrop complex, extremely stony','1 Vanet - Ratake families complex, very stony',
    '2 Haploborolis - Rock outcrop complex, rubbly','3 Ratake family - Rock outcrop complex, rubbly','4 Vanet family - Rock outcrop complex complex, rubbly',
    '5 Vanet - Wetmore families - Rock outcrop complex, stony','6 Gothic family','7 Supervisor - Limber families complex','8 Troutville family, very stony',
    '9 Bullwark - Catamount families - Rock outcrop complex, rubbly','10 Bullwark - Catamount families - Rock land complex, rubbly','11 Legault family - Rock land complex, stony',
    '12 Catamount family - Rock land - Bullwark family complex, rubbly','13 Pachic Argiborolis - Aquolis complex','14 unspecified in the USFS Soil and ELU Survey','15 Cryaquolis - Cryoborolis complex','16 Gateview family - Cryaquolis complex',
    '17 Rogert family, very stony','18 Typic Cryaquolis - Borohemists complex','19 Typic Cryaquepts - Typic Cryaquolls complex','20 Typic Cryaquolls - Leighcan family, till substratum complex',
    '21 Leighcan family, till substratum, extremely bouldery','22 Leighcan family, till substratum - Typic Cryaquolls complex','23 Leighcan family, extremely stony','24 Leighcan family, warm, extremely stony',
    '25 Granile - Catamount families complex, very stony','26 Leighcan family, warm - Rock outcrop complex, extremely stony','27 Leighcan family - Rock outcrop complex, extremely stony',
    '28 Como - Legault families complex, extremely stony','29 Como family - Rock land - Legault family complex, extremely stony','30 Leighcan - Catamount families complex, extremely stony',
    '31 Catamount family - Rock outcrop - Leighcan family complex, extremely stony','32 Leighcan - Catamount families - Rock outcrop complex, extremely stony','33 Cryorthents - Rock land complex, extremely stony','34 Cryumbrepts - Rock outcrop - Cryaquepts complex',
    '35 Bross family - Rock land - Cryumbrepts complex, extremely stony','36 Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony','37 Leighcan - Moran families - Cryaquolls complex, extremely stony',
    '38 Moran family - Cryorthents - Leighcan family complex, extremely stony','39 Moran family - Cryorthents - Rock land complex, extremely stony'))

    Soil = int(Soil_Type.split(' ')[0])  # Obtener el número de suelo como entero

    # Crear una lista para almacenar los valores de las variables
    Soil_Types = [0] * 40
    Soil_Types[Soil] = 1  # Establecer el valor correspondiente a la ubicación del suelo

    # Desempaquetar los valores de la lista en variables separadas
    (Soil_Type_0, Soil_Type_1, Soil_Type_2, Soil_Type_3, Soil_Type_4, Soil_Type_5, Soil_Type_6, Soil_Type_7, Soil_Type_8, Soil_Type_9,
     Soil_Type_10, Soil_Type_11, Soil_Type_12, Soil_Type_13, Soil_Type_14, Soil_Type_15, Soil_Type_16, Soil_Type_17, Soil_Type_18, Soil_Type_19,
    Soil_Type_20, Soil_Type_21, Soil_Type_22, Soil_Type_23, Soil_Type_24, Soil_Type_25, Soil_Type_26, Soil_Type_27, Soil_Type_28, Soil_Type_29,
    Soil_Type_30, Soil_Type_31, Soil_Type_32, Soil_Type_33, Soil_Type_34, Soil_Type_35, Soil_Type_36, Soil_Type_37, Soil_Type_38, Soil_Type_39)=Soil_Types

    species = ''

#CAMBIAR LO DEL SOIL TYPE!!


    if st.button('Forest Cover Type Result'):
        species = forest_prediction([Elevation,Aspect,Slope,Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area_0, Wilderness_Area_1, Wilderness_Area_2, Wilderness_Area_3,Soil_Type_0,Soil_Type_1 ,Soil_Type_2 ,Soil_Type_3 ,Soil_Type_4 ,Soil_Type_5 ,Soil_Type_6 ,Soil_Type_7 ,Soil_Type_8 ,Soil_Type_9 ,Soil_Type_10,Soil_Type_11,Soil_Type_12,Soil_Type_13,Soil_Type_14,Soil_Type_15,Soil_Type_16,Soil_Type_17,Soil_Type_18,Soil_Type_19,Soil_Type_20,Soil_Type_21,Soil_Type_22,Soil_Type_23,Soil_Type_24,Soil_Type_25,Soil_Type_26,Soil_Type_27,Soil_Type_28,Soil_Type_29,Soil_Type_30,Soil_Type_31,Soil_Type_32,Soil_Type_33,Soil_Type_34,Soil_Type_35,Soil_Type_36,Soil_Type_37,Soil_Type_38,Soil_Type_39], classifier_name)
        
    st.success(species)
    if (species == 'Spruce/Fir Forest Type'):
        st.image('./Spruce-Fir.jpg')
        st.caption('Spruce/Fir Forest Type') 
    elif (species == 'Lodgepole Pine Forest Type'):
        st.image('./Lodgepole-Pine.jpg')
        st.caption( 'Lodgepole Pine Forest Type')
    elif (species == 'Ponderosa Pine Forest Type'):
        st.image('./PonderosaPine.jpg')
        st.caption('Ponderosa Pine Forest Type')
    elif (species == 'Cottonwood/Willow Forest Type'):
        st.image('./Cottonwood-Willow.webp')
        st.caption( 'Cottonwood/Willow Forest Type')
    elif (species == 'Aspen Forest Type'):
        st.image('./Aspen.webp')
        st.caption( 'Aspen Forest Type')
    elif (species == 'Douglas-fir Forest Type'):
        st.image('./Douglas-fir.jpg')
        st.caption( 'Douglas-fir Forest Type')
    elif (species == 'Krummholz Forest Type'):
        st.image('./Krummholz.jpg')
        st.caption(  'Krummholz Forest Type')
    


if __name__ == '__main__':
     main()    
    
    
