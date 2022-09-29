from isort import stream
from matplotlib import image
import numpy as np
import pickle 
import base64
import streamlit as st

load_model=pickle.load(open('D:\chennai house price\gredient_model.sav', 'rb'))
#creating function ftomlor prediction

def chennai_house_pred(input_data):
    input_data_as_numpy_array=np.asanyarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=load_model.predict(input_data_reshaped)
    return prediction

def main():
    st.title("CHENNAI HOUSE PRICE PEDICTION")
    AREA=st.selectbox("SELECT AREA",['KARAPAKKAM','ANNA NAGAR', 'ADYAR', 'CHROMPET', 'KK NAGAR', 'T NAGAR','VELACHERY'])
    AREA1={'KARAPAKKAM': 0, 'ANNA NAGAR': 5, 'ADYAR': 1, 'CHROMPET': 2, 'KK NAGAR': 4, 'T NAGAR': 6, 'VELACHERY': 3}
    AREA_SQFT=st.number_input("ENTER AREA IN SQFT")
    c1,c2=st.columns(2)
    BEDROOM=c1.radio(" BEDROOM",[1,2,3,4])
    BATHROOM=c2.radio(" BATHROOM",[1,2])
    ROOM=st.slider("ROOM",min_value=1,max_value=6,step=1)
    PARKING_FACILITY=st.radio("PARKING FACILTY AVAILABILITY",["YES","NO"])
    SALE_COUND=st.selectbox("SALE CONDITION",['PARTIAL','FAMILY','ABNORMAL','NORMAL SALE','ADJUST LAND'])
    UTALITY_AVAIL=st.selectbox("UTILITY AVAILABILITY",['AllPub', 'ELO', 'NoSeWa'])
    MZZONE=st.selectbox("MZZONE",['A', 'C', 'I', 'RH', 'RL', 'RM'])
    STREET=st.selectbox("STREET",['Gravel', 'No Access', 'Paved'])
    BULDTYPE=st.selectbox("BUILDING TYPE",['Commercial', 'HOUSE', 'OTHERS'])
    AGE=st.slider("AGE OF THE HOUSE",min_value=1,max_value=100,step=1)
    SALE_COUND1={"PARTIAL":0, "FAMILY":1,"ABNORMAL":2, "NORMAL SALE":3, "ADJUST LAND":4}
    PARKING_FACILITY1={"YES":1,"NO":0}
    UTALITY_AVAIL1={'AllPub': 2, 'ELO': 0, 'NoSeWa': 1}
    MZZONE_1={'A': 0, 'C': 1, 'I': 2, 'RH': 3, 'RL': 4, 'RM': 5}
    STREET1={'Gravel':2, 'No Access':0, 'Paved':1}
    BULDTYPE1={'Commercial':2, 'House':0, 'Others':1}
    
    prediction=""
    if st.button(" prediction result>>>"):
        prediction=chennai_house_pred([AREA1.get(AREA),AREA_SQFT, BEDROOM, BATHROOM,SALE_COUND1.get(SALE_COUND), 
        ROOM, PARKING_FACILITY1.get(PARKING_FACILITY), UTALITY_AVAIL1.get(UTALITY_AVAIL), MZZONE_1.get(MZZONE), STREET1.get(STREET),
        1 if BULDTYPE1.get(BULDTYPE)==0 else 0,
        1 if BULDTYPE1.get(BULDTYPE)==2 else 0,
        1 if BULDTYPE1.get(BULDTYPE)==1 else 0, AGE])

    st.success(prediction)

if __name__=='__main__':
    main()
    