import streamlit as st
from streamlit_option_menu import option_menu
from joblib import load
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(layout="wide")

st.title(":rainbow[Industrical Copper Modeling]")
st.write("This project is about Machine Learning model that could preduct Price & Status of the product")
tab1,tab2,tab3 = st.tabs(["HOME",'**Get Price**','**Get Status**'])

with tab1:
    st.subheader(":green[Get the predicted Selling price of Copper]")
    st.markdown("**Enter the following parameters to Predict both price and status**")

    
    item_type_options = {'W':5, 'WI':3, 'S':1, 'Others':2, 'PL':6, 'IPL':0, 'SLAWR':4}
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40.,
                            25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                    '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                    '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                    '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                    '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
    
    #with st.form("my_form"):
    col1, col2, col3 = st.columns([5, 2, 5])
    with col1:
                st.write(' ')
                
                selected_item= st.selectbox("Item Type", item_type_options.items(), key=2)
                item_type=selected_item[1]
                country = st.selectbox("Country", sorted(country_options), key=3)
                application = st.selectbox("Application", sorted(application_options), key=4)
                product_ref = st.selectbox("Product Reference", product, key=5)
                thickness = st.number_input("thickness")
                quantity = st.number_input("quantity tons")
                width = st.number_input("width")
                item_date = st.date_input("item date",value=None)
                delivery_date = st.date_input("Delivery date",value=None)
                selling_price = st.number_input("Selling Price")

                data={
                        "country"       : country,
                        "item"          : item_type,
                        "application"   : application,
                        "width"         : width,
                        "product_ref"   : product_ref,
                        "quantity"      : quantity,
                        "selling_price" : selling_price,
                        "thickness"     : thickness,
                        "item_date"     : item_date,
                        "delivery_date" : delivery_date                 
                        
                        }
                df = pd.DataFrame([data])
                df["item_date"] = pd.to_datetime(df["item_date"],format='%Y%m%d')
                df["delivery_date"] = pd.to_datetime(df["delivery_date"],format='%Y%m%d')
                df["quantity_tons_log"] = np.log(df["quantity"])
                df['selling_price_log'] = np.log(df["selling_price"])
                df['thickness_log'] = np.log(df['thickness'])
                df["delivery_days"] = (df["delivery_date"] - df["item_date"]).dt.days
                df['item_date_day'] = df['item_date'].dt.day
                df['item_date_month'] = df['item_date'].dt.month
                df['item_date_year'] = df['item_date'].dt.year
                df['delivery_date_day'] = df['delivery_date'].dt.day
                df['delivery_date_month'] = df['delivery_date'].dt.month
                df['delivery_date_year'] = df['delivery_date'].dt.year


                df1=df.copy()
                reg=df1.drop(["item_date","delivery_date","quantity",'thickness','selling_price_log',"selling_price"],axis=1)
                cl =df1.drop(["item_date","delivery_date","quantity",'thickness','selling_price'],axis=1)

with tab2:
                
                if st.button("REg"):
                        st.dataframe(reg)
                model=joblib.load(r'D:\projectyoutube\copper\copper_rf_model.pkl')
                scaler_reg = joblib.load(r"D:\projectyoutube\copper\copper_scaler_model.pkl")
                scaled_data = scaler_reg.transform(reg)
                if st.button("Predict Selling Price"):
                        predictions = model.predict(scaled_data)
                        st.write(f"Predicted Selling Price:$ {np.around(np.exp(predictions[0]),decimals=2)}")
with tab3:
                   if st.button('Class'):
                        st.dataframe(cl)
                   model=joblib.load(r'D:\projectyoutube\copper\copper_cl_model.pkl')
                   scaler_class = joblib.load(r"D:\projectyoutube\copper\copper_cl_scaler.pkl")
                   scaled_cl_data = scaler_class.transform(cl)
                   if st.button("Predict Status"):
                        prediction = model.predict(scaled_cl_data)
                        st.write(f"Predicted Status:'{prediction}")