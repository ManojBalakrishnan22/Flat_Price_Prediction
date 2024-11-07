import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('/home/dev105/Pictures/Projects/Sale/old/Resale_xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# CSS for styling
st.markdown(
    """
    <style>
    /* Main Page styling */
    .main {
                    background-color: #e0f7fa; 
                }

    /* Title Styling */
    .stTitle {
        color: #4CAF50;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        padding: 20px 0;
    }

    /* Button styleing */
    .stButton button {
        width: 300px;
        height: 60px;
        font-size: 20px;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }

    /* Input styleing */
    .stSelectbox, .stNumberInput {
        background-color: #ffffff;
        border-radius: 10px;
        border: 2px solid #ccc;
        padding: 10px;
        margin: 10px 0;
        font-size: 18px;
    }

    /* Write Styleing */
    .stWrite {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }

    </style>
    """, unsafe_allow_html=True
)

# encoders for categorical features
town_encoder = LabelEncoder()
flat_type_encoder = LabelEncoder()
storey_range_encoder = LabelEncoder()
flat_model_encoder = LabelEncoder()

# Cleaned data
df = pd.read_csv('/home/dev105/Pictures/Projects/df2.csv')

town_encoder.fit(df['town'].unique())
flat_type_encoder.fit(df['flat_type'].unique())
storey_range_encoder.fit(df['storey_range'].unique())
flat_model_encoder.fit(df['flat_model'].unique())

# Title page
st.title("🏠 Sales Price Prediction")

# Input fields
def user_input_features():
    col1, col2 = st.columns(2)

    with col1:
        month = st.selectbox('Month', options=df['month'].unique())
        floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=df['floor_area_sqm'].min(), max_value=df['floor_area_sqm'].max())
        town = st.selectbox('Town', options=df['town'].unique())

    with col2:
        flat_type = st.selectbox('Flat Type', options=df['flat_type'].unique())
        storey_range = st.selectbox('Storey Range', options=df['storey_range'].unique())
        flat_model = st.selectbox('Flat Model', options=df['flat_model'].unique())

    # Encode the categorical features
    town_encoded = town_encoder.transform([town])[0]
    flat_type_encoded = flat_type_encoder.transform([flat_type])[0]
    storey_range_encoded = storey_range_encoder.transform([storey_range])[0]
    flat_model_encoded = flat_model_encoder.transform([flat_model])[0]

    data = {
        'month': month,
        'floor_area_sqm': floor_area_sqm,
        'town_encoded': town_encoded,
        'flat_type_encoded': flat_type_encoded,
        'storey_range_encoded': storey_range_encoded,
        'flat_model_encoded': flat_model_encoded
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# Prediction Part
if st.button('🔮 Predict'):
    prediction = model.predict(df_input)
    st.write(f"💡 **Prediction**: ${prediction[0]:,.2f}")
    st.balloons()
