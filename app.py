import numpy as np
import pickle
import streamlit as st
import requests
from streamlit_lottie import st_lottie


st.set_page_config(page_title="Crop Recommender System", layout="wide")

# --- Important Functions ---
def load_url(url):
  r = requests.get(url)  # to access the animation link
  if r.status_code != 200:
    return None
  return r.json()

# Animation Assests
plant = load_url("https://assets9.lottiefiles.com/packages/lf20_xd9ypluc.json")
farmer = load_url("https://assets7.lottiefiles.com/packages/lf20_sgn7zslb.json")


# --- Load The Model ---
recommender = pickle.load(open('CropRecommender.sav', 'rb'))


def crs_output(input_data):
  input_array = np.array(input_data)
  final_input = input_array.reshape(1, -1)
  prediction = recommender.predict(final_input)
  output = prediction[0]
  return (f'Field conditions are most suitable for {output}')


def main():

  with st.container():
    c1, c2 = st.columns((1, 1.5))
    with c1:
      st.title("Crop Recommender System")
      st.write("Data Source: Crop Recommendation Dataset Kaggle")
      st.write("This dataset contains 2200 datapoints")
      st.write('Model Accuracy: 98%')
      st.write("This Project was created for Sandip Foundation MiniProject")
      st.write("")
      st.write("")
      st.write('---Scroll down to test the model---')

    with c2:
      st_lottie(farmer, height=650, key="farmer")

  with st.container():
    col1, col2 = st.columns((1, 1))
    with col1:

      st.header("Enter your crop details:")
      # -- Time to take user input --
      # our model takes 7 parameters so we need 7 input fields

      n = st.number_input("Nitrogen Level of Your field", min_value=0.0, max_value=100.0)
      p = st.number_input("Phosphorus Level of Your field", min_value=0.0, max_value=100.0)
      k = st.number_input("Potassium Level of Your field", min_value=0.0, max_value=100.0)
      temperature = st.number_input("Average temperature(°C) in your area", min_value=0.0, max_value=100.0)
      humidity = st.number_input("Relative humidity in %", min_value=0.0, max_value=100.0)
      ph_value = st.number_input("Ph value of the soil", min_value=0.0, max_value=15.0)
      rainfall = st.number_input("Average rainfall(in mm) in your area", min_value=0.0)

      # User Input for Crop Name
      user_crop = st.text_input("Enter a Crop Name (e.g., Wheat, Corn)")

      # --- Code for recommendation ---
      crop_output = ""
      # --- creating a button ---
      if st.button("Find The Crop"):
        crop_output = crs_output([n, p, k, temperature, humidity, ph_value, rainfall])
        with col2:
          st_lottie(plant, height=800, key="plant")

        # Favourability Message
        if user_crop.lower() == crop_output.lower():
          st.success(f"{user_crop} is a favourable crop for your field conditions!")
        else:
          st.warning(f"{user_crop} might not be the most favourable crop. Our model recommends {crop_output} based on your input.")

  st.success(crop_output)  # This line can be removed, redundant with button press


if __name__ == '__main__':
  main()
