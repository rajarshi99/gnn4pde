import pandas as pd
import streamlit as st
import os
from PIL import Image

def load_data(filename):
    return pd.read_csv(filename, skipinitialspace=True)

def get_image_files(root_folder):
    image_files = []
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(folder_path, file))
    return image_files

st.title("Monitor Runs")

filename = "feb4.csv"

df = load_data(filename)
st.dataframe(df)

col1, col2 = st.columns(2)

with col1:
    root_folder = st.selectbox(df['path'])
    image_files = get_image_files(root_folder)
    image1 = st.selectbox("Select first image", image_files, key="image1")
    if image1:
        st.image(Image.open(image1), use_column_width=True)

with col2:
    root_folder = st.selectbox(df['path'])
    image2 = st.selectbox("Select second image", image_files, key="image2")
    if image2:
        st.image(Image.open(image2), use_column_width=True)

# import streamlit as st
# import pandas as pd
# from pathlib import Path
# from PIL import Image
# 
# def load_data(filename):
#     return pd.read_csv(filename, skipinitialspace=True)
# 
# def get_images_in_folder(folder_name):
#     folder_path = Path(folder_name)
#     # st.image(image_path, caption='Description of the image', width=300)
#     image_files = [f for f in folder_path.iterdir() ] #if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
#     return image_files
# 
# st.title("Monitor Runs")
# 
# if st.button('Image'):
#     image_files = get_images_in_folder(selected_path)
#     selected_image = st.selectbox('Select an image:', image_files)
# 
#     image_list.append(Image.open(selected_image))
# 
# filename = "feb4.csv"
# 
# df = load_data(filename)
# st.dataframe(df)
# 
# image_list = []
# 
# selected_path = st.selectbox('Select a folder:', df['path'])
# 
# if st.button('Display'):
#     st.write(len(image_list))
#     cols = st.columns(len(image_list))
#     for col,img in zip(cols, image_list):
#         with col:
#             st.image(img)
# 
# 
