import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Beranda")
app_mode = st.sidebar.selectbox("Pilih Halaman",["Beranda","Tentang","Pengenalan Penyakit"])

#Main Page
if(app_mode=="Beranda"):
    st.header("Sistem Pengenalan Penyakit Tanaman")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Selamat datang di Sistem Pengenalan Penyakit Tanaman! 🌿🔍

    Misi kami adalah membantu mengidentifikasi penyakit tanaman secara efisien. Unggah gambar tanaman, dan sistem kami akan menganalisanya untuk mendeteksi tanda-tanda penyakit. Mari bersama-sama melindungi hasil panen kita dan memastikan panen yang lebih sehat!

    ### Bagaimana Cara Kerjanya
    - **Unggah Gambar:** Buka halaman Pengenalan Penyakit dan unggah gambar tanaman yang diduga terkena penyakit.
    - **Analisis:** Sistem kami akan memproses gambar menggunakan algoritma canggih untuk mengidentifikasi kemungkinan penyakit.
    - **Hasil:** Lihat hasil dan rekomendasi untuk tindakan selanjutnya.

    ### Mengapa Memilih Kami?
    - **Akurasi:** Sistem kami menggunakan teknik machine learning mutakhir untuk deteksi penyakit yang akurat.
    - **User-Friendly:** Antarmuka yang sederhana dan intuitif untuk pengalaman pengguna yang lancar.
    - **Cepat dan Efisien:** Dapatkan hasil dalam hitungan detik, memungkinkan pengambilan keputusan yang cepat.

    ### Mulai Sekarang
    Klik Halaman **Pengenalan Penyakit** di sidebar untuk mengunggah gambar dan merasakan kekuatan Sistem Pengenalan Penyakit Tanaman kami!

    ### Tentang Kami
    Pelajari lebih lanjut tentang proyek ini, tim kami, dan tujuan kami di halaman  **Tentang Kami**.
    """)

#About Project
elif(app_mode=="Tentang"):
    st.header("Tentang")
    st.markdown("""
                #### Tentang Dataset
                Dataset ini dibuat ulang menggunakan augmentasi offline dari dataset asli. Dataset asli dapat ditemukan di repositori GitHub ini.
                Dataset ini terdiri dari sekitar 87 ribu gambar RGB daun tanaman sehat dan terkena penyakit yang dikategorikan ke dalam 38 kelas yang berbeda. Total dataset ini dibagi menjadi rasio 80/20 antara set pelatihan dan validasi, dengan mempertahankan struktur direktori.
                Sebuah direktori baru yang berisi 33 gambar uji dibuat kemudian untuk tujuan prediksi.
                #### Konten
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Pengenalan Penyakit"):
    st.header("Pengenalan Penyakit")
    test_image = st.file_uploader("Pilih Gambar:")
    if(st.button("Tampilkan Gambar")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Prediksi")):
        st.snow()
        st.write("Prediksi Kami")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 
                      'Apple___Black_rot',
                       'Apple___Cedar_apple_rust', 
                       'Apple___healthy',
                    'Blueberry___healthy', 
                    'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 
                    'Corn_(maize)___Northern_Leaf_Blight', 
                    'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 
                    'Grape___Esca_(Black_Measles)', 
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 
                    'Orange___Haunglongbing_(Citrus_greening)', 
                    'Peach___Bacterial_spot',
                    'Peach___healthy', 
                    'Pepper,_bell___Bacterial_spot', 
                    'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 
                    'Potato___Late_blight', 
                    'Potato___healthy', 
                    'Raspberry___healthy', 
                    'Soybean___healthy', 
                    'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 
                    'Strawberry___healthy', 
                    'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 
                    'Tomato___Late_blight', 
                    'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 
                    'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                    'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
