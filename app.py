import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from google import genai

st.set_page_config(page_title="AI Deteksi Pneumonia", page_icon="🫁", layout="centered")
st.title("🫁 Sistem Skrining Pneumonia & Laporan Medis Otomatis")
st.write("Unggah citra X-Ray dada. AI akan mendeteksi Pneumonia dan membuat draf laporan medis.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_pneumonia.h5')

model = load_model()

# GANTI DENGAN API KEY KAMU
client = genai.Client(api_key="AQ.Ab8RN6KVU_KjMcP2eILrEf0A0elpj5_sojfJagpFA-Xz5ldB9g") 

uploaded_file = st.file_uploader("Pilih gambar X-Ray (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)
        
    with col2:
        st.write("### Hasil Analisis AI:")
        if st.button("Mulai Analisis 🔍"):
            with st.spinner("AI sedang menganalisis gambar..."):
                # Preprocessing
                img = Image.open(uploaded_file).convert('L')
                img = img.resize((150, 150))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                img_array = np.expand_dims(img_array, axis=-1)
                
                # Prediksi
                prediction = model.predict(img_array)
                prob = prediction[0][0]
                
                if prob > 0.5:
                    hasil = "NORMAL"
                    warna = "green"
                    prob_persen = prob * 100
                else:
                    hasil = "PNEUMONIA"
                    warna = "red"
                    prob_persen = (1 - prob) * 100
                
                st.markdown(f"<h3 style='color: {warna};'>Terdeteksi: {hasil}</h3>", unsafe_allow_html=True)
                st.write(f"Tingkat Keyakinan: **{prob_persen:.2f}%**")
                
                st.write("---")
                st.write("### 📝 Draf Laporan Medis Otomatis:")
                prompt = f"Model AI mendeteksi X-Ray dengan hasil: {hasil} ({prob_persen:.2f}%). Buatkan draf laporan medis singkat dan saran tindakan awal untuk perawat di puskesmas."
                
                response = client.models.generate_content(
                    model="gemini-3.8-flash", 
                    contents=prompt
                )
                st.info(response.text)