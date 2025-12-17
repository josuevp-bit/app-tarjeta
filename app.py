import streamlit as st
from streamlit_cropper import st_cropper # LIBRERÍA NUEVA
import requests
import base64
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from fpdf import FPDF
import io
import tempfile
import json
import time

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Scanner Manual", layout="wide")
st.title("✂️ Escáner Manual (Estilo CamScanner)")

# Sidebar
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if not api_key:
    st.warning("Ingresa tu API Key para comenzar.")
    st.stop()

# ==========================================
# --- FUNCIONES DE MEJORA (POST-RECORTE) ---
# ==========================================

def mejora_final_color(image_pil):
    # Convertir a OpenCV
    img = np.array(image_pil)
    
    # 1. Aumentar nitidez y contraste (Ya que el usuario recortó bien)
    # Convertir a LAB para mejorar solo la luminosidad
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl,a,b))
    img_final = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    
    # Un poco de enfoque suave
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img_final = cv2.filter2D(src=img_final, ddepth=-1, kernel=kernel)
    
    return Image.fromarray(img_final)

# ==========================================
# --- FUNCIONES DE CONEXIÓN A GOOGLE ---
# ==========================================

def encontrar_modelo_activo(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code != 200: return 'gemini-1.5-flash'
        data = response.json()
        candidatos = [m['name'].replace('models/', '') for m in data.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
        for m in candidatos: 
            if 'flash' in m and 'legacy' not in m: return m
        if candidatos: return candidatos[0]
        return 'gemini-1.5-flash'
    except:
        return 'gemini-1.5-flash'

def extraer_datos_http(image_pil, key):
    modelo_elegido = encontrar_modelo_activo(key)
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt_text = """
    Analiza esta tarjeta de circulación. Extrae datos en JSON:
    'Propietario', 'Placa', 'Serie_VIN', 'Marca', 'Modelo', 'Año', 'Motor'.
    Si no es visible, pon "ILEGIBLE".
    """
    
    data = {"contents": [{"parts": [{"text": prompt_text}, {"inline_data": {"mime_type": "image/jpeg", "data": img_str}}]}]}
    headers = {'Content-Type': 'application/json'}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo_elegido}:generateContent?key={key}"
    
    try:
        time.sleep(1)
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            try:
                texto = result['candidates'][0]['content']['parts'][0]['text']
                if "```json" in texto: texto = texto.split("```json")[1].split("```")[0]
                elif "```" in texto: texto = texto.split("```")[1].split("```")[0]
                return json.loads(texto)
            except: return None
        return None
    except: return None

def generar_pdf(image_pil):
    pdf = FPDF(orientation='P', unit='mm', format='Letter')
    pdf.add_page()
    
    pdf_w_max = 190
    pdf_h_max = 130
    
    img_w, img_h = image_pil.size
    ratio = pdf_w_max / img_w
    final_w = pdf_w_max
    final_h = img_h * ratio
    
    if final_h > pdf_h_max:
        ratio = pdf_h_max / img_h
        final_h = pdf_h_max
        final_w = img_w * ratio

    x_pos = (216 - final_w) / 2
    y_pos = 15 

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_pil.save(tmp.name, quality=100)
        pdf.image(tmp.name, x=x_pos, y=y_pos, w=final_w, h=final_h) 
        
    return pdf.output(dest='S').encode('latin1')

# --- INTERFAZ ---
uploaded_file = st.file_uploader("1. Sube la foto original", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # --- COLUMNA DE RECORTE (IZQUIERDA) ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("2. Ajusta el recuadro rojo")
        st.info("Mueve las esquinas del cuadro rojo para seleccionar SOLO la tarjeta.")
        
        # WIDGET DE RECORTE (streamlit-cropper)
        # realtime_update=True hace que veas el resultado al instante
        cropped_img = st_cropper(image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
        
        st.caption("La imagen recortada se ve a la derecha ->")

    # --- COLUMNA DE RESULTADO (DERECHA) ---
    with col2:
        st.subheader("3. Vista Previa")
        # Mostrar lo que el usuario está recortando
        st.image(cropped_img, caption="Así quedará tu tarjeta", use_container_width=True)
        
        st.write("---")
        # Botón maestro
        if st.button("✅ Confirmar Recorte y Extraer Datos", type="primary"):
            
            with st.spinner('Mejorando calidad y leyendo textos...'):
                # 1. Aplicar filtros de mejora a la imagen YA recortada por el humano
                img_final = mejora_final_color(cropped_img)
                
                # Mostrar versión HD
                st.image(img_final, caption="Imagen Mejorada (HD)", use_container_width=True)
                
                # 2. Leer datos
                datos = extraer_datos_http(img_final, api_key)
                
                if datos:
                    df = pd.DataFrame([datos])
                    def color_rojo(val):
                        return 'background-color: #ffcccc; color: red; font-weight: bold' if str(val).upper() == 'ILEGIBLE' else ''
                    st.success("¡Lectura Completada!")
                    st.dataframe(df.style.map(color_rojo))
                else:
                    st.error("No se pudieron extraer datos. Revisa que el recorte no corte letras.")

                # 3. PDF
                pdf_bytes = generar_pdf(img_final)
                st.download_button("Descargar PDF Listo", pdf_bytes, "tarjeta_imprimir.pdf", "application/pdf")
