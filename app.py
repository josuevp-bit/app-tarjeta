import streamlit as st
from streamlit_cropper import st_cropper
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
st.set_page_config(page_title="Scanner Manual Pro", layout="wide")
st.title("✂️ Escáner de Alta Precisión")

# Sidebar
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if not api_key:
    st.warning("Ingresa tu API Key para comenzar.")
    st.stop()

# ==========================================
# --- MEJORA DE IMAGEN (SOLO LUZ, SIN BORRAR DETALLES) ---
# ==========================================

def mejora_natural(image_pil):
    # Convertimos a OpenCV
    img = np.array(image_pil)
    
    # Solo ajustamos un poco el brillo y contraste para que no se vea oscura
    # No usamos filtros de desenfoque ni "pintura al oleo"
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10) # alpha=1.1 (poquito contraste), beta=10 (poquito brillo)
    
    return Image.fromarray(img)

# ==========================================
# --- CONEXIÓN GOOGLE (OCR) ---
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
    image_pil.save(buffered, format="JPEG", quality=100) # Calidad máxima
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

st.info("1. Sube tu foto y recorta manualmente la tarjeta.")
uploaded_file = st.file_uploader("Subir imagen", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # --- ÁREA DE RECORTE (PANTALLA COMPLETA) ---
    st.write("### 2. Ajusta el recuadro rojo")
    st.caption("Usa las esquinas para seleccionar la tarjeta. Ahora tienes más espacio.")
    
    # EL CAMBIO CLAVE: Quitamos las columnas.
    # box_color='red' es el color del cuadro.
    # aspect_ratio=None permite cualquier forma (no fuerza cuadrado).
    cropped_img = st_cropper(image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    
    st.write("---")
    
    # Botón grande
    if st.button("✅ RECORTAR Y EXTRAER DATOS", type="primary", use_container_width=True):
        
        # --- RESULTADOS (AQUÍ SÍ USAMOS COLUMNAS) ---
        col_res1, col_res2 = st.columns([1, 1])
        
        with st.spinner('Procesando imagen en alta calidad...'):
            # 1. Mejora sutil (respetando calidad original)
            img_final = mejora_natural(cropped_img)
            
            with col_res1:
                st.subheader("Imagen Final")
                st.image(img_final, caption="Tarjeta Lista", use_container_width=True)
            
            # 2. Leer datos
            datos = extraer_datos_http(img_final, api_key)
            
            with col_res2:
                st.subheader("Datos Detectados")
                if datos:
                    df = pd.DataFrame([datos])
                    def color_rojo(val):
                        return 'background-color: #ffcccc; color: red; font-weight: bold' if str(val).upper() == 'ILEGIBLE' else ''
                    st.dataframe(df.style.map(color_rojo))
                else:
                    st.error("No se pudieron extraer datos. Verifica que el recorte incluya el texto.")

            # 3. PDF
            pdf_bytes = generar_pdf(img_final)
            st.success("¡Proceso terminado!")
            st.download_button("⬇️ Descargar PDF Listo para Imprimir", pdf_bytes, "tarjeta_imprimir.pdf", "application/pdf", use_container_width=True)
