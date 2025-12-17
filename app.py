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
st.set_page_config(page_title="Scanner Pro HD", layout="wide")
st.title("✂️ Escáner HD (Enfoque + Claridad)")

# Sidebar
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if not api_key:
    st.warning("Ingresa tu API Key para comenzar.")
    st.stop()

# ==========================================
# --- MEJORA INTELIGENTE (HD NATURAL) ---
# ==========================================

def mejora_inteligente(image_pil):
    # Convertir PIL a OpenCV
    img = np.array(image_pil)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Reducción de Ruido Conservadora
    # h=3 es muy bajo para no borrar texto fino, pero quita el "polvo" digital
    img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 3, 3, 7, 21)

    # 2. Mejora de Claridad (Solo en canal de Luz)
    # Convertimos a LAB para tocar solo la "Luminosidad" (L) y dejar los colores (A, B) quietos
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE suave: Aumenta contraste local para que las letras resalten del fondo
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl,a,b))
    img_cv = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 3. Enfoque "Unsharp Mask" (Técnica Fotográfica)
    # En lugar de un filtro agresivo, restamos una versión borrosa para resaltar solo bordes
    gaussian = cv2.GaussianBlur(img_cv, (0, 0), 3.0)
    # Fórmula: Original * 1.5 - Borrosa * 0.5 = Imagen con bordes definidos
    img_cv = cv2.addWeighted(img_cv, 1.5, gaussian, -0.5, 0)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

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
    image_pil.save(buffered, format="JPEG", quality=100)
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

st.info("1. Sube tu foto y ajusta el cuadro rojo para cubrir toda la tarjeta.")
uploaded_file = st.file_uploader("Subir imagen", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # --- ÁREA DE RECORTE (PANTALLA COMPLETA) ---
    st.write("### 2. Recorte Manual")
    
    # box_color='red', aspect_ratio=None (Libre)
    cropped_img = st_cropper(image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    
    st.write("---")
    
    if st.button("✅ PROCESAR IMAGEN", type="primary", use_container_width=True):
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with st.spinner('Aplicando enfoque y mejorando definición...'):
            # 1. Mejora Inteligente
            img_final = mejora_inteligente(cropped_img)
            
            with col_res1:
                st.subheader("Resultado HD")
                st.image(img_final, caption="Tarjeta Enfocada y Aclarada", use_container_width=True)
            
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
                    st.error("No se pudieron extraer datos.")

            # 3. PDF
            pdf_bytes = generar_pdf(img_final)
            st.success("¡Imagen mejorada con éxito!")
            st.download_button("⬇️ Descargar PDF", pdf_bytes, "tarjeta_hd.pdf", "application/pdf", use_container_width=True)
