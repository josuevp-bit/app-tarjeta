import streamlit as st
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
st.set_page_config(page_title="Lector Universal", layout="wide")
st.title("🚗 Escáner Inteligente (Auto-Detección de Modelos)")

# Sidebar
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if not api_key:
    st.warning("Ingresa tu API Key para comenzar.")
    st.stop()

# --- 1. FUNCIÓN MAESTRA: BUSCAR MODELO DISPONIBLE ---
def encontrar_modelo_activo(key):
    """Consulta a Google qué modelos tiene disponibles tu cuenta"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.sidebar.error(f"Error listando modelos: {response.status_code}")
            return None
            
        data = response.json()
        modelos = data.get('models', [])
        
        # Filtramos solo los que sirven para 'generateContent'
        candidatos = []
        for m in modelos:
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                nombre_limpio = m['name'].replace('models/', '')
                candidatos.append(nombre_limpio)
        
        # Lógica de Selección Inteligente
        # 1. Preferimos 'flash' (rápido y gratis)
        for m in candidatos:
            if 'flash' in m and 'legacy' not in m:
                return m
        
        # 2. Si no hay flash, buscamos 'pro' (pero no vision-legacy)
        for m in candidatos:
            if 'pro' in m and 'vision' not in m:
                return m

        # 3. Si no, devolvemos el experimental (gemini-2.0 o similar)
        if candidatos:
            return candidatos[0]
            
        return None

    except Exception as e:
        st.sidebar.error(f"Error de conexión al buscar modelos: {e}")
        return None

# --- 2. FUNCIONES DE IMAGEN ---
def mejorar_imagen(image_pil):
    img = np.array(image_pil)
    # Ajuste suave
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.2, gaussian, -0.2, 0)
    return Image.fromarray(img)

# --- 3. FUNCION OCR ---
def extraer_datos_http(image_pil, key):
    # PASO A: Buscar el modelo correcto
    modelo_elegido = encontrar_modelo_activo(key)
    
    if not modelo_elegido:
        st.error("❌ Tu API Key no tiene acceso a ningún modelo de generación de contenido. Verifica en Google AI Studio.")
        return None
        
    st.toast(f"🤖 Usando modelo: {modelo_elegido}")
    
    # PASO B: Preparar imagen
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt_text = """
    Analiza esta tarjeta de circulación. Extrae en JSON:
    'Propietario', 'Placa', 'Serie_VIN', 'Marca', 'Modelo', 'Año', 'Motor'.
    Si no se lee, pon "ILEGIBLE". Solo JSON.
    """
    
    data = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_str
                }}
            ]
        }]
    }
    headers = {'Content-Type': 'application/json'}

    # PASO C: Intentar conectar
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo_elegido}:generateContent?key={key}"
    
    try:
        # Pausa de seguridad para evitar error 429 (Cuota)
        time.sleep(2) 
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            result = response.json()
            try:
                texto = result['candidates'][0]['content']['parts'][0]['text']
                if "```json" in texto:
                    texto = texto.split("```json")[1].split("```")[0]
                elif "```" in texto:
                    texto = texto.split("```")[1].split("```")[0]
                return json.loads(texto)
            except:
                st.error("La IA respondió pero no pudimos leer el JSON.")
                return None
                
        elif response.status_code == 429:
            st.error("⏳ Cuota excedida (Error 429). Estás usando el plan gratuito.")
            st.info("Espera 1 minuto y vuelve a intentar. Google limita la velocidad en cuentas gratis.")
            return None
        else:
            st.error(f"Error del modelo {modelo_elegido} ({response.status_code}):")
            st.write(response.text)
            return None

    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

# --- PDF ---
def generar_pdf(image_pil):
    pdf = FPDF(orientation='P', unit='mm', format='Letter')
    pdf.add_page()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_pil.save(tmp.name)
        pdf.image(tmp.name, x=10, y=10, w=195, h=120) 
    pdf.set_font("Arial", size=12)
    pdf.text(10, 140, "Copia de Tarjeta de Circulación - Procesada")
    return pdf.output(dest='S').encode('latin1')

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Sube la Tarjeta", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    
    if st.button("Procesar Imagen"):
        with st.spinner('Consultando modelos disponibles y leyendo...'):
            img_mejorada = mejorar_imagen(image)
            with col2:
                st.image(img_mejorada, caption="Mejorada", use_container_width=True)
            
            datos = extraer_datos_http(image, api_key)
            
            if datos:
                df = pd.DataFrame([datos])
                def color_rojo(val):
                    return 'background-color: #ffcccc; color: red; font-weight: bold' if str(val).upper() == 'ILEGIBLE' else ''
                st.subheader("Datos Extraídos")
                st.dataframe(df.style.map(color_rojo))
            
            pdf_bytes = generar_pdf(img_mejorada)
            st.download_button("Descargar PDF", pdf_bytes, "tarjeta_lista.pdf", "application/pdf")
