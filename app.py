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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Lector Inteligente", layout="wide")
st.title("🚗 Escáner de Tarjetas (Auto-Detección)")

# Sidebar
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if not api_key:
    st.warning("Ingresa tu API Key en la barra lateral para comenzar.")
    st.stop()

# --- 1. FUNCIÓN PARA DESCUBRIR MODELOS DISPONIBLES ---
def obtener_mejor_modelo(key):
    """Pregunta a Google qué modelos tiene habilitada esta llave"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Filtramos solo los que sirven para generar contenido
            modelos_disponibles = []
            for m in data.get('models', []):
                if 'generateContent' in m.get('supportedGenerationMethods', []):
                    # Guardamos el nombre limpio (sin 'models/')
                    nombre = m['name'].replace('models/', '')
                    modelos_disponibles.append(nombre)
            
            # Lógica de preferencia: Buscamos el mejor disponible
            prioridades = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro', 'gemini-pro', 'gemini-pro-vision']
            
            for p in prioridades:
                for m in modelos_disponibles:
                    if p in m: # Si encontramos uno de nuestros preferidos
                        return m
            
            # Si no hay ninguno de los conocidos, devolvemos el primero que haya
            if modelos_disponibles:
                return modelos_disponibles[0]
                
    except Exception as e:
        print(f"Error listando modelos: {e}")
    
    # Si todo falla, regresamos el estándar por defecto
    return 'gemini-1.5-flash'

# --- 2. FUNCIONES DE IMAGEN ---
def mejorar_imagen(image_pil):
    img = np.array(image_pil)
    # Ajuste suave de luz y contraste
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    # Enfoque suave
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.2, gaussian, -0.2, 0)
    return Image.fromarray(img)

# --- 3. FUNCION OCR ---
def extraer_datos_http(image_pil, key):
    # PASO A: Encontrar el modelo correcto para TU cuenta
    modelo_a_usar = obtener_mejor_modelo(key)
    st.toast(f"Usando modelo IA: {modelo_a_usar}") # Aviso en pantalla

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

    # PASO C: Enviar petición
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo_a_usar}:generateContent?key={key}"
    
    try:
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
                st.error("La IA respondió pero no en formato JSON. Intenta otra foto.")
                return None
        else:
            # Si falla, mostramos el error exacto y la lista de modelos que intentamos leer
            st.error(f"Error de Google ({response.status_code}):")
            st.code(response.text)
            st.warning(f"Intentamos usar el modelo '{modelo_a_usar}' pero falló.")
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
uploaded_file = st.file_uploader("Sube la Tarjeta (Foto)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    
    if st.button("Procesar Imagen"):
        with st.spinner('Detectando modelo de IA y leyendo...'):
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
