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
st.set_page_config(page_title="Lector Tarjetas", layout="wide")
st.title("🚗 Escáner de Tarjetas (Modo Gratuito - Flash)")

# Sidebar
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if not api_key:
    st.warning("Ingresa tu API Key para comenzar.")
    st.stop()

# --- FUNCIONES DE IMAGEN ---
def mejorar_imagen(image_pil):
    img = np.array(image_pil)
    # Ajuste suave
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.2, gaussian, -0.2, 0)
    return Image.fromarray(img)

# --- FUNCION OCR (SOLO MODELOS GRATUITOS) ---
def extraer_datos_http(image_pil, key):
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # AQUÍ ESTÁ EL CAMBIO: Solo usamos "flash". 
    # Quitamos "pro" y "latest" para evitar el error 429.
    modelos_gratuitos = [
        "gemini-1.5-flash",         # El estándar gratuito
        "gemini-1.5-flash-latest",  # Su variante más nueva
        "gemini-1.5-flash-001",     # La versión específica
        "gemini-1.5-flash-8b"       # Versión ultra ligera (por si las otras fallan)
    ]

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

    errores = []

    for modelo in modelos_gratuitos:
        # Pausa de seguridad de 1 segundo para no saturar si reintentamos
        time.sleep(1) 
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={key}"
        
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
                    
                    st.toast(f"✅ Éxito usando modelo: {modelo}")
                    return json.loads(texto)
                except:
                    continue
            
            # Si es error 429 (Resource Exhausted), es crucial esperar un poco y seguir
            elif response.status_code == 429:
                errores.append(f"{modelo}: Cuota excedida (429)")
                time.sleep(2) # Esperar 2 segundos extra antes del siguiente modelo
                continue
            else:
                errores.append(f"{modelo}: Error {response.status_code}")
                continue

        except Exception as e:
            errores.append(f"{modelo}: {str(e)}")
            continue

    st.error("❌ No se pudo procesar. Razones:")
    st.write(errores)
    st.info("💡 Si ves 'Cuota excedida' en todos, espera 1 minuto y vuelve a intentar. El plan gratuito tiene límites por minuto.")
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
        with st.spinner('Procesando con modelo gratuito...'):
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
