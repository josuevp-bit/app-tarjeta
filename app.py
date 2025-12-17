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
st.set_page_config(page_title="Lector de Tarjetas", layout="wide")
st.title("🚗 Escáner de Tarjetas (Versión Actualizada)")

# Sidebar
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if not api_key:
    st.warning("Ingresa tu API Key en la barra lateral para comenzar.")
    st.stop()

# --- FUNCIONES DE IMAGEN (NATURAL) ---
def mejorar_imagen(image_pil):
    # Convertir a array numpy
    img = np.array(image_pil)
    
    # 1. Ajuste Básico de Brillo y Contraste
    # Aumentamos ligeramente el contraste (1.1) y el brillo (10)
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)

    # 2. Enfoque MUY sutil para definir letras
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.2, gaussian, -0.2, 0)
    
    return Image.fromarray(img)

# --- FUNCION OCR (CON REPORTE DE ERRORES REAL) ---
def extraer_datos_http(image_pil, key):
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # SOLO usamos modelos que existen HOY (Diciembre 2025)
    modelos_posibles = [
        "gemini-1.5-flash",    # El más rápido y estable para texto
        "gemini-1.5-pro",      # El más potente (backup)
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

    # Variable para guardar TODOS los errores y mostrártelos
    errores_acumulados = []

    for modelo in modelos_posibles:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={key}"
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            
            if response.status_code == 200:
                result = response.json()
                try:
                    # Intentar leer la respuesta
                    texto = result['candidates'][0]['content']['parts'][0]['text']
                    
                    # Limpiar JSON (quitar ```json y ```)
                    if "```json" in texto:
                        texto = texto.split("```json")[1].split("```")[0]
                    elif "```" in texto:
                        texto = texto.split("```")[1].split("```")[0]
                    
                    return json.loads(texto) # ¡ÉXITO!
                except Exception as e:
                    errores_acumulados.append(f"Modelo {modelo} respondió pero falló al leer el JSON: {str(e)}")
                    continue
            else:
                # Si Google da error (ej. 400, 403, 404, 429)
                error_info = response.json()
                mensaje = error_info.get('error', {}).get('message', response.text)
                errores_acumulados.append(f"Modelo {modelo} dió error {response.status_code}: {mensaje}")
                continue

        except Exception as e:
            errores_acumulados.append(f"Fallo de conexión con {modelo}: {str(e)}")
            continue

    # Si llegamos aquí, NINGUNO funcionó. Mostramos el reporte completo.
    st.error("❌ No se pudo procesar la imagen.")
    with st.expander("Ver detalles técnicos del error (Importante)"):
        for error in errores_acumulados:
            st.write(f"- {error}")
            
    st.info("💡 Solución probable: Si el error dice 'API key not valid', revisa tu clave. Si dice 'User location is not supported', intenta usar una VPN o revisa la configuración de tu cuenta.")
    return None

# --- PDF ---
def generar_pdf(image_pil):
    pdf = FPDF(orientation='P', unit='mm', format='Letter')
    pdf.add_page()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_pil.save(tmp.name)
        # Ajuste para media carta
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
        with st.spinner('Procesando...'):
            # 1. Mejorar imagen
            img_mejorada = mejorar_imagen(image)
            with col2:
                st.image(img_mejorada, caption="Mejorada (Natural)", use_container_width=True)
            
            # 2. Extraer datos
            datos = extraer_datos_http(image, api_key)
            
            if datos:
                df = pd.DataFrame([datos])
                def color_rojo(val):
                    return 'background-color: #ffcccc; color: red; font-weight: bold' if str(val).upper() == 'ILEGIBLE' else ''
                st.subheader("Datos Extraídos")
                st.dataframe(df.style.map(color_rojo))
            
            # 3. PDF
            pdf_bytes = generar_pdf(img_mejorada)
            st.download_button("Descargar PDF", pdf_bytes, "tarjeta_lista.pdf", "application/pdf")
