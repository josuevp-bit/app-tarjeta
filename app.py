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
st.title("🚗 Escáner de Tarjetas (Modo Directo)")

api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if not api_key:
    st.warning("Ingresa tu API Key para comenzar.")
    st.stop()

# --- FUNCIONES DE IMAGEN ---
def mejorar_imagen(image_pil):
    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# --- FUNCION OCR (CONEXIÓN DIRECTA MULTI-INTENTO) ---
def extraer_datos_http(image_pil, key):
    # 1. Convertir imagen a Base64
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Lista de modelos para probar (si falla uno, sigue al otro)
    modelos_posibles = [
        "gemini-1.5-flash-latest", # Opción 1: La más nueva
        "gemini-1.5-flash-001",    # Opción 2: La versión estable específica
        "gemini-1.5-flash",        # Opción 3: La estándar
        "gemini-1.5-pro"           # Opción 4: La versión Pro (más potente)
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

    # Bucle de intentos
    for modelo in modelos_posibles:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={key}"
        
        try:
            # print(f"Probando modelo: {modelo}...") # (Opcional para depurar)
            response = requests.post(url, headers=headers, data=json.dumps(data))
            
            if response.status_code == 200:
                # ¡ÉXITO!
                result = response.json()
                try:
                    texto_respuesta = result['candidates'][0]['content']['parts'][0]['text']
                    # Limpiar JSON
                    if "```json" in texto_respuesta:
                        texto_respuesta = texto_respuesta.split("```json")[1].split("```")[0]
                    elif "```" in texto_respuesta:
                        texto_respuesta = texto_respuesta.split("```")[1].split("```")[0]
                    return json.loads(texto_respuesta)
                except:
                    continue # Si la respuesta está rota, probamos el siguiente
            else:
                # Si error es 404 (modelo no encontrado), seguimos al siguiente
                continue

        except Exception as e:
            continue

    # Si llegamos aquí, fallaron los 4 modelos
    st.error("Error: Se probaron 4 versiones del modelo de Google y ninguna respondió. Verifica tu API Key o intenta más tarde.")
    return None
            
        # 4. Procesar respuesta
        result = response.json()
        try:
            texto_respuesta = result['candidates'][0]['content']['parts'][0]['text']
            # Limpiar JSON
            if "```json" in texto_respuesta:
                texto_respuesta = texto_respuesta.split("```json")[1].split("```")[0]
            elif "```" in texto_respuesta:
                texto_respuesta = texto_respuesta.split("```")[1].split("```")[0]
            return json.loads(texto_respuesta)
        except:
            st.error("Google respondió, pero no se pudo entender el formato. Intenta otra foto.")
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
        with st.spinner('Procesando...'):
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
            st.download_button("Descargar PDF", pdf_bytes, "tarjeta.pdf", "application/pdf")
