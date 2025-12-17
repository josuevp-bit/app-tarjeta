import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from fpdf import FPDF
import io
import tempfile

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Lector de Tarjetas de Circulación", layout="wide")

st.title("🚗 Escáner y Mejorador de Tarjetas de Circulación")
st.markdown("Sube tu tarjeta, obtén los datos y descarga el PDF optimizado.")

# Sidebar para API Key (Para seguridad)
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")
if not api_key:
    st.warning("Por favor ingresa tu API Key en la barra lateral para continuar.")
    st.stop()

genai.configure(api_key=api_key)

# --- FUNCIONES DE PROCESAMIENTO DE IMAGEN ---
def mejorar_imagen(image_pil):
    # Convertir PIL a OpenCV
    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Reducción de Ruido (Denoising)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # 2. Mejora de Nitidez (Sharpening kernel)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    # 3. Corrección de Iluminación (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    # Esto ayuda con los "rayos de luz" o zonas oscuras
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 4. Saturación (Coloreado vívido)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 30) # Aumentar saturación
    img = cv2.merge((h, s, v))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return Image.fromarray(img)

# --- FUNCION DE EXTRACCIÓN DE DATOS (GEMINI) ---
def extraer_datos(image_pil):
    model = genai.GenerativeModel('gemini-pro-vision')
    
    prompt = """
    Actúa como un sistema OCR experto en tarjetas de circulación mexicanas.
    Analiza esta imagen y extrae la siguiente información en formato JSON estricto.
    Campos requeridos: 'Propietario', 'Placa', 'Serie_VIN', 'Marca', 'Modelo', 'Año', 'Motor'.
    
    Reglas:
    1. Si un dato es legible, escríbelo tal cual.
    2. Si un dato NO es legible o está borroso, escribe exactamente la palabra "ILEGIBLE".
    3. No inventes datos.
    """
    
    response = model.generate_content([prompt, image_pil])
    try:
        # Limpieza básica para obtener el JSON del texto
        text = response.text.replace("```json", "").replace("```", "")
        return eval(text) # Convertir string a dict
    except:
        return None

# --- FUNCION PDF ---
def generar_pdf(image_pil):
    pdf = FPDF(orientation='P', unit='mm', format='Letter')
    pdf.add_page()
    
    # Guardar imagen temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_pil.save(tmp.name)
        
        # Tamaño carta es aprox 216x279 mm. Media hoja es 216 x 140.
        # Ajustamos la imagen para que ocupe la mitad superior (aprox 190mm de ancho para márgenes)
        pdf.image(tmp.name, x=10, y=10, w=195, h=120) 
        
    pdf.set_font("Arial", size=12)
    pdf.text(10, 140, "Copia de Tarjeta de Circulación - Verificación")
    
    return pdf.output(dest='S').encode('latin1')

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Sube la foto de la Tarjeta", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    
    with st.spinner('Mejorando imagen y leyendo datos...'):
        # 1. Procesar Imagen
        img_mejorada = mejorar_imagen(image)
        
        with col2:
            st.image(img_mejorada, caption="Mejorada (Nitidez + Luz)", use_container_width=True)
        
        # 2. Extraer Datos
        datos = extraer_datos(image)
        
        if datos:
            df = pd.DataFrame([datos])
            
            # Lógica para marcar en rojo (Styling)
            def color_rojo_si_ilegible(val):
                color = 'background-color: #ffcccc; color: red; font-weight: bold' if str(val).upper() == 'ILEGIBLE' else ''
                return color

            st.subheader("📋 Datos Extraídos (Copia y Pega)")
            st.dataframe(df.style.map(color_rojo_si_ilegible))
        else:
            st.error("No se pudieron leer los datos. Intenta con una foto más clara.")

        # 3. Descargar PDF
        pdf_bytes = generar_pdf(img_mejorada)
        st.download_button(
            label="📄 Descargar PDF (Media Carta)",
            data=pdf_bytes,
            file_name="tarjeta_circulacion_mejorada.pdf",
            mime="application/pdf"
        )
