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
st.set_page_config(page_title="Escáner Pro", layout="wide")
st.title("🚗 Escáner de Tarjetas (Recorte Automático)")

# Sidebar
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if not api_key:
    st.warning("Ingresa tu API Key para comenzar.")
    st.stop()

# ==========================================
# --- FUNCIONES AVANZADAS DE VISION (CV2) ---
# ==========================================

def ordenar_puntos(pts):
    # Ordena las coordenadas: arriba-izq, arriba-der, abajo-der, abajo-izq
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def transformar_perspectiva(image, pts):
    # Esta función "endereza" la imagen si está chueca
    rect = ordenar_puntos(pts)
    (tl, tr, br, bl) = rect
    
    # Calcular ancho y alto máximo de la nueva imagen
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[0] - br[0]) ** 2))
    heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[0] - bl[0]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Puntos destino (imagen plana)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    # Calcular matriz de transformación y aplicar
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def mejora_final_color(img_cv2):
    # Aplica mejoras suaves a la imagen ya recortada
    # Aumentar contraste y brillo ligeramente
    img_cv2 = cv2.convertScaleAbs(img_cv2, alpha=1.1, beta=15)
    # Enfoque suave
    gaussian = cv2.GaussianBlur(img_cv2, (0, 0), 2.0)
    img_cv2 = cv2.addWeighted(img_cv2, 1.3, gaussian, -0.3, 0)
    return img_cv2

def detectar_y_recortar_tarjeta(image_pil):
    # 1. Preparar imagen (escala de grises y borrosidad para detectar bordes)
    img_orig = np.array(image_pil)
    img_cv = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR) # BGR para OpenCV
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Detectar bordes (Canny)
    edged = cv2.Canny(blurred, 50, 150)
    
    # 3. Encontrar contornos
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] # Los 5 más grandes
    
    card_contour = None
    for c in contours:
        # Aproximar el contorno a un polígono
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Si el polígono tiene 4 puntos, asumimos que es la tarjeta
        if len(approx) == 4:
            card_contour = approx
            break
            
    # 4. Si se encontró la tarjeta, recortar y enderezar
    if card_contour is not None:
        # reshape para la función de transformación
        pts = card_contour.reshape(4, 2)
        warped = transformar_perspectiva(img_cv, pts)
        st.toast("✅ Tarjeta detectada y recortada automáticamente.")
        # Mejorar la imagen recortada
        warped_improved = mejora_final_color(warped)
        return Image.fromarray(cv2.cvtColor(warped_improved, cv2.COLOR_BGR2RGB))
    else:
        st.toast("⚠️ No se detectaron bordes claros. Usando imagen completa.")
        # Fallback: Si no encuentra bordes, usa la original con mejora simple
        img_improved = mejora_final_color(img_cv)
        return Image.fromarray(cv2.cvtColor(img_improved, cv2.COLOR_BGR2RGB))

# ==========================================
# --- FIN FUNCIONES VISION ---
# ==========================================


# --- 1. FUNCIÓN MAESTRA: BUSCAR MODELO DISPONIBLE ---
def encontrar_modelo_activo(key):
    """Consulta a Google qué modelos tiene disponibles tu cuenta"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            # st.sidebar.error(f"Error listando modelos: {response.status_code}")
            return 'gemini-1.5-flash' # Default seguro
            
        data = response.json()
        modelos = data.get('models', [])
        candidatos = []
        for m in modelos:
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                nombre_limpio = m['name'].replace('models/', '')
                candidatos.append(nombre_limpio)
        
        # Prioridad: flash > pro > experimental
        for m in candidatos:
            if 'flash' in m and 'legacy' not in m: return m
        for m in candidatos:
            if 'pro' in m and 'vision' not in m: return m
        if candidatos: return candidatos[0]
        return 'gemini-1.5-flash' # Fallback final

    except:
        return 'gemini-1.5-flash'

# --- 2. FUNCION OCR ---
def extraer_datos_http(image_pil, key):
    modelo_elegido = encontrar_modelo_activo(key)
    # st.toast(f"🤖 Leyendo con: {modelo_elegido}") # (Opcional: reducir ruido visual)
    
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt_text = """
    Analiza esta tarjeta de circulación. Extrae en JSON:
    'Propietario', 'Placa', 'Serie_VIN', 'Marca', 'Modelo', 'Año', 'Motor'.
    Si no se lee, pon "ILEGIBLE". Solo JSON.
    """
    
    data = {"contents": [{"parts": [{"text": prompt_text}, {"inline_data": {"mime_type": "image/jpeg", "data": img_str}}]}]}
    headers = {'Content-Type': 'application/json'}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo_elegido}:generateContent?key={key}"
    
    try:
        time.sleep(1) # Pequeña pausa para evitar saturación
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            result = response.json()
            try:
                texto = result['candidates'][0]['content']['parts'][0]['text']
                if "```json" in texto: texto = texto.split("```json")[1].split("```")[0]
                elif "```" in texto: texto = texto.split("```")[1].split("```")[0]
                return json.loads(texto)
            except:
                st.error("No se pudo interpretar el resultado de la IA.")
                return None
        elif response.status_code == 429:
            st.warning("⏳ Cuota excedida momentáneamente. Espera unos segundos.")
            return None
        else:
            st.error(f"Error al leer datos ({response.status_code})")
            return None
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

# --- PDF ---
def generar_pdf(image_pil):
    pdf = FPDF(orientation='P', unit='mm', format='Letter')
    pdf.add_page()
    # Calcular dimensiones para centrar en media hoja superior
    # Ancho carta ~216mm. Margen 10mm cada lado -> 196mm ancho útil.
    pdf_w = 190
    # Calcular alto proporcional
    img_w, img_h = image_pil.size
    pdf_h = (img_h * pdf_w) / img_w
    
    # Limitar altura si es muy larga (ej. foto vertical) a media hoja (aprox 130mm)
    if pdf_h > 130:
        pdf_h = 130
        pdf_w = (img_w * pdf_h) / img_h

    # Centrar horizontalmente
    x_pos = (216 - pdf_w) / 2
    y_pos = 15 # Un poco de margen superior

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_pil.save(tmp.name, quality=95)
        pdf.image(tmp.name, x=x_pos, y=y_pos, w=pdf_w, h=pdf_h) 
        
    pdf.set_font("Arial", size=10)
    pdf.text(x_pos, y_pos + pdf_h + 10, "Copia Digital - Tarjeta de Circulación")
    return pdf.output(dest='S').encode('latin1')

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Sube la Tarjeta", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    
    if st.button("Procesar Imagen"):
        with st.spinner('Detectando tarjeta, recortando y leyendo...'):
            
            # 1. NUEVO: Detección y Recorte Automático
            img_mejorada = detectar_y_recortar_tarjeta(image)
            
            with col2:
                st.image(img_mejorada, caption="Procesada (Recortada)", use_container_width=True)
            
            # 2. Extraer datos (Usando la imagen ya recortada)
            datos = extraer_datos_http(img_mejorada, api_key)
            
            if datos:
                df = pd.DataFrame([datos])
                def color_rojo(val):
                    return 'background-color: #ffcccc; color: red; font-weight: bold' if str(val).upper() == 'ILEGIBLE' else ''
                st.subheader("Datos Extraídos")
                st.dataframe(df.style.map(color_rojo))
            
            # 3. PDF
            pdf_bytes = generar_pdf(img_mejorada)
            st.download_button("Descargar PDF", pdf_bytes, "tarjeta_lista.pdf", "application/pdf")
