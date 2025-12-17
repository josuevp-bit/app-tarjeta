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
st.set_page_config(page_title="Scanner Pro", layout="wide")
st.title("🚗 Escáner de Tarjetas (Algoritmo de Contraste)")

# Sidebar
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if not api_key:
    st.warning("Ingresa tu API Key para comenzar.")
    st.stop()

# ==========================================
# --- NUEVA VISIÓN ARTIFICIAL (TIPO CAMSCANNER) ---
# ==========================================

def ordenar_puntos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Arriba-Izq
    rect[2] = pts[np.argmax(s)] # Abajo-Der
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Arriba-Der
    rect[3] = pts[np.argmax(diff)] # Abajo-Izq
    return rect

def transformar_perspectiva(image, pts):
    rect = ordenar_puntos(pts)
    (tl, tr, br, bl) = rect
    
    # Calcular ancho máximo
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Calcular alto máximo
    heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[0] - br[0]) ** 2))
    heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[0] - bl[0]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Matriz de destino
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detectar_y_recortar_tarjeta(image_pil):
    try:
        # Convertir a formato OpenCV
        img_orig = np.array(image_pil)
        img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
        original_h, original_w = img.shape[:2]
        
        # 1. Preprocesamiento agresivo para ignorar brillos
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # USAMOS THRESHOLD ADAPTATIVO en vez de Canny
        # Esto separa el fondo oscuro de la tarjeta clara
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Invertimos si la tarjeta es clara sobre fondo oscuro (para tener contornos blancos)
        thresh = cv2.bitwise_not(thresh)
        
        # "Dilatar" para cerrar huecos causados por letras negras o brillos
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # 2. Encontrar contornos en la imagen binaria
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        target_contour = None
        
        for c in contours:
            area = cv2.contourArea(c)
            # Filtro: Debe ser al menos el 15% de la imagen
            if area < (original_h * original_w * 0.15):
                continue
                
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                target_contour = approx
                break
        
        if target_contour is not None:
            pts = target_contour.reshape(4, 2)
            warped = transformar_perspectiva(img, pts)
            
            # Verificación extra: El resultado debe ser horizontal (aprox)
            h, w = warped.shape[:2]
            if h > w: # Si salió vertical, la rotamos
                 warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            
            st.toast("✅ Recorte inteligente exitoso.")
            # Mejora final de imagen recortada
            warped = cv2.convertScaleAbs(warped, alpha=1.2, beta=10)
            return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            
        else:
            # PLAN B MEJORADO: Recorte central
            # Si no encuentra bordes, recorta el 10% de cada lado para quitar mesa
            st.toast("⚠️ Borde difuso. Aplicando recorte central.")
            margin_y = int(original_h * 0.10)
            margin_x = int(original_w * 0.10)
            cropped = img[margin_y:original_h-margin_y, margin_x:original_w-margin_x]
            cropped = cv2.convertScaleAbs(cropped, alpha=1.2, beta=15)
            return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
    except Exception as e:
        print(f"Error cv2: {e}")
        return image_pil

# ==========================================
# --- FIN VISIÓN ---
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
    
    # Lógica para ocupar media hoja superior (aprox 140mm alto max)
    pdf_w_max = 190
    pdf_h_max = 130
    
    img_w, img_h = image_pil.size
    
    # Calcular factor de escala para ajustar al ancho
    ratio = pdf_w_max / img_w
    final_w = pdf_w_max
    final_h = img_h * ratio
    
    # Si la altura se pasa de media hoja, ajustamos por altura
    if final_h > pdf_h_max:
        ratio = pdf_h_max / img_h
        final_h = pdf_h_max
        final_w = img_w * ratio

    # Centrar horizontalmente
    x_pos = (216 - final_w) / 2
    y_pos = 15 

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_pil.save(tmp.name, quality=100)
        pdf.image(tmp.name, x=x_pos, y=y_pos, w=final_w, h=final_h) 
        
    return pdf.output(dest='S').encode('latin1')

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Sube la Tarjeta", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    
    if st.button("Procesar Imagen"):
        with st.spinner('Analizando estructura y recortando...'):
            img_procesada = detectar_y_recortar_tarjeta(image)
            
            with col2:
                st.image(img_procesada, caption="Resultado Final", use_container_width=True)
            
            datos = extraer_datos_http(img_procesada, api_key)
            
            if datos:
                df = pd.DataFrame([datos])
                def color_rojo(val):
                    return 'background-color: #ffcccc; color: red; font-weight: bold' if str(val).upper() == 'ILEGIBLE' else ''
                st.subheader("Datos")
                st.dataframe(df.style.map(color_rojo))
            
            pdf_bytes = generar_pdf(img_procesada)
            st.download_button("Descargar PDF", pdf_bytes, "tarjeta_recortada.pdf", "application/pdf")
