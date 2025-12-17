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
st.set_page_config(page_title="Lector Blindado", layout="wide")
st.title("🚗 Escáner de Tarjetas (Modo Seguro)")

# Sidebar
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if not api_key:
    st.warning("Ingresa tu API Key para comenzar.")
    st.stop()

# ==========================================
# --- FUNCIONES DE VISIÓN (MEJORADAS Y SEGURAS) ---
# ==========================================

def ordenar_puntos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def transformar_perspectiva(image, pts):
    rect = ordenar_puntos(pts)
    (tl, tr, br, bl) = rect
    
    # Calcular dimensiones con seguridad
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[0] - br[0]) ** 2))
    heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[0] - bl[0]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Validación anti-error gris: Si el recorte es minúsculo, abortamos
    if maxWidth < 50 or maxHeight < 50:
        raise ValueError("Recorte demasiado pequeño")

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def mejora_final_color(img_cv2):
    # Aumentar contraste y brillo de forma segura
    img_cv2 = cv2.convertScaleAbs(img_cv2, alpha=1.2, beta=15)
    return img_cv2

def detectar_y_recortar_tarjeta(image_pil):
    try:
        img_orig = np.array(image_pil)
        img_cv = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
        original_h, original_w = img_cv.shape[:2]
        area_total = original_h * original_w

        # 1. Preprocesamiento
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150) # Ajustado para ignorar ruido fino

        # 2. Contornos
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3] # Solo los 3 más grandes
        
        card_contour = None
        
        for c in contours:
            area = cv2.contourArea(c)
            # REGLA DE SEGURIDAD 1: Si el contorno es menor al 10% de la imagen, es ruido. Ignorar.
            if area < (area_total * 0.10):
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                card_contour = approx
                break
        
        # 3. Decisión
        if card_contour is not None:
            pts = card_contour.reshape(4, 2)
            try:
                warped = transformar_perspectiva(img_cv, pts)
                
                # REGLA DE SEGURIDAD 2: Verificar aspecto (que no sea una tira fina)
                h, w = warped.shape[:2]
                if h > 0 and w > 0:
                    aspect_ratio = w / h
                    # Las tarjetas suelen ser rectangulares (ratio entre 1.3 y 1.8 aprox)
                    # Si el ratio es muy loco (ej. 0.1 o 10), el recorte falló.
                    if 0.5 < aspect_ratio < 3.0:
                        st.toast("✅ Recorte exitoso.")
                        warped = mejora_final_color(warped)
                        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            except Exception:
                pass # Si falla el warp, seguimos al fallback

        # FALLBACK (Plan B): Si no se pudo recortar bien, usamos la original mejorada
        st.toast("⚠️ Fondo complejo: Usando imagen completa (Seguro).")
        img_improved = mejora_final_color(img_cv)
        return Image.fromarray(cv2.cvtColor(img_improved, cv2.COLOR_BGR2RGB))
        
    except Exception as e:
        st.error(f"Error procesando imagen: {e}")
        return image_pil

# ==========================================
# --- FIN VISIÓN ---
# ==========================================


# --- API GOOGLE ---
def encontrar_modelo_activo(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url)
        if response.status_code != 200: return 'gemini-1.5-flash'
        data = response.json()
        candidatos = [m['name'].replace('models/', '') for m in data.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
        
        # Prioridad
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
    Analiza esta tarjeta de circulación mexicana. Extrae en JSON:
    'Propietario', 'Placa', 'Serie_VIN', 'Marca', 'Modelo', 'Año', 'Motor'.
    Si algún dato no es visible, pon "ILEGIBLE".
    Responde SOLAMENTE el JSON.
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
            except:
                return None
        return None
    except Exception:
        return None

# --- PDF ---
def generar_pdf(image_pil):
    pdf = FPDF(orientation='P', unit='mm', format='Letter')
    pdf.add_page()
    pdf_w = 190
    img_w, img_h = image_pil.size
    pdf_h = (img_h * pdf_w) / img_w
    if pdf_h > 130:
        pdf_h = 130
        pdf_w = (img_w * pdf_h) / img_h
    x_pos = (216 - pdf_w) / 2
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_pil.save(tmp.name, quality=95)
        pdf.image(tmp.name, x=x_pos, y=10, w=pdf_w, h=pdf_h) 
        
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
            # Paso 1: Intentar recortar, pero si falla, devolver la original mejorada
            img_procesada = detectar_y_recortar_tarjeta(image)
            
            with col2:
                st.image(img_procesada, caption="Resultado (Recorte o Mejora)", use_container_width=True)
            
            # Paso 2: Leer datos de la imagen resultante
            datos = extraer_datos_http(img_procesada, api_key)
            
            if datos:
                df = pd.DataFrame([datos])
                def color_rojo(val):
                    return 'background-color: #ffcccc; color: red; font-weight: bold' if str(val).upper() == 'ILEGIBLE' else ''
                st.subheader("Datos Extraídos")
                st.dataframe(df.style.map(color_rojo))
            else:
                st.error("No se pudieron leer datos. Intenta una foto con menos reflejo.")
            
            # Paso 3: PDF
            pdf_bytes = generar_pdf(img_procesada)
            st.download_button("Descargar PDF", pdf_bytes, "tarjeta_final.pdf", "application/pdf")
