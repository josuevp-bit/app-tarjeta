import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import base64
import cv2
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from fpdf import FPDF
import io
import tempfile
import json
import time

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Scanner 4 Puntos", layout="wide")
st.title("📐 Escáner de Perspectiva (4 Puntos)")

# Sidebar
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key", type="password")

if not api_key:
    st.warning("Ingresa tu API Key para comenzar.")
    st.stop()

# ==========================================
# --- TRUCO PARA QUE SE VEA LA IMAGEN ---
# ==========================================
def imagen_para_canvas(image_pil):
    """Prepara la imagen para que el navegador la pueda mostrar sin errores"""
    # 1. Asegurar que sea RGB (quita transparencias raras)
    img = image_pil.convert("RGB")
    # 2. Guardar en memoria
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    # 3. Codificar
    img_str = base64.b64encode(buffered.getvalue()).decode()
    # 4. Retornar objeto Imagen compatible (NO string) para st_canvas
    return Image.open(io.BytesIO(base64.b64decode(img_str)))

# ==========================================
# --- LÓGICA DE GEOMETRÍA ---
# ==========================================

def ordenar_puntos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

def enderezar_perspectiva(image_pil, puntos_canvas, factor_escala):
    img = np.array(image_pil.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    pts = np.array(puntos_canvas, dtype="float32") * factor_escala
    rect = ordenar_puntos(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[0] - br[0]) ** 2))
    heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[0] - bl[0]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

# ==========================================
# --- MEJORA HD ---
# ==========================================
def mejora_hd(image_pil):
    img = np.array(image_pil)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Mejora suave
    img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 3, 3, 7, 21)
    
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    img_cv = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    gaussian = cv2.GaussianBlur(img_cv, (0, 0), 3.0)
    img_cv = cv2.addWeighted(img_cv, 1.5, gaussian, -0.5, 0)
    
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# ==========================================
# --- CONEXIÓN GOOGLE ---
# ==========================================
def encontrar_modelo_activo(key):
    # Forzamos flash porque es el más estable
    return 'gemini-1.5-flash'

def extraer_datos_http(image_pil, key):
    modelo_elegido = encontrar_modelo_activo(key)
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt_text = """
    Analiza esta tarjeta de circulación. Extrae datos en JSON:
    'Propietario', 'Placa', 'Serie_VIN', 'Marca', 'Modelo', 'Año', 'Motor'.
    Si no es visible, pon "ILEGIBLE". Solo responde JSON.
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
if 'rotation' not in st.session_state: st.session_state.rotation = 0

uploaded_file = st.file_uploader("1. Sube tu foto", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Cargar y corregir orientación EXIF automática
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("↺ Rotar Izq"): st.session_state.rotation += 90
    with col_r2:
        if st.button("↻ Rotar Der"): st.session_state.rotation -= 90
    
    if st.session_state.rotation != 0:
        image = image.rotate(st.session_state.rotation, expand=True)

    # --- SELECCIÓN DE 4 PUNTOS ---
    st.write("### 2. Marca las 4 esquinas")
    st.info("Haz clic en las 4 esquinas de la tarjeta.")

    # Ajuste de tamaño seguro para visualización
    # Usamos un ancho fijo de 700px para que se vea bien en laptop
    ancho_canvas = 700
    w_original, h_original = image.size
    factor_escala = w_original / ancho_canvas
    alto_canvas = int(h_original / factor_escala)
    
    # Redimensionamos la imagen SOLO para mostrarla en el canvas
    img_resized = image.resize((ancho_canvas, alto_canvas))
    
    # Limpiamos la imagen para evitar errores de formato
    bg_image = imagen_para_canvas(img_resized)

    # Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=bg_image, # Aquí pasamos la imagen ya curada
        update_streamlit=True,
        height=alto_canvas,
        width=ancho_canvas,
        drawing_mode="point", 
        point_display_radius=6,
        key="canvas",
    )

    puntos = []
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        puntos = [[obj["left"], obj["top"]] for obj in objects if obj["type"] == "circle"]

    st.caption(f"Puntos marcados: {len(puntos)} / 4")

    if len(puntos) == 4:
        if st.button("✅ PROCESAR", type="primary", use_container_width=True):
            
            with st.spinner('Transformando...'):
                img_warp = enderezar_perspectiva(image, puntos, factor_escala)
                img_final = mejora_hd(img_warp)
                
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    st.subheader("Imagen Final")
                    st.image(img_final, caption="Resultado", use_container_width=True)
                
                datos = extraer_datos_http(img_final, api_key)
                
                with col_res2:
                    st.subheader("Datos")
                    if datos:
                        df = pd.DataFrame([datos])
                        st.data_editor(df, num_rows="dynamic")
                    else:
                        st.error("No se pudo leer texto.")
                
                pdf_bytes = generar_pdf(img_final)
                st.download_button("Descargar PDF", pdf_bytes, "tarjeta.pdf", "application/pdf", use_container_width=True)
    
    elif len(puntos) > 4:
        st.warning("Demasiados puntos. Usa la flecha 'Deshacer' (⟲) en el menú del canvas.")
