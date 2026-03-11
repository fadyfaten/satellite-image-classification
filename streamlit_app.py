# ===================================================
# تطبيق Streamlit لتصنيف الصور الفضائية - النسخة النهائية
# ===================================================

import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import tempfile
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import pandas as pd
import time
import gc
import psutil
import platform
import io
import numpy.ma as ma
from skimage.transform import resize
import warnings
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# ===================================================
# إعداد الصفحة - في البداية دائماً
# ===================================================
st.set_page_config(
    page_title="نظام تصنيف الصور الفضائية",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================
# دوال مساعدة
# ===================================================

def get_file_size_mb(file_path):
    """حساب حجم الملف بالـ MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def sanitize_filename(filename):
    """تحويل اسم الملف إلى أحرف آمنة"""
    # استخراج الامتداد
    if '.' in filename:
        name, ext = filename.rsplit('.', 1)
    else:
        name, ext = filename, ''
    
    # إزالة الأحرف غير الآمنة واستبدالها
    safe_name = re.sub(r'[^\w\s-]', '', name)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    
    # إذا أصبح الاسم فارغاً، استخدم وقت الرفع
    if not safe_name:
        safe_name = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # إعادة الاسم مع الامتداد
    if ext:
        return f"{safe_name}.{ext}"
    return safe_name

def normalize_band(band_data):
    """تطبيع آمن للباند"""
    if band_data.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    
    # البحث عن القيم الصالحة
    valid_data = band_data[~np.isnan(band_data)]
    
    if len(valid_data) == 0:
        return np.zeros_like(band_data, dtype=np.uint8)
    
    min_val = valid_data.min()
    max_val = valid_data.max()
    
    if max_val <= min_val:
        return np.zeros_like(band_data, dtype=np.uint8)
    
    # تطبيع مباشر
    normalized = ((band_data - min_val) / (max_val - min_val) * 255)
    normalized = np.nan_to_num(normalized, nan=0).astype(np.uint8)
    
    return normalized

def load_and_align_bands(red_path, green_path, blue_path):
    """تحميل ومحاذاة الباندات الثلاثة"""
    try:
        # فتح الملفات
        with rasterio.open(red_path) as src_r:
            profile = src_r.profile
            crs = src_r.crs
            transform = src_r.transform
            red = src_r.read(1)
            red_nodata = src_r.nodata
        
        with rasterio.open(green_path) as src_g:
            green = src_g.read(1)
            green_nodata = src_g.nodata
        
        with rasterio.open(blue_path) as src_b:
            blue = src_b.read(1)
            blue_nodata = src_b.nodata
        
        # التأكد من تطابق الأبعاد
        height = min(red.shape[0], green.shape[0], blue.shape[0])
        width = min(red.shape[1], green.shape[1], blue.shape[1])
        
        red = red[:height, :width].astype(np.float32)
        green = green[:height, :width].astype(np.float32)
        blue = blue[:height, :width].astype(np.float32)
        
        # استبدال قيم NoData
        for band, nodata in [(red, red_nodata), (green, green_nodata), (blue, blue_nodata)]:
            if nodata is not None:
                band[band == nodata] = np.nan
        
        return red, green, blue, profile, height, width, crs, transform
        
    except Exception as e:
        st.error(f"❌ خطأ في قراءة الملفات: {str(e)}")
        raise e

def process_classification(red, green, blue, model):
    """تصنيف الصورة"""
    
    height, width = red.shape
    
    # إنشاء مصفوفة التصنيف
    classification = np.full((height, width), -1, dtype=np.int8)
    
    # البحث عن البكسلات الصالحة
    valid_mask = ~(np.isnan(red) | np.isnan(green) | np.isnan(blue))
    valid_indices = np.where(valid_mask)
    
    if len(valid_indices[0]) > 0:
        # تجميع القيم الصالحة
        red_valid = red[valid_mask]
        green_valid = green[valid_mask]
        blue_valid = blue[valid_mask]
        
        # تطبيع
        for band in [red_valid, green_valid, blue_valid]:
            if len(band) > 0:
                band_min = band.min()
                band_max = band.max()
                if band_max > band_min:
                    band = (band - band_min) / (band_max - band_min)
        
        # تجهيز features
        X = np.column_stack([red_valid, green_valid, blue_valid])
        
        # تصنيف
        y_pred = model.predict(X)
        
        # حفظ النتائج
        classification[valid_indices] = y_pred
    
    return classification

def create_rgb_preview(red, green, blue, max_size=800):
    """إنشاء صورة RGB مصغرة للعرض"""
    
    h, w = red.shape
    
    # حساب أبعاد الصورة المصغرة
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
    else:
        new_h, new_w = h, w
    
    # أخذ عينة
    row_step = max(1, h // new_h)
    col_step = max(1, w // new_w)
    
    red_sample = red[::row_step, ::col_step].copy()
    green_sample = green[::row_step, ::col_step].copy()
    blue_sample = blue[::row_step, ::col_step].copy()
    
    # معالجة القيم NaN
    for band in [red_sample, green_sample, blue_sample]:
        band[np.isnan(band)] = 0
    
    # تطبيع للعرض
    for band in [red_sample, green_sample, blue_sample]:
        if band.max() > 0:
            band = (band / band.max() * 255).astype(np.uint8)
    
    rgb = np.stack([red_sample, green_sample, blue_sample], axis=-1)
    
    return rgb.astype(np.uint8)

def get_system_info():
    """الحصول على معلومات النظام"""
    info = {
        "النظام": platform.system(),
        "المعالج": platform.processor(),
        "الذاكرة المتوفرة": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
        "إجمالي الذاكرة": f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
    }
    return info

def test_file_validity(file_path):
    """اختبار صلاحية الملف"""
    try:
        with rasterio.open(file_path) as src:
            # محاولة قراءة بعض المعلومات
            profile = src.profile
            shape = src.shape
            return True, f"ملف صالح - الأبعاد: {shape[0]} x {shape[1]}"
    except Exception as e:
        return False, str(e)

# ===================================================
# إنشاء مجلد .streamlit وإعدادات التكوين
# ===================================================
def setup_streamlit_config():
    """إنشاء ملف تكوين Streamlit"""
    config_dir = os.path.join(os.path.dirname(__file__), '.streamlit')
    config_file = os.path.join(config_dir, 'config.toml')
    
    # إنشاء المجلد إذا لم يكن موجوداً
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    # إنشاء ملف التكوين إذا لم يكن موجوداً
    if not os.path.exists(config_file):
        config_content = """
[server]
# زيادة حد رفع الملفات إلى 1 جيجابايت
maxUploadSize = 1024
enableCORS = false
maxMessageSize = 1024

[browser]
# تمكين جمع البيانات
gatherUsageStats = false
"""
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print("✅ تم إنشاء ملف تكوين Streamlit بنجاح")

# استدعاء الدالة لإنشاء ملف التكوين
setup_streamlit_config()

# ===================================================
# CSS مخصص
# ===================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Cairo', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-right: 4px solid #28a745;
        font-weight: 600;
    }
    
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-right: 4px solid #ffc107;
        font-weight: 600;
    }
    
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-right: 4px solid #dc3545;
        font-weight: 600;
    }
    
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-right: 4px solid #17a2b8;
        font-weight: 600;
    }
    
    .band-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 3px solid;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s;
    }
    
    .band-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .red-card { border-color: #ff4444; }
    .green-card { border-color: #44ff44; }
    .blue-card { border-color: #4444ff; }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 0.75rem;
        border: none;
        border-radius: 10px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .file-info {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================
# الترويسة
# ===================================================
st.markdown("""
<div class="main-header">
    <h1>🛰️ نظام تصنيف الصور الفضائية المتقدم</h1>
    <p>باستخدام تقنيات الذكاء الاصطناعي وتحليل المرئيات الفضائية</p>
</div>
""", unsafe_allow_html=True)

# ===================================================
# معلومات رفع الملفات
# ===================================================
st.markdown("""
<div class="file-info">
    📌 الحد الأقصى لرفع الملفات: <b>1 جيجابايت</b><br>
    📌 الصيغ المدعومة: TIFF, TIF, GeoTIFF<br>
    📌 يفضل استخدام أسماء ملفات إنجليزية لتجنب المشاكل
</div>
""", unsafe_allow_html=True)

# ===================================================
# تهيئة session state
# ===================================================
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
if 'bands_ready' not in st.session_state:
    st.session_state['bands_ready'] = False
if 'classification_done' not in st.session_state:
    st.session_state['classification_done'] = False
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = []
if 'upload_errors' not in st.session_state:
    st.session_state['upload_errors'] = []

# ===================================================
# الشريط الجانبي
# ===================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Landsat_8_Logo.svg/200px-Landsat_8_Logo.svg.png", width=200)
    
    st.markdown("## 📊 معلومات النظام")
    
    # عرض معلومات النظام
    sys_info = get_system_info()
    for key, value in sys_info.items():
        st.markdown(f"**{key}:** {value}")
    
    st.markdown("---")
    st.markdown("## 📦 تحميل النموذج")
    
    # خيارات تحميل النموذج
    model_source = st.radio(
        "مصدر النموذج:",
        ["📁 المسار المحلي", "📤 رفع ملف النموذج"]
    )
    
    model = None
    model_info = None
    
    if model_source == "📁 المسار المحلي":
        project_path = st.text_input(
            "مسار المشروع:",
            value=r'C:\Users\ps\Desktop\Final project'
        )
        
        model_path = os.path.join(project_path, 'models', 'decision_tree_model.pkl')
        info_path = os.path.join(project_path, 'models', 'model_info.pkl')
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                if os.path.exists(info_path):
                    model_info = joblib.load(info_path)
                st.markdown('<div class="success-box">✅ النموذج محمل بنجاح</div>', unsafe_allow_html=True)
                st.session_state['model_loaded'] = True
                st.session_state['model'] = model
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ خطأ في تحميل النموذج: {e}</div>', unsafe_allow_html=True)
                st.session_state['model_loaded'] = False
        else:
            st.markdown(f'<div class="warning-box">⚠️ النموذج غير موجود في المسار</div>', unsafe_allow_html=True)
            st.session_state['model_loaded'] = False
    
    else:
        uploaded_model = st.file_uploader(
            "اختر ملف النموذج (.pkl)",
            type=['pkl']
        )
        
        if uploaded_model is not None:
            try:
                safe_name = sanitize_filename(uploaded_model.name)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                    tmp.write(uploaded_model.getvalue())
                    model = joblib.load(tmp.name)
                    st.session_state['processed_files'].append(tmp.name)
                
                st.markdown('<div class="success-box">✅ تم تحميل النموذج بنجاح</div>', unsafe_allow_html=True)
                st.session_state['model_loaded'] = True
                st.session_state['model'] = model
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ خطأ في تحميل النموذج: {e}</div>', unsafe_allow_html=True)
                st.session_state['model_loaded'] = False
    
    # عرض معلومات النموذج
    if model_info and st.session_state['model_loaded']:
        st.markdown("---")
        st.markdown("## 📈 أداء النموذج")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("تدريب", f"{model_info.get('accuracy_train', 0):.1%}")
        with col2:
            st.metric("تحقق", f"{model_info.get('accuracy_val', 0):.1%}")
        with col3:
            st.metric("اختبار", f"{model_info.get('accuracy_test', 0):.1%}")
        
        if 'feature_importance' in model_info:
            st.markdown("### 🎯 أهمية الباندات")
            for band, imp in zip(['Red', 'Green', 'Blue'], model_info['feature_importance']):
                st.progress(float(imp), text=f"{band}: {imp:.1%}")
    
    st.markdown("---")
    st.markdown("## 📋 تعليمات الاستخدام")
    
    with st.expander("📖 طريقة الاستخدام", expanded=True):
        st.markdown("""
        1. **تأكد من تحميل النموذج** ✅
        2. **ارفع الباندات الثلاثة**:
           - 🔴 الباند الأحمر
           - 🟢 الباند الأخضر
           - 🔵 الباند الأزرق
        3. **انتظر معاينة الباندات**
        4. **اضغط بدء التصنيف**
        5. **حمّل النتائج**
        """)

# ===================================================
# رفع الباندات
# ===================================================
st.header("📤 رفع الباندات الثلاثة")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="band-card red-card">', unsafe_allow_html=True)
    st.markdown("### 🔴 الباند الأحمر (Red)")
    red_file = st.file_uploader(
        "اختر صورة الباند الأحمر",
        type=['tif', 'tiff', 'geotiff'],
        key="red_uploader",
        help="اختر ملف TIFF أو GeoTIFF"
    )
    if red_file is not None:
        try:
            safe_name = sanitize_filename(red_file.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix='_red.tif') as tmp:
                tmp.write(red_file.getvalue())
                file_path = tmp.name
                st.session_state['red_path'] = file_path
                st.session_state['processed_files'].append(file_path)
            
            # اختبار صلاحية الملف
            is_valid, message = test_file_validity(file_path)
            if is_valid:
                st.success(f"✅ {safe_name}")
                st.caption(f"الحجم: {get_file_size_mb(file_path):.2f} MB")
                st.caption(f"📊 {message}")
            else:
                st.error(f"❌ الملف غير صالح: {message}")
                # حذف الملف المؤقت
                try:
                    os.unlink(file_path)
                    if 'red_path' in st.session_state:
                        del st.session_state['red_path']
                except:
                    pass
        except Exception as e:
            st.error(f"❌ خطأ في رفع الملف: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="band-card green-card">', unsafe_allow_html=True)
    st.markdown("### 🟢 الباند الأخضر (Green)")
    green_file = st.file_uploader(
        "اختر صورة الباند الأخضر",
        type=['tif', 'tiff', 'geotiff'],
        key="green_uploader",
        help="اختر ملف TIFF أو GeoTIFF"
    )
    if green_file is not None:
        try:
            safe_name = sanitize_filename(green_file.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix='_green.tif') as tmp:
                tmp.write(green_file.getvalue())
                file_path = tmp.name
                st.session_state['green_path'] = file_path
                st.session_state['processed_files'].append(file_path)
            
            # اختبار صلاحية الملف
            is_valid, message = test_file_validity(file_path)
            if is_valid:
                st.success(f"✅ {safe_name}")
                st.caption(f"الحجم: {get_file_size_mb(file_path):.2f} MB")
                st.caption(f"📊 {message}")
            else:
                st.error(f"❌ الملف غير صالح: {message}")
                # حذف الملف المؤقت
                try:
                    os.unlink(file_path)
                    if 'green_path' in st.session_state:
                        del st.session_state['green_path']
                except:
                    pass
        except Exception as e:
            st.error(f"❌ خطأ في رفع الملف: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="band-card blue-card">', unsafe_allow_html=True)
    st.markdown("### 🔵 الباند الأزرق (Blue)")
    blue_file = st.file_uploader(
        "اختر صورة الباند الأزرق",
        type=['tif', 'tiff', 'geotiff'],
        key="blue_uploader",
        help="اختر ملف TIFF أو GeoTIFF"
    )
    if blue_file is not None:
        try:
            safe_name = sanitize_filename(blue_file.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix='_blue.tif') as tmp:
                tmp.write(blue_file.getvalue())
                file_path = tmp.name
                st.session_state['blue_path'] = file_path
                st.session_state['processed_files'].append(file_path)
            
            # اختبار صلاحية الملف
            is_valid, message = test_file_validity(file_path)
            if is_valid:
                st.success(f"✅ {safe_name}")
                st.caption(f"الحجم: {get_file_size_mb(file_path):.2f} MB")
                st.caption(f"📊 {message}")
            else:
                st.error(f"❌ الملف غير صالح: {message}")
                # حذف الملف المؤقت
                try:
                    os.unlink(file_path)
                    if 'blue_path' in st.session_state:
                        del st.session_state['blue_path']
                except:
                    pass
        except Exception as e:
            st.error(f"❌ خطأ في رفع الملف: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# ===================================================
# التحقق من رفع الباندات
# ===================================================
all_bands_loaded = all([
    'red_path' in st.session_state,
    'green_path' in st.session_state,
    'blue_path' in st.session_state
])

if all_bands_loaded:
    st.markdown("""
    <div class="success-box">
        ✅ تم رفع جميع الباندات بنجاح! يمكنك الآن معاينة الصور.
    </div>
    """, unsafe_allow_html=True)

# ===================================================
# معاينة الباندات
# ===================================================
if all_bands_loaded:
    st.markdown("---")
    st.header("👁️ معاينة وتحليل الباندات")
    
    try:
        with st.spinner('جاري تحميل ومعالجة الباندات...'):
            red, green, blue, profile, height, width, crs, transform = load_and_align_bands(
                st.session_state['red_path'],
                st.session_state['green_path'],
                st.session_state['blue_path']
            )
        
        # عرض معلومات الأبعاد
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📐 الأبعاد", f"{height} x {width}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🗺️ نظام الإحداثيات", str(crs).split(':')[-1] if crs else "غير معروف")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 إجمالي البكسلات", f"{height * width:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # عرض إحصائيات الباندات
        st.subheader("📈 إحصائيات الباندات")
        stats_data = []
        for name, band in [('Red', red), ('Green', green), ('Blue', blue)]:
            valid_data = band[~np.isnan(band)]
            if len(valid_data) > 0:
                stats_data.append({
                    'الباند': name,
                    'الحد الأدنى': f"{valid_data.min():.2f}",
                    'الحد الأقصى': f"{valid_data.max():.2f}",
                    'المتوسط': f"{valid_data.mean():.2f}"
                })
        st.table(pd.DataFrame(stats_data))
        
        # إنشاء وعرض الصورة المصغرة
        st.subheader("🖼️ صورة RGB تجريبية")
        preview = create_rgb_preview(red, green, blue)
        st.image(preview, caption="صورة RGB مصغرة", use_container_width=True)
        
        # حفظ الباندات
        st.session_state['processed_red'] = red
        st.session_state['processed_green'] = green
        st.session_state['processed_blue'] = blue
        st.session_state['profile'] = profile
        st.session_state['height'] = height
        st.session_state['width'] = width
        st.session_state['bands_ready'] = True
        
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ خطأ في معالجة الباندات: {str(e)}</div>', unsafe_allow_html=True)
        st.session_state['bands_ready'] = False

# ===================================================
# بدء التصنيف
# ===================================================
if (st.session_state.get('bands_ready', False) and 
    st.session_state.get('model_loaded', False)):
    
    st.markdown("---")
    st.header("🚀 تصنيف الصورة")
    
    if st.button("🚀 بدء التصنيف", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()
        
        start_time = time.time()
        
        try:
            status_text.text("📦 تجهيز البيانات...")
            progress_bar.progress(30)
            
            red = st.session_state['processed_red']
            green = st.session_state['processed_green']
            blue = st.session_state['processed_blue']
            
            status_text.text("🤖 جاري التصنيف...")
            progress_bar.progress(60)
            
            # تصنيف الصورة
            classification = process_classification(
                red, green, blue, 
                st.session_state['model']
            )
            
            status_text.text("🖼️ تجهيز النتائج...")
            progress_bar.progress(90)
            
            # حفظ النتائج
            st.session_state['classification'] = classification
            st.session_state['classification_done'] = True
            
            # إنشاء صور للعرض
            red_norm = normalize_band(red)
            green_norm = normalize_band(green)
            blue_norm = normalize_band(blue)
            st.session_state['display_image'] = np.stack([red_norm, green_norm, blue_norm], axis=-1)
            
            # عينة من التصنيف للعرض
            h, w = classification.shape
            sample_size = min(500, h, w)
            step_h = max(1, h // sample_size)
            step_w = max(1, w // sample_size)
            st.session_state['display_class'] = classification[::step_h, ::step_w]
            
            progress_bar.progress(100)
            elapsed_time = time.time() - start_time
            
            status_text.text("✅ تم التصنيف بنجاح!")
            time_text.text(f"⏱️ الوقت المستغرق: {elapsed_time:.2f} ثانية")
            
            # إحصائيات سريعة
            valid_pixels = classification[classification != -1]
            if len(valid_pixels) > 0:
                unique, counts = np.unique(valid_pixels, return_counts=True)
                total_classified = np.sum(counts)
                
                stats_summary = "📊 إحصائيات: "
                for cls, count in zip(unique, counts):
                    percentage = count / total_classified * 100
                    stats_summary += f"{['عمراني', 'زراعي', 'مائي'][int(cls)]}: {percentage:.1f}% | "
                
                st.info(stats_summary)
            
            st.balloons()
            
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ خطأ أثناء التصنيف: {str(e)}</div>', unsafe_allow_html=True)
        
        finally:
            progress_bar.empty()
            status_text.empty()
            time_text.empty()
            gc.collect()

# ===================================================
# عرض النتائج
# ===================================================
if st.session_state.get('classification_done', False):
    st.markdown("---")
    st.header("📊 نتائج التصنيف")
    
    # عرض الصور
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ الصورة الأصلية")
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(st.session_state['display_image'])
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader("🏷️ خريطة التصنيف")
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        
        cmap = ListedColormap(['red', 'green', 'blue'])
        display_class = st.session_state['display_class'].copy()
        display_class_masked = ma.masked_where(display_class == -1, display_class)
        
        im = ax2.imshow(display_class_masked, cmap=cmap, alpha=0.8)
        ax2.axis('off')
        
        patches = [
            mpatches.Patch(color='red', label='🏙️ عمراني'),
            mpatches.Patch(color='green', label='🌾 زراعي'),
            mpatches.Patch(color='blue', label='💧 مائي')
        ]
        ax2.legend(handles=patches, loc='upper left')
        
        st.pyplot(fig2)
        plt.close(fig2)
    
    # إحصائيات مفصلة
    st.markdown("---")
    st.subheader("📈 إحصائيات مفصلة")
    
    classification = st.session_state['classification']
    valid_pixels = classification[classification != -1]
    
    if len(valid_pixels) > 0:
        unique, counts = np.unique(valid_pixels, return_counts=True)
        total_pixels = len(valid_pixels)
        
        # جدول الإحصائيات
        stats_data = []
        class_names = ['مناطق عمرانية', 'مناطق زراعية', 'مسطحات مائية']
        
        for cls, count in zip(unique, counts):
            percentage = count / total_pixels * 100
            stats_data.append({
                'الفئة': class_names[int(cls)],
                'عدد البكسلات': f"{count:,}",
                'النسبة المئوية': f"{percentage:.2f}%"
            })
        
        st.table(pd.DataFrame(stats_data))
    
    # تنزيل النتائج
    st.markdown("---")
    st.subheader("📥 تنزيل النتائج")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # تنزيل الصورة المصنفة
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_output:
                output_path = tmp_output.name
            
            profile = st.session_state['profile']
            profile.update(
                count=1,
                dtype='int8',
                compress='lzw',
                nodata=-1
            )
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(st.session_state['classification'].astype('int8'), 1)
            
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="📥 تنزيل الصورة المصنفة",
                    data=f,
                    file_name="classified_image.tif",
                    mime="image/tiff",
                    use_container_width=True
                )
            
            # تنظيف الملف المؤقت
            try:
                os.unlink(output_path)
            except:
                pass
        except Exception as e:
            st.error(f"خطأ في إنشاء ملف التنزيل: {str(e)}")
    
    with col2:
        # تنزيل الإحصائيات
        if len(valid_pixels) > 0:
            stats_df = pd.DataFrame(stats_data)
            csv_buffer = io.StringIO()
            stats_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 تنزيل الإحصائيات",
                data=csv_buffer.getvalue(),
                file_name="classification_stats.csv",
                mime="text/csv",
                use_container_width=True
            )

# ===================================================
# تنظيف الملفات المؤقتة
# ===================================================
def cleanup():
    """تنظيف الملفات المؤقتة"""
    for file_path in st.session_state.get('processed_files', []):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass

import atexit
atexit.register(cleanup)

# ===================================================
# نصائح
# ===================================================
if not st.session_state.get('model_loaded'):
    st.markdown("""
    <div class="info-box">
        💡 يرجى تحميل النموذج أولاً من القائمة الجانبية
    </div>
    """, unsafe_allow_html=True)

if st.session_state.get('model_loaded') and not all_bands_loaded:
    st.markdown("""
    <div class="info-box">
        💡 يرجى رفع الباندات الثلاثة للبدء
    </div>
    """, unsafe_allow_html=True)

# ===================================================
# تنزيل الصفحة
# ===================================================
st.markdown("""
<div class="footer">
    <p>© 2025 - نظام تصنيف الصور الفضائية المتقدم | الإصدار النهائي</p>
    <p>يدعم رفع الملفات حتى 1 جيجابايت - صيغ TIFF, GeoTIFF</p>
</div>
""", unsafe_allow_html=True)