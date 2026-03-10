# ===================================================
# تطبيق Streamlit لتصنيف الصور الفضائية - نظام رفع ثلاث صور منفصلة
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
from rasterio.windows import Window
import gc
import psutil
import platform
from PIL import Image

# إعداد الصفحة
st.set_page_config(
    page_title="تصنيف الصور الفضائية",
    page_icon="🛰️",
    layout="wide"
)

# ===================================================
# دوال مساعدة
# ===================================================

def get_file_size_mb(file_path):
    """حساب حجم الملف بالـ MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def normalize_band(band_data):
    """تطبيع آمن للباند"""
    band_data = band_data.astype(np.float32)
    min_val = band_data.min()
    max_val = band_data.max()
    
    if max_val > min_val:
        normalized = ((band_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(band_data, dtype=np.uint8)
    
    return normalized

def load_and_align_bands(red_path, green_path, blue_path):
    """تحميل ومحاذاة الباندات الثلاثة"""
    
    # فتح الملفات
    with rasterio.open(red_path) as src_r:
        red = src_r.read(1).astype(np.float32)
        profile = src_r.profile
        crs = src_r.crs
        transform = src_r.transform
    
    with rasterio.open(green_path) as src_g:
        green = src_g.read(1).astype(np.float32)
    
    with rasterio.open(blue_path) as src_b:
        blue = src_b.read(1).astype(np.float32)
    
    # التأكد من تطابق الأبعاد
    height = min(red.shape[0], green.shape[0], blue.shape[0])
    width = min(red.shape[1], green.shape[1], blue.shape[1])
    
    red = red[:height, :width]
    green = green[:height, :width]
    blue = blue[:height, :width]
    
    return red, green, blue, profile, height, width, crs, transform

def process_classification(red, green, blue, model):
    """تصنيف الصورة باستخدام الباندات الثلاثة"""
    
    height, width = red.shape
    
    # تطبيع
    for band in [red, green, blue]:
        band_max = band.max()
        if band_max > 0:
            band /= band_max
    
    # تجهيز features
    X = np.column_stack([
        red.reshape(-1),
        green.reshape(-1),
        blue.reshape(-1)
    ])
    
    # تصنيف
    y_pred = model.predict(X)
    classification = y_pred.reshape(height, width)
    
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
    
    red_sample = red[::row_step, ::col_step]
    green_sample = green[::row_step, ::col_step]
    blue_sample = blue[::row_step, ::col_step]
    
    # تطبيع للعرض
    for band in [red_sample, green_sample, blue_sample]:
        band_max = band.max()
        if band_max > 0:
            band[:] = (band / band_max * 255)
    
    rgb = np.stack([red_sample, green_sample, blue_sample], axis=-1).astype(np.uint8)
    
    return rgb

# ===================================================
# CSS مخصص
# ===================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-right: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-right: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-right: 4px solid #dc3545;
    }
    .band-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid;
        margin: 0.5rem 0;
    }
    .red-card { border-color: #ff4444; }
    .green-card { border-color: #44ff44; }
    .blue-card { border-color: #4444ff; }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================
# الترويسة
# ===================================================
st.markdown('<div class="main-header"><h1>🛰️ نظام تصنيف الصور الفضائية</h1></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">✨ نظام رفع ثلاث صور منفصلة (واحدة لكل باند) ✨</p>', unsafe_allow_html=True)

# ===================================================
# تحميل النموذج
# ===================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Landsat_8_Logo.svg/200px-Landsat_8_Logo.svg.png", width=200)
    
    st.markdown("## 📊 معلومات النموذج")
    
    # المسارات
    project_path = r'C:\Users\ps\Desktop\project'
    model_path = os.path.join(project_path, 'models', 'decision_tree_model.pkl')
    info_path = os.path.join(project_path, 'models', 'model_info.pkl')
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.session_state['model'] = model
            st.session_state['model_loaded'] = True
            st.markdown('<div class="success-box">✅ النموذج محمل بنجاح</div>', unsafe_allow_html=True)
            
            # محاولة تحميل معلومات النموذج
            if os.path.exists(info_path):
                model_info = joblib.load(info_path)
                
                st.markdown("### 📈 أداء النموذج")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("تدريب", f"{model_info.get('accuracy_train', 0.95):.1%}")
                with col2:
                    st.metric("تحقق", f"{model_info.get('accuracy_val', 0.94):.1%}")
                with col3:
                    st.metric("اختبار", f"{model_info.get('accuracy_test', 0.93):.1%}")
                
                if 'feature_importance' in model_info:
                    st.markdown("### 🎯 أهمية الباندات")
                    for band, imp in zip(['Red', 'Green', 'Blue'], model_info['feature_importance']):
                        st.progress(float(imp), text=f"{band}: {imp:.1%}")
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ خطأ في تحميل النموذج: {e}</div>', unsafe_allow_html=True)
            st.session_state['model_loaded'] = False
    else:
        st.markdown(f'<div class="warning-box">⚠️ النموذج غير موجود في المسار:\n{model_path}</div>', unsafe_allow_html=True)
        st.session_state['model_loaded'] = False
    
    st.markdown("---")
    st.markdown("### 📥 معلومات")
    st.markdown("""
    **طريقة الاستخدام:**
    1. ارفع الباند الأحمر (Red)
    2. ارفع الباند الأخضر (Green)
    3. ارفع الباند الأزرق (Blue)
    4. اضغط "بدء التصنيف"
    """)

# ===================================================
# رفع الصور الثلاثة
# ===================================================
st.header("📤 رفع الباندات الثلاثة")

st.markdown(f'''
<div class="file-info">
    📌 الحد الأقصى لكل ملف: <b>1GB</b><br>
    📌 الصيغ المدعومة: TIFF, TIF, GeoTIFF
</div>
''', unsafe_allow_html=True)

# ثلاثة أعمدة لرفع الباندات
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="band-card red-card">', unsafe_allow_html=True)
    st.markdown("### 🔴 الباند الأحمر (Red)")
    red_file = st.file_uploader(
        "اختر صورة الباند الأحمر",
        type=['tif', 'tiff', 'geotiff'],
        key="red_uploader"
    )
    if red_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='_red.tif') as tmp:
            tmp.write(red_file.getvalue())
            st.session_state['red_path'] = tmp.name
        st.success(f"✅ تم رفع الباند الأحمر: {red_file.name}")
        st.metric("الحجم", f"{get_file_size_mb(st.session_state['red_path']):.2f} MB")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="band-card green-card">', unsafe_allow_html=True)
    st.markdown("### 🟢 الباند الأخضر (Green)")
    green_file = st.file_uploader(
        "اختر صورة الباند الأخضر",
        type=['tif', 'tiff', 'geotiff'],
        key="green_uploader"
    )
    if green_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='_green.tif') as tmp:
            tmp.write(green_file.getvalue())
            st.session_state['green_path'] = tmp.name
        st.success(f"✅ تم رفع الباند الأخضر: {green_file.name}")
        st.metric("الحجم", f"{get_file_size_mb(st.session_state['green_path']):.2f} MB")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="band-card blue-card">', unsafe_allow_html=True)
    st.markdown("### 🔵 الباند الأزرق (Blue)")
    blue_file = st.file_uploader(
        "اختر صورة الباند الأزرق",
        type=['tif', 'tiff', 'geotiff'],
        key="blue_uploader"
    )
    if blue_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='_blue.tif') as tmp:
            tmp.write(blue_file.getvalue())
            st.session_state['blue_path'] = tmp.name
        st.success(f"✅ تم رفع الباند الأزرق: {blue_file.name}")
        st.metric("الحجم", f"{get_file_size_mb(st.session_state['blue_path']):.2f} MB")
    st.markdown('</div>', unsafe_allow_html=True)

# التحقق من رفع جميع الباندات
all_bands_loaded = all([
    'red_path' in st.session_state,
    'green_path' in st.session_state,
    'blue_path' in st.session_state
])

if all_bands_loaded:
    st.markdown('<div class="success-box">✅ تم رفع جميع الباندات بنجاح!</div>', unsafe_allow_html=True)

# ===================================================
# معاينة الباندات
# ===================================================
if all_bands_loaded:
    st.markdown("---")
    st.header("👁️ معاينة الباندات")
    
    try:
        # تحميل الباندات
        red, green, blue, profile, height, width, crs, transform = load_and_align_bands(
            st.session_state['red_path'],
            st.session_state['green_path'],
            st.session_state['blue_path']
        )
        
        # عرض معلومات الأبعاد
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("أبعاد الصورة", f"{height} x {width}")
        with col2:
            st.metric("نظام الإحداثيات", str(crs).split(':')[-1] if crs else "غير معروف")
        with col3:
            st.metric("إجمالي البكسلات", f"{height * width:,}")
        
        # إنشاء صورة RGB مصغرة للعرض
        preview = create_rgb_preview(red, green, blue)
        
        # عرض الصورة المصغرة
        st.image(preview, caption="صورة RGB تجريبية (مصغرة)", use_container_width=True)
        
        # حفظ الباندات في session state
        st.session_state['processed_red'] = red
        st.session_state['processed_green'] = green
        st.session_state['processed_blue'] = blue
        st.session_state['profile'] = profile
        st.session_state['height'] = height
        st.session_state['width'] = width
        st.session_state['bands_ready'] = True
        
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ خطأ في معالجة الباندات: {e}</div>', unsafe_allow_html=True)
        st.session_state['bands_ready'] = False

# ===================================================
# بدء التصنيف
# ===================================================
if (st.session_state.get('bands_ready', False) and 
    st.session_state.get('model_loaded', False)):
    
    st.markdown("---")
    st.header("🚀 تصنيف الصورة")
    
    if st.button("بدء التصنيف", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()
        
        start_time = time.time()
        
        try:
            status_text.text("جاري تصنيف الصورة...")
            progress_bar.progress(30)
            
            # تنفيذ التصنيف
            classification = process_classification(
                st.session_state['processed_red'],
                st.session_state['processed_green'],
                st.session_state['processed_blue'],
                st.session_state['model']
            )
            
            progress_bar.progress(70)
            
            # حفظ النتائج
            st.session_state['classification'] = classification
            st.session_state['classification_done'] = True
            
            # إنشاء صور للعرض
            red_norm = normalize_band(st.session_state['processed_red'])
            green_norm = normalize_band(st.session_state['processed_green'])
            blue_norm = normalize_band(st.session_state['processed_blue'])
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
            st.balloons()
            
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ خطأ أثناء التصنيف: {e}</div>', unsafe_allow_html=True)
            with st.expander("تفاصيل الخطأ"):
                import traceback
                st.code(traceback.format_exc())
        
        finally:
            progress_bar.empty()
            status_text.empty()
            time_text.empty()

# ===================================================
# عرض النتائج
# ===================================================
if st.session_state.get('classification_done', False):
    st.markdown("---")
    st.header("📊 نتائج التصنيف")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ الصورة الأصلية (RGB)")
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(st.session_state['display_image'])
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader("🏷️ نتيجة التصنيف")
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        
        cmap = ListedColormap(['red', 'green', 'blue'])
        im = ax2.imshow(st.session_state['display_class'], cmap=cmap, alpha=0.8)
        ax2.axis('off')
        
        patches = [
            mpatches.Patch(color='red', label='🏙️ Urban (عمراني)'),
            mpatches.Patch(color='green', label='🌾 Agricultural (زراعي)'),
            mpatches.Patch(color='blue', label='💧 Water (مائي)')
        ]
        ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        st.pyplot(fig2)
        plt.close(fig2)
    
    # إحصائيات
    st.markdown("---")
    st.subheader("📈 إحصائيات التصنيف")
    
    unique, counts = np.unique(st.session_state['classification'], return_counts=True)
    total_pixels = np.sum(counts)
    
    stats_data = []
    class_names = ['Urban', 'Agricultural', 'Water']
    class_icons = ['🏙️', '🌾', '💧']
    
    for cls, count in zip(unique, counts):
        if int(cls) < 3:
            percentage = count / total_pixels * 100
            stats_data.append({
                'الفئة': f"{class_icons[int(cls)]} {class_names[int(cls)]}",
                'عدد البكسلات': f"{count:,}",
                'النسبة': f"{percentage:.2f}%"
            })
    
    st.table(pd.DataFrame(stats_data))
    
    # رسم بياني
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    values = [float(s['النسبة'].replace('%', '')) for s in stats_data]
    labels = [s['الفئة'] for s in stats_data]
    colors = ['red', 'green', 'blue']
    
    bars = ax3.bar(labels, values, color=colors)
    ax3.set_ylabel('النسبة المئوية')
    ax3.set_title('توزيع الفئات')
    
    for bar, v in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{v:.1f}%', ha='center')
    
    st.pyplot(fig3)
    plt.close(fig3)
    
    # تنزيل النتيجة
    st.markdown("---")
    st.subheader("📥 تنزيل النتيجة")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_output:
        output_path = tmp_output.name
    
    profile = st.session_state['profile']
    profile.update(
        count=1,
        dtype='uint8',
        compress='lzw',
        bigtiff='YES'
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(st.session_state['classification'].astype('uint8'), 1)
    
    output_size = get_file_size_mb(output_path)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("حجم الملف الناتج", f"{output_size:.2f} MB")
    
    with col2:
        with open(output_path, 'rb') as f:
            st.download_button(
                label="📥 تنزيل الصورة المصنفة",
                data=f,
                file_name="classified_image.tif",
                mime="image/tiff",
                use_container_width=True
            )
    
    try:
        os.unlink(output_path)
    except:
        pass

# ===================================================
# تنظيف الملفات المؤقتة
# ===================================================
def cleanup():
    paths = ['red_path', 'green_path', 'blue_path']
    for path_key in paths:
        if path_key in st.session_state:
            try:
                os.unlink(st.session_state[path_key])
            except:
                pass

import atexit
atexit.register(cleanup)

# ===================================================
# تذييل الصفحة
# ===================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>© 2025 - مشروع تصنيف الصور الفضائية | نظام رفع ثلاث صور منفصلة</p>
        <p>يدعم رفع الباندات: Red, Green, Blue بشكل منفصل حتى 1GB لكل باند</p>
    </div>
    """,
    unsafe_allow_html=True
)