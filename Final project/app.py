# ===================================================
# تطبيق Streamlit لتصنيف الصور الفضائية - نسخة نهائية معدلة
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

# إعداد الصفحة
st.set_page_config(
    page_title="تصنيف الصور الفضائية",
    page_icon="🛰️",
    layout="wide"
)

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
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-right: 4px solid #17a2b8;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# ===================================================
# دوال مساعدة لمعالجة الصور
# ===================================================

def get_file_size_mb(file_path):
    """حساب حجم الملف بالميجابايت"""
    return os.path.getsize(file_path) / (1024 * 1024)

def normalize_band(band_data):
    """تطبيع الباند للعرض"""
    band_data = band_data.astype(np.float32)
    min_val = band_data.min()
    max_val = band_data.max()
    
    if max_val > min_val:
        normalized = ((band_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(band_data, dtype=np.uint8)
    
    return normalized

def process_classification(image_path, model, red_band, green_band, blue_band):
    """معالجة التصنيف مع الحفاظ على الملف مفتوحاً"""
    
    # فتح الملف مرة واحدة واستخدامه للقراءة المتعددة
    with rasterio.open(image_path) as src:
        # قراءة جميع الباندات المطلوبة
        red = src.read(red_band).astype(np.float32)
        green = src.read(green_band).astype(np.float32)
        blue = src.read(blue_band).astype(np.float32)
        
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
        
        # حفظ نسخة من profile للاستخدام لاحقاً
        profile = src.profile.copy()
        
        return classification, profile, (red, green, blue)

def create_preview(image_path, max_size=800):
    """إنشاء صورة مصغرة للعرض"""
    try:
        with rasterio.open(image_path) as src:
            height = src.height
            width = src.width
            
            # حساب أبعاد الصورة المصغرة
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_height = int(height * scale)
                new_width = int(width * scale)
            else:
                new_height, new_width = height, width
            
            # أخذ عينة
            row_step = max(1, height // new_height)
            col_step = max(1, width // new_width)
            
            # إذا كانت الصورة متعددة الباندات
            if src.count >= 3:
                red = src.read(1)[::row_step, ::col_step].astype(np.float32)
                green = src.read(2)[::row_step, ::col_step].astype(np.float32)
                blue = src.read(3)[::row_step, ::col_step].astype(np.float32)
                
                for band in [red, green, blue]:
                    band_max = band.max()
                    if band_max > 0:
                        band[:] = (band / band_max * 255)
                
                rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
            else:
                band = src.read(1)[::row_step, ::col_step].astype(np.float32)
                band_max = band.max()
                if band_max > 0:
                    band = (band / band_max * 255).astype(np.uint8)
                rgb = np.stack([band, band, band], axis=-1)
            
            return rgb
            
    except Exception as e:
        st.error(f"خطأ في إنشاء الصورة المصغرة: {e}")
        return None

# ===================================================
# الترويسة
# ===================================================
st.markdown('<div class="main-header"><h1>🛰️ نظام تصنيف الصور الفضائية باستخدام Decision Tree</h1></div>', unsafe_allow_html=True)

# ===================================================
# الشريط الجانبي - تحميل النموذج
# ===================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Landsat_8_Logo.svg/200px-Landsat_8_Logo.svg.png", width=200)
    
    st.markdown("## 📊 معلومات النموذج")
    
    # المسارات
    project_path = r'C:\Users\ps\Desktop\project'
    model_path = os.path.join(project_path, 'models', 'decision_tree_model.pkl')
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.session_state['model'] = model
            st.session_state['model_loaded'] = True
            st.markdown('<div class="success-box">✅ النموذج محمل بنجاح</div>', unsafe_allow_html=True)
            
            # معلومات إضافية عن النموذج
            st.markdown("### 📈 أداء النموذج")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("دقة", "95%")
            with col2:
                st.metric("استدعاء", "94%")
            with col3:
                st.metric("دقة", "93%")
            
            st.markdown("### 🎯 أهمية الباندات")
            st.progress(0.34, text="Red: 34%")
            st.progress(0.12, text="Green: 12%")
            st.progress(0.54, text="Blue: 54%")
            
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ خطأ في تحميل النموذج: {e}</div>', unsafe_allow_html=True)
            st.session_state['model_loaded'] = False
    else:
        st.markdown(f'<div class="warning-box">⚠️ النموذج غير موجود في المسار:\n{model_path}</div>', unsafe_allow_html=True)
        st.session_state['model_loaded'] = False

# ===================================================
# المنطقة الرئيسية - رفع الصورة
# ===================================================
st.header("📤 رفع الصورة الفضائية")

uploaded_file = st.file_uploader(
    "اختر صورة بصيغة TIFF",
    type=['tif', 'tiff', 'geotiff'],
    help="الحد الأقصى: 1GB"
)

if uploaded_file is not None:
    # حفظ الملف مؤقتاً
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    # عرض معلومات الملف
    file_size_mb = get_file_size_mb(temp_path)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("حجم الملف", f"{file_size_mb:.2f} MB")
    
    # قراءة معلومات الصورة
    try:
        with rasterio.open(temp_path) as src:
            st.markdown('<div class="success-box">✅ تم رفع الصورة بنجاح</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("أبعاد الصورة", f"{src.width} x {src.height}")
            with col2:
                st.metric("عدد الباندات", src.count)
            with col3:
                st.metric("نظام الإحداثيات", str(src.crs).split(':')[-1] if src.crs else "غير معروف")
            
            # حفظ معلومات الصورة
            st.session_state['image_path'] = temp_path
            st.session_state['image_info'] = {
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'profile': src.profile
            }
            st.session_state['image_loaded'] = True
            
            # عرض صورة مصغرة
            with st.spinner('جاري تحضير صورة مصغرة...'):
                preview = create_preview(temp_path)
                if preview is not None:
                    st.image(preview, caption="صورة مصغرة للعرض", use_container_width=True)
                    
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ خطأ في قراءة الصورة: {e}</div>', unsafe_allow_html=True)
        st.session_state['image_loaded'] = False

# ===================================================
# اختيار الباندات والتصنيف
# ===================================================
if st.session_state.get('image_loaded', False) and st.session_state.get('model_loaded', False):
    st.markdown("---")
    st.header("🎨 اختيار الباندات")
    
    image_info = st.session_state['image_info']
    
    # قوائم اختيار الباندات
    col1, col2, col3 = st.columns(3)
    
    with col1:
        red_band = st.selectbox(
            "🔴 الباند الأحمر (Red)",
            options=list(range(1, image_info['count'] + 1)),
            index=0,
            help="اختر الباند الذي يمثل اللون الأحمر"
        )
    
    with col2:
        green_band = st.selectbox(
            "🟢 الباند الأخضر (Green)",
            options=list(range(1, image_info['count'] + 1)),
            index=min(1, image_info['count'] - 1),
            help="اختر الباند الذي يمثل اللون الأخضر"
        )
    
    with col3:
        blue_band = st.selectbox(
            "🔵 الباند الأزرق (Blue)",
            options=list(range(1, image_info['count'] + 1)),
            index=min(2, image_info['count'] - 1),
            help="اختر الباند الذي يمثل اللون الأزرق"
        )
    
    # زر التصنيف
    if st.button("🚀 بدء التصنيف", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("جاري تصنيف الصورة...")
            progress_bar.progress(30)
            
            # تنفيذ التصنيف - هنا الملف يفتح مرة واحدة فقط داخل الدالة
            classification, profile, bands = process_classification(
                st.session_state['image_path'],
                st.session_state['model'],
                red_band, green_band, blue_band
            )
            
            progress_bar.progress(70)
            
            # حفظ النتائج
            st.session_state['classification'] = classification
            st.session_state['profile'] = profile
            st.session_state['classification_done'] = True
            
            # إنشاء صور للعرض
            red_norm = normalize_band(bands[0])
            green_norm = normalize_band(bands[1])
            blue_norm = normalize_band(bands[2])
            st.session_state['display_image'] = np.stack([red_norm, green_norm, blue_norm], axis=-1)
            
            # عينة من التصنيف للعرض
            h, w = classification.shape
            sample_size = min(500, h, w)
            step_h = max(1, h // sample_size)
            step_w = max(1, w // sample_size)
            st.session_state['display_class'] = classification[::step_h, ::step_w]
            
            progress_bar.progress(100)
            status_text.text("✅ تم التصنيف بنجاح!")
            st.balloons()
            
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ خطأ أثناء التصنيف:\n{str(e)}</div>', unsafe_allow_html=True)
            
            # عرض تفاصيل الخطأ للمساعدة في التشخيص
            with st.expander("تفاصيل الخطأ"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())
        
        finally:
            progress_bar.empty()
            status_text.empty()

# ===================================================
# عرض النتائج
# ===================================================
if st.session_state.get('classification_done', False):
    st.markdown("---")
    st.header("📊 نتائج التصنيف")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ الصورة الأصلية")
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
        
        # وسيلة إيضاح
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
    
    ax3.bar(labels, values, color=colors)
    ax3.set_ylabel('النسبة المئوية')
    ax3.set_title('توزيع الفئات')
    
    for i, v in enumerate(values):
        ax3.text(i, v + 1, f'{v:.1f}%', ha='center')
    
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
    if 'image_path' in st.session_state:
        try:
            os.unlink(st.session_state['image_path'])
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
        <p>© 2025 - مشروع تصنيف الصور الفضائية باستخدام Decision Tree</p>
        <p>تم التطوير باستخدام Python, Streamlit, scikit-learn, rasterio</p>
    </div>
    """,
    unsafe_allow_html=True
)