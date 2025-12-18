# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from visualizations import *

# Custom CSS untuk tampilan yang lebih profesional
st.set_page_config(
    page_title="Analisis & Prediksi Konsumsi Kafein",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #f8f9fa;
    }
    
    .block-container {
        background: white;
        border-radius: 10px;
        padding: 2rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1a237e;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        color: #2c3e50;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    }
    
    .metric-title {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #2c3e50;
    }
    
    .metric-change {
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        display: inline-block;
    }
    
    .positive-change {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .negative-change {
        background: #ffebee;
        color: #c62828;
    }
    
    .neutral-change {
        background: #e3f2fd;
        color: #1565c0;
    }
    
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        margin-bottom: 1.5rem;
    }
    
    .feature-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.2s ease;
    }
    
    .feature-box:hover {
        background: #e9ecef;
        transform: translateX(2px);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Memuat data cleaning"""
    try:
        if os.path.exists("Datacleaning.csv"):
            df = pd.read_csv("Datacleaning.csv")
            return df
        else:
            st.warning("File Datacleaning.csv tidak ditemukan!")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_original_data():
    """Memuat data asli dari notebook"""
    try:
        if os.path.exists("toba_caffeine_survey.xlsx"):
            df = pd.read_excel("toba_caffeine_survey.xlsx")
            return df
        else:
            st.warning("File toba_caffeine_survey.xlsx tidak ditemukan!")
            return None
    except Exception as e:
        st.error(f"Error loading original data: {e}")
        return None

@st.cache_resource
def load_model_results():
    """Memuat hasil model dari file pickle"""
    try:
        if os.path.exists("model_results.pkl"):
            with open("model_results.pkl", "rb") as f:
                results = pickle.load(f)
            
            # Pastikan semua komponen penting ada
            required_keys = ['models', 'data', 'metrics', 'feature_importance', 
                           'mappings', 'feature_names', 'df_original']
            
            if all(key in results for key in required_keys):
                st.success("Model berhasil dimuat dari model_results.pkl")
                return results
            else:
                st.warning("Format model_results.pkl tidak lengkap")
                return None
        else:
            st.warning("File model_results.pkl tidak ditemukan!")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_input_form(df_original, model_info, model_type):
    """Membuat form input user sesuai TOP feature importance tiap model"""
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<h3 class="sub-header">Input Data untuk Prediksi {model_type}</h3>',
            unsafe_allow_html=True
        )

        input_data = {}

        # ============================================
        # FEATURE SESUAI FEATURE IMPORTANCE PER MODEL
        # ============================================
        
        if model_type == "Kualitas Tidur":
            # Berdasarkan feature importance regresi
            important_features = [
                'digestion_issues_from_caffeine',
                'physical_side_effects_impact',
                'stress_effect_caffeine',
                'concentration_without_caffeine',
                'caffeine_before_sleep',
                'mood_drop_without_caffeine',
                'confidence_boost_from_caffeine',
                'caffeine_reason',
                'work_duration_increase',
                'favorite_caffeine_drink'
            ]

        elif model_type == "Pengaruh pada Mood":
            # Berdasarkan feature importance klasifikasi mood
            important_features = [
                'caffeine_time',
                'favorite_caffeine_drink',
                'feeling_without_caffeine',
                'age',
                'caffeine_before_sleep',
                'mood_drop_without_caffeine',
                'caffeine_intensity',
                'gender',
                'confidence_boost_from_caffeine',
                'digestion_issues_from_caffeine'
            ]

        else:  # Pengaruh pada Fokus
            # Berdasarkan feature importance klasifikasi fokus
            important_features = [
                'stress_effect_caffeine',
                'caffeine_reason',
                'concentration_without_caffeine',
                'digestion_issues_from_caffeine',
                'physical_side_effects_impact',
                'work_duration_increase',
                'confidence_boost_from_caffeine',
                'mood_drop_without_caffeine',
                'caffeine_before_sleep',
                'caffeine_time'
            ]

        # Ambil hanya fitur yang benar-benar ada di dataset
        available_features = [f for f in important_features if f in df_original.columns]

        cols = st.columns(2)
        col_idx = 0

        for feature_name in available_features:
            with cols[col_idx % 2]:
                st.markdown('<div class="feature-box">', unsafe_allow_html=True)

                # Label yang lebih deskriptif
                feature_labels = {
                    # Fitur untuk semua model
                    'digestion_issues_from_caffeine': 'Seberapa sering mengalami masalah pencernaan setelah konsumsi kafein?',
                    'physical_side_effects_impact': 'Seberapa besar dampak efek samping fisik dari kafein?',
                    'stress_effect_caffeine': 'Bagaimana pengaruh kafein terhadap tingkat stres Anda?',
                    'concentration_without_caffeine': 'Bagaimana tingkat konsentrasi Anda tanpa konsumsi kafein?',
                    'caffeine_before_sleep': 'Seberapa sering Anda mengonsumsi kafein sebelum tidur?',
                    'mood_drop_without_caffeine': 'Seberapa besar penurunan mood yang Anda alami tanpa kafein?',
                    'confidence_boost_from_caffeine': 'Seberapa besar peningkatan percaya diri yang Anda dapat setelah konsumsi kafein?',
                    'caffeine_reason': 'Apa alasan utama Anda mengonsumsi kafein?',
                    'work_duration_increase': 'Seberapa besar peningkatan durasi kerja yang Anda alami setelah mengonsumsi kafein?',
                    'favorite_caffeine_drink': 'Apa minuman kafein favorit Anda?',
                    
                    # Fitur khusus mood
                    'caffeine_time': 'Kapan waktu favorit Anda mengonsumsi kafein?',
                    'feeling_without_caffeine': 'Bagaimana perasaan Anda tanpa mengonsumsi kafein?',
                    'caffeine_intensity': 'Seberapa intens konsumsi kafein Anda?',
                    'age': 'Berapa usia Anda?',
                    'gender': 'Jenis kelamin Anda?'
                }

                label = feature_labels.get(
                    feature_name,
                    feature_name.replace('_', ' ').title()
                )

                # ==============================
                # NUMERIK
                # ==============================
                if feature_name in df_original.select_dtypes(include=[np.number]).columns:
                    min_val = float(df_original[feature_name].min())
                    max_val = float(df_original[feature_name].max())
                    default_val = float(df_original[feature_name].median())

                    if feature_name == 'age':
                        min_val, max_val, default_val = 15, 65, 25

                    input_data[feature_name] = st.slider(
                        label,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        key=f"{model_type}_{feature_name}"
                    )

                # ==============================
                # KATEGORIKAL - PERBAIKAN KHUSUS untuk favorite_caffeine_drink
                # ==============================
                else:
                    # Dapatkan unique values, konversi ke string
                    unique_vals = (
                        df_original[feature_name]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    
                    # ================================================
                    # FILTER KHUSUS UNTUK favorite_caffeine_drink
                    # ================================================
                    if feature_name == 'favorite_caffeine_drink':
                        # Hapus opsi "Tidak ada" atau yang sejenis
                        filtered_vals = []
                        for val in unique_vals:
                            val_lower = val.lower()
                            # Filter out "tidak ada" dan variasi lainnya
                            if ('tidak ada' not in val_lower and 
                                'tidak memilih' not in val_lower and
                                'tidak tahu' not in val_lower and
                                'other' not in val_lower and
                                'lainnya' not in val_lower):
                                filtered_vals.append(val)
                        
                        # Jika ada hasil filter, gunakan
                        if filtered_vals:
                            unique_vals = sorted(set(filtered_vals))
                    
                    # ================================================
                    # FILTER KHUSUS UNTUK caffeine_reason
                    # ================================================
                    elif feature_name == 'caffeine_reason':
                        # Hanya tampilkan pilihan yang diinginkan
                        allowed_options = [
                            'Kebiasaan',
                            'Menambah Fokus',
                            'Menemani Aktivitas Sosial',
                            'Menghindari Kantuk',
                            'Tidak Mengonsumsi'
                        ]
                        
                        # Filter untuk hanya mempertahankan pilihan yang diinginkan
                        filtered_vals = []
                        for val in unique_vals:
                            # Cari kecocokan dengan pilihan yang diinginkan
                            for allowed in allowed_options:
                                if allowed.lower() in val.lower():
                                    # Gunakan versi standar dari allowed options
                                    filtered_vals.append(allowed)
                                    break
                        
                        # Jika ada hasil filter, gunakan
                        if filtered_vals:
                            unique_vals = sorted(set(filtered_vals))
                        else:
                            # Jika tidak ada kecocokan, gunakan allowed options langsung
                            unique_vals = allowed_options
                    
                    # ================================================
                    # FILTER UMUM untuk fitur kategorikal lainnya
                    # ================================================
                    else:
                        # Filter umum: hapus opsi yang tidak diinginkan
                        unique_vals = [val for val in unique_vals 
                                     if 'tidak memilih' not in val.lower() 
                                     and 'tidak tahu' not in val.lower()
                                     and 'lainnya' not in val.lower()
                                     and 'other' not in val.lower()
                                     and 'gatau' not in val.lower()
                                     and 'fomo' not in val.lower()
                                     and 'karna haus' not in val.lower()
                                     and 'karena suka aja' not in val.lower()
                                     and 'menambah zat besi' not in val.lower()]
                    
                    # Urutkan dan pastikan tidak kosong
                    unique_vals = sorted(set(unique_vals))
                    
                    if len(unique_vals) > 0:
                        input_data[feature_name] = st.selectbox(
                            label,
                            unique_vals,
                            key=f"{model_type}_{feature_name}"
                        )
                    else:
                        # Jika setelah filter kosong, gunakan semua values
                        unique_vals = sorted(set(df_original[feature_name].dropna().astype(str).unique().tolist()))
                        input_data[feature_name] = st.selectbox(
                            label,
                            unique_vals,
                            key=f"{model_type}_{feature_name}"
                        )

                st.markdown('</div>', unsafe_allow_html=True)

            col_idx += 1

        st.markdown('</div>', unsafe_allow_html=True)

    return input_data

def encode_input_data(input_data, categorical_mapping, feature_names, model_type):
    """Mengonversi input user ke format one-hot encoding sesuai dengan model"""
    
    input_df = pd.DataFrame([input_data])
    
    # Handling khusus untuk kolom usia
    if 'age' in input_df.columns:
        input_df['age'] = pd.to_numeric(input_df['age'], errors='coerce')
    
    # One-hot encoding untuk kolom kategorikal
    for col in input_df.select_dtypes(include=['object']).columns:
        if col in categorical_mapping:
            unique_vals = categorical_mapping[col]
            for val in unique_vals:
                col_name = f"{col}_{val}"
                input_df[col_name] = (input_df[col].astype(str) == str(val)).astype(int)
    
    # Hapus kolom kategorikal asli yang sudah diencode
    categorical_cols = list(categorical_mapping.keys())
    input_df = input_df.drop(columns=[col for col in categorical_cols if col in input_df.columns], errors='ignore')
    
    # Tambahkan kolom yang hilang dengan nilai 0
    for feature in feature_names:
        if feature not in input_df.columns:
            # Jika fitur numerik, isi dengan median atau 0
            if 'age' in feature.lower() or 'sleep' in feature.lower():
                input_df[feature] = 0
            else:
                input_df[feature] = 0
    
    # Urutkan kolom sesuai dengan feature_names
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    # Skalakan fitur numerik jika diperlukan (asumsi scaling seperti di notebook)
    if 'age' in input_df.columns:
        # Normalisasi manual (sesuaikan dengan preprocessing di notebook)
        age_mean = 22.16  # Rata-rata dari data
        age_std = 4.81    # Std dev dari data
        input_df['age'] = (input_df['age'] - age_mean) / age_std
    
    return input_df

def show_visualizations_page(df):
    """Menampilkan halaman visualisasi menggunakan functions dari visualizations.py"""
    
    st.markdown('<h1 class="main-header">Visualisasi Data Konsumsi Kafein</h1>', unsafe_allow_html=True)
    
    if df is None:
        st.warning("Data tidak tersedia.")
        return
    
    # Pilihan kategori visualisasi
    categories = [
        "Semua Visualisasi",
        "Demografi",
        "Konsumsi Kafein",
        "Kualitas Tidur",
        "Mood & Perasaan",
        "Fokus & Konsentrasi",
        "Korelasi & Statistik"
    ]
    
    selected_category = st.selectbox("Pilih Kategori Visualisasi", categories)
    
    if selected_category == "Semua Visualisasi":
        st.markdown('<div class="sub-header">Semua Visualisasi</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribusi Usia
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Distribusi Usia**")
            fig = plot_age_distribution(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Distribusi Gender
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Distribusi Gender**")
            fig = plot_gender_distribution(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Minuman Kafein Favorit
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Minuman Kafein Favorit**")
            fig = plot_favorite_drink(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Kelompok Usia
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Distribusi Kelompok Usia**")
            fig = plot_age_group_distribution(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Kualitas Tidur
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Distribusi Kualitas Tidur**")
            fig = plot_sleep_quality(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Fokus dengan Kafein
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Pengaruh pada Fokus**")
            fig = plot_focus_boost_caffeine(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_category == "Demografi":
        st.markdown('<div class="sub-header">Visualisasi Demografi</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Distribusi Usia**")
            fig = plot_age_distribution(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Distribusi Gender**")
            fig = plot_gender_distribution(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Kelompok Usia
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("**Distribusi Kelompok Usia**")
        fig = plot_age_group_distribution(df)
        if fig:
            st.pyplot(fig)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_category == "Konsumsi Kafein":
        st.markdown('<div class="sub-header">Visualisasi Konsumsi Kafein</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Minuman Kafein Favorit**")
            fig = plot_favorite_drink(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Waktu Konsumsi Kafein**")
            fig = plot_caffeine_time(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Konsumsi Sebelum Tidur**")
            fig = plot_caffeine_before_sleep(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Frekuensi Konsumsi**")
            fig = plot_caffeine_frequency(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_category == "Kualitas Tidur":
        st.markdown('<div class="sub-header">Visualisasi Kualitas Tidur</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("**Distribusi Kualitas Tidur**")
        fig = plot_sleep_quality(df)
        if fig:
            st.pyplot(fig)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Statistik kualitas tidur
        if 'sleep_quality' in df.columns:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rata-rata", f"{df['sleep_quality'].mean():.2f}")
            with col2:
                st.metric("Median", f"{df['sleep_quality'].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df['sleep_quality'].std():.2f}")
            with col4:
                st.metric("Range", f"{df['sleep_quality'].min():.0f} - {df['sleep_quality'].max():.0f}")
    
    elif selected_category == "Mood & Perasaan":
        st.markdown('<div class="sub-header">Visualisasi Mood & Perasaan</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Perasaan Tanpa Kafein**")
            fig = plot_feeling_without_caffeine(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Mood Tanpa Kafein**")
            fig = plot_mood_without_caffeine(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Efek pada stres
        if 'stress_effect_caffeine' in df.columns:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Efek Kafein pada Stres**")
            fig = plot_stress_effect(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_category == "Fokus & Konsentrasi":
        st.markdown('<div class="sub-header">Visualisasi Fokus & Konsentrasi</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Pengaruh pada Fokus**")
            fig = plot_focus_boost_caffeine(df)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if 'concentration_without_caffeine' in df.columns:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("**Konsentrasi Tanpa Kafein**")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                conc_counts = df['concentration_without_caffeine'].value_counts()
                
                colors = plt.cm.Set3(np.arange(len(conc_counts)))
                bars = ax.bar(conc_counts.index, conc_counts.values, color=colors)
                
                ax.set_xlabel('Tingkat Konsentrasi')
                ax.set_ylabel('Jumlah Responden')
                ax.set_title('Distribusi Konsentrasi Tanpa Kafein', fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom')
                
                st.pyplot(fig)
                plt.close()
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_category == "Korelasi & Statistik":
        st.markdown('<div class="sub-header">Analisis Korelasi & Statistik</div>', unsafe_allow_html=True)
        
        # Matriks korelasi
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("**Matriks Korelasi**")
        fig = plot_correlation_matrix(df)
        if fig:
            st.pyplot(fig)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Masalah pencernaan
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("**Masalah Pencernaan dari Kafein**")
        fig = plot_digestion_issues(df)
        if fig:
            st.pyplot(fig)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tampilkan visualisasi intensitas kafein jika ada
    if 'caffeine_intensity' in df.columns:
        st.markdown('<div class="sub-header">Intensitas Konsumsi Kafein</div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        intensity_counts = df['caffeine_intensity'].value_counts()
        colors = ['#4CAF50', '#FFC107', '#F44336', '#9E9E9E']
        
        # Urutkan berdasarkan tingkat intensitas
        order = ['Rendah', 'Sedang', 'Tinggi', 'Tidak Diketahui']
        intensity_counts = intensity_counts.reindex([o for o in order if o in intensity_counts.index])
        
        bars = ax.bar(intensity_counts.index, intensity_counts.values, 
                     color=colors[:len(intensity_counts)])
        
        ax.set_title('Distribusi Intensitas Konsumsi Kafein', fontsize=14, fontweight='bold')
        ax.set_xlabel('Intensitas Konsumsi')
        ax.set_ylabel('Jumlah Responden')
        
        # Tambahkan nilai di atas bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()

def show_prediction_page(model_results):
    """Menampilkan halaman prediksi"""
    st.markdown('<h1 class="main-header">Prediksi</h1>', unsafe_allow_html=True)
    
    if model_results is None:
        st.error("Model tidak ditemukan. Pastikan file model_results.pkl ada di folder yang sama.")
        return
    
    # Ambil data yang diperlukan
    df_original = model_results.get('df_original')
    categorical_mapping = model_results.get('mappings', {}).get('categorical_mapping', {})
    feature_names = model_results.get('feature_names', [])
    models = model_results.get('models', {})
    le_focus = model_results.get('mappings', {}).get('focus_encoder')
    
    if df_original is None:
        st.error("Data tidak tersedia untuk input form.")
        return
    
    # Pilih jenis prediksi
    model_type = st.selectbox(
        "Pilih Jenis Prediksi",
        ["Kualitas Tidur", "Pengaruh pada Mood", "Pengaruh pada Fokus"]
    )
    
    # Buat form input berdasarkan jenis prediksi
    input_data = create_input_form(df_original, model_results, model_type)
    
    if st.button("Lakukan Prediksi", type="primary"):
        if input_data:
            try:
                # Encode input data
                input_encoded = encode_input_data(
                    input_data,
                    categorical_mapping,
                    feature_names,
                    model_type
                )
                
                with st.spinner("Menganalisis..."):
                    if model_type == "Kualitas Tidur":
                        model = models.get('regression')
                        if model:
                            prediction = model.predict(input_encoded)[0]
                            
                            # Batasi prediksi antara 1-5
                            prediction = max(1.0, min(5.0, prediction))
                            
                            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                            st.markdown(f"## Prediksi Kualitas Tidur: {prediction:.2f}/5")
                            
                            # Visualisasi skor
                            fig, ax = plt.subplots(figsize=(10, 3))
                            ax.barh([0], [5], color='lightgray', alpha=0.3, height=0.3)
                            ax.barh([0], [prediction], color='skyblue', height=0.3)
                            ax.set_xlim(0, 5)
                            ax.set_xticks([1, 2, 3, 4, 5])
                            ax.set_yticks([])
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            ax.set_xlabel('Skala Kualitas Tidur (1 = Sangat Buruk, 5 = Sangat Baik)')
                            
                            # Warna berdasarkan skor
                            if prediction >= 4.0:
                                ax.barh([0], [prediction], color='#4CAF50', height=0.3)
                            elif prediction >= 3.0:
                                ax.barh([0], [prediction], color='#FFC107', height=0.3)
                            else:
                                ax.barh([0], [prediction], color='#F44336', height=0.3)
                            
                            st.pyplot(fig)
                            
                            # Interpretasi berdasarkan skala 1-5
                            if prediction >= 4.0:
                                st.success("**Baik** - Kualitas tidur Anda baik, kafein tidak mengganggu tidur Anda")
                            elif prediction >= 3.0:
                                st.warning("**Cukup** - Kualitas tidur cukup, pertimbangkan mengatur konsumsi kafein")
                            else:
                                st.error("**Buruk** - Kualitas tidur terganggu, kurangi konsumsi kafein terutama sebelum tidur")
                            
                            # Analisis faktor berdasarkan top features regresi
                            st.markdown("### Analisis Faktor:")
                            
                            # 1. Masalah pencernaan
                            if 'digestion_issues_from_caffeine' in input_data:
                                digestion = str(input_data['digestion_issues_from_caffeine']).lower()
                                if any(word in digestion for word in ['sering', 'selalu', 'parah', 'banyak', 'sangat sering']):
                                    st.info("⚠ **Masalah pencernaan**: Konsumsi kafein yang menyebabkan masalah pencernaan dapat mengganggu kualitas tidur")
                                elif any(word in digestion for word in ['jarang', 'tidak pernah']):
                                    st.info("✅ **Masalah pencernaan**: Tidak ada masalah pencernaan berarti kafein tidak mengganggu sistem pencernaan Anda")
                            
                            # 2. Efek samping fisik
                            if 'physical_side_effects_impact' in input_data:
                                physical = input_data['physical_side_effects_impact']
                                if isinstance(physical, str):
                                    physical_str = physical.lower()
                                    if any(word in physical_str for word in ['sering', 'selalu', 'mengganggu']):
                                        st.info("⚠ **Efek samping fisik**: Efek samping fisik yang signifikan dapat mengurangi kualitas tidur")
                            
                            # 3. Kafein sebelum tidur
                            if 'caffeine_before_sleep' in input_data:
                                before_sleep = str(input_data['caffeine_before_sleep']).lower()
                                if any(word in before_sleep for word in ['sering', 'selalu', 'seringkali']):
                                    st.info("⚠ **Konsumsi sebelum tidur**: Kafein sebelum tidur dapat mengganggu pola tidur dan mengurangi kualitas tidur")
                                elif 'tidak pernah' in before_sleep:
                                    st.info("✅ **Konsumsi sebelum tidur**: Tidak mengonsumsi kafein sebelum tidur membantu menjaga kualitas tidur")
                            
                            # 4. Stres
                            if 'stress_effect_caffeine' in input_data:
                                stress = str(input_data['stress_effect_caffeine']).lower()
                                if 'mengurangi' in stress:
                                    st.info("✅ **Efek pada stres**: Kafein membantu mengurangi stres, yang dapat meningkatkan kualitas tidur")
                                elif 'meningkatkan' in stress:
                                    st.info("⚠ **Efek pada stres**: Kafein meningkatkan stres, yang dapat mengganggu kualitas tidur")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error("Model regresi tidak ditemukan")
                    
                    elif model_type == "Pengaruh pada Mood":
                        model = models.get('mood_classifier')
                        if model:
                            prediction = model.predict(input_encoded)[0]
                            proba = model.predict_proba(input_encoded)[0]
                            
                            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                            st.markdown("## Hasil Prediksi Pengaruh pada Mood")
                            
                            mood_labels = ["Buruk", "Netral", "Baik"]
                            mood_descriptions = [
                                "Kafein cenderung memperburuk mood Anda",
                                "Kafein tidak berpengaruh signifikan pada mood Anda",
                                "Kafein cenderung meningkatkan mood Anda"
                            ]
                            colors = ['#FF6B6B', '#FFD166', '#06D6A0']
                            
                            cols = st.columns(3)
                            for i, (label, color, desc) in enumerate(zip(mood_labels, colors, mood_descriptions)):
                                with cols[i]:
                                    st.markdown(f'<div style="background: {color}; padding: 1rem; border-radius: 8px; color: white; text-align: center; margin: 0.5rem;">', unsafe_allow_html=True)
                                    st.markdown(f"**{label}**", unsafe_allow_html=True)
                                    st.markdown(f"**{proba[i]*100:.1f}%**", unsafe_allow_html=True)
                                    if i == prediction:
                                        st.markdown("**✓ Diprediksi**", unsafe_allow_html=True)
                                        st.caption(desc)
                                    st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Visualisasi probabilitas
                            fig, ax = plt.subplots(figsize=(10, 4))
                            bars = ax.bar(mood_labels, proba, color=colors)
                            ax.set_ylabel('Probabilitas')
                            ax.set_title('Probabilitas Pengaruh Kafein pada Mood', fontweight='bold')
                            ax.set_ylim(0, 1)
                            
                            # Tambahkan nilai di atas bar
                            for bar, prob in zip(bars, proba):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                            
                            # Highlight prediksi
                            bars[prediction].set_edgecolor('black')
                            bars[prediction].set_linewidth(2)
                            
                            st.pyplot(fig)
                            
                            # Analisis faktor berdasarkan top features mood
                            st.markdown("### Analisis Faktor:")
                            
                            # 1. Waktu konsumsi
                            if 'caffeine_time' in input_data:
                                time = str(input_data['caffeine_time']).lower()
                                if 'pagi' in time:
                                    st.info("✅ **Waktu konsumsi**: Konsumsi di pagi hari cenderung meningkatkan mood")
                                elif 'malam' in time or 'sore' in time:
                                    st.info("⚠ **Waktu konsumsi**: Konsumsi di malam/sore hari dapat mempengaruhi mood negatif")
                            
                            # 2. Minuman favorit
                            if 'favorite_caffeine_drink' in input_data:
                                drink = str(input_data['favorite_caffeine_drink'])
                                st.info(f"🍵 **Minuman favorit**: {drink} - Pilihan minuman mempengaruhi pengalaman konsumsi kafein")
                            
                            # 3. Perasaan tanpa kafein
                            if 'feeling_without_caffeine' in input_data:
                                feeling = str(input_data['feeling_without_caffeine']).lower()
                                if any(word in feeling for word in ['gelisah', 'buruk', 'cemas', 'tidak tenang', 'tidak fokus']):
                                    st.info("⚠ **Perasaan tanpa kafein**: Ketergantungan pada kafein dapat mempengaruhi mood negatif")
                                elif any(word in feeling for word in ['biasa', 'normal', 'tidak ada perubahan']):
                                    st.info("✅ **Perasaan tanpa kafein**: Tidak bergantung pada kafein membantu menjaga mood stabil")
                            
                            # 4. Penurunan mood tanpa kafein
                            if 'mood_drop_without_caffeine' in input_data:
                                mood_drop = str(input_data['mood_drop_without_caffeine']).lower()
                                if any(word in mood_drop for word in ['besar', 'signifikan', 'sangat', 'parah']):
                                    st.info("⚠ **Penurunan mood**: Ketergantungan tinggi pada kafein untuk menjaga mood")
                                elif 'tidak ada' in mood_drop or 'sedikit' in mood_drop:
                                    st.info("✅ **Penurunan mood**: Tidak bergantung pada kafein untuk mood")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error("Model mood classifier tidak ditemukan")
                    
                    else:  # Pengaruh pada Fokus
                        model = models.get('focus_classifier')
                        if model and le_focus:
                            prediction = model.predict(input_encoded)[0]
                            prediction_label = le_focus.inverse_transform([prediction])[0]
                            proba = model.predict_proba(input_encoded)[0]
                            
                            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                            st.markdown(f"## Hasil Prediksi Pengaruh pada Fokus")
                            st.markdown(f"### {prediction_label}")
                            
                            # Dapatkan semua label dan urutkan berdasarkan probabilitas
                            all_labels = le_focus.classes_
                            sorted_indices = np.argsort(proba)[::-1]
                            top_labels = all_labels[sorted_indices][:5]
                            top_probs = proba[sorted_indices][:5]
                            
                            # Tampilkan top 5 probabilitas
                            for label, prob in zip(top_labels, top_probs):
                                st.markdown(f"**{label}**: {prob:.1%}")
                                st.progress(float(prob))
                            
                            # Visualisasi
                            fig, ax = plt.subplots(figsize=(10, 5))
                            y_pos = np.arange(len(top_labels))
                            
                            bars = ax.barh(y_pos, top_probs, color=plt.cm.viridis(np.linspace(0, 1, len(top_labels))))
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(top_labels)
                            ax.set_xlabel('Probabilitas')
                            ax.set_title('Top 5 Prediksi Pengaruh pada Fokus', fontweight='bold')
                            ax.set_xlim(0, 1)
                            
                            # Tambahkan nilai di ujung bar
                            for bar, prob in zip(bars, top_probs):
                                width = bar.get_width()
                                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                       f'{prob:.1%}', va='center')
                            
                            # Highlight prediksi utama
                            pred_idx = np.where(top_labels == prediction_label)[0]
                            if len(pred_idx) > 0:
                                bars[pred_idx[0]].set_edgecolor('red')
                                bars[pred_idx[0]].set_linewidth(3)
                            
                            st.pyplot(fig)
                            
                            # Analisis faktor berdasarkan top features fokus
                            st.markdown("### Analisis Faktor:")
                            
                            # 1. Efek pada stres
                            if 'stress_effect_caffeine' in input_data:
                                stress = str(input_data['stress_effect_caffeine']).lower()
                                if 'mengurangi' in stress:
                                    st.info("✅ **Efek pada stres**: Kafein membantu mengurangi stres, meningkatkan fokus")
                                elif 'meningkatkan' in stress:
                                    st.info("⚠ **Efek pada stres**: Kafein meningkatkan stres, mengurangi fokus")
                            
                            # 2. Alasan konsumsi
                            if 'caffeine_reason' in input_data:
                                reason = str(input_data['caffeine_reason']).lower()
                                if any(word in reason for word in ['fokus', 'konsentrasi', 'kerja', 'produktivitas']):
                                    st.info("✅ **Alasan konsumsi**: Tujuan konsumsi untuk fokus meningkatkan efektivitas")
                                elif any(word in reason for word in ['kebiasaan', 'rasa', 'teman']):
                                    st.info("ℹ️ **Alasan konsumsi**: Konsumsi untuk kebiasaan mungkin kurang efektif untuk fokus")
                            
                            # 3. Konsentrasi tanpa kafein
                            if 'concentration_without_caffeine' in input_data:
                                concentration = input_data['concentration_without_caffeine']
                                if isinstance(concentration, (int, float, np.integer, np.floating)):
                                    if concentration < 3:  # Asumsi skala 1-5
                                        st.info("✅ **Konsentrasi tanpa kafein**: Kafein dapat membantu meningkatkan konsentrasi Anda secara signifikan")
                                    else:
                                        st.info("ℹ️ **Konsentrasi tanpa kafein**: Anda sudah memiliki konsentrasi yang baik tanpa kafein")
                            
                            # 4. Peningkatan durasi kerja
                            if 'work_duration_increase' in input_data:
                                work_duration = input_data['work_duration_increase']
                                if isinstance(work_duration, (int, float, np.integer, np.floating)):
                                    if work_duration > 3:  # Asumsi skala 1-5
                                        st.info("✅ **Peningkatan produktivitas**: Kafein membantu meningkatkan durasi dan produktivitas kerja")
                                    elif work_duration < 2:
                                        st.info("⚠ **Peningkatan produktivitas**: Kafein mungkin kurang efektif untuk meningkatkan produktivitas Anda")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error("Model focus classifier tidak ditemukan")
            
            except Exception as e:
                st.error(f"Error dalam prediksi: {str(e)}")
                st.error(f"Detail error: {e.__class__.__name__}")
                import traceback
                st.code(traceback.format_exc())

def show_evaluation_page(model_results):
    """Menampilkan halaman evaluasi model"""
    st.markdown('<h1 class="main-header">Evaluasi Model</h1>', unsafe_allow_html=True)
    
    if model_results is None:
        st.error("Model tidak ditemukan.")
        return
    
    metrics = model_results.get('metrics', {})
    feature_importance = model_results.get('feature_importance', {})
    data = model_results.get('data', {})
    
    # Tampilkan metrik performa dalam card
    st.markdown('<h3 class="sub-header">Ringkasan Performa Model</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        reg_metrics = metrics.get('regression', {})
        r2 = reg_metrics.get('r2', 0)
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">R² Score Regresi</div>
            <div class="metric-value">{r2:.4f}</div>
            <div class="metric-change {"positive-change" if r2 > 0.7 else "neutral-change" if r2 > 0.5 else "negative-change"}">
                {"Baik" if r2 > 0.7 else "Cukup" if r2 > 0.5 else "Kurang"}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        rmse = reg_metrics.get('rmse', 0)
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">RMSE</div>
            <div class="metric-value">{rmse:.4f}</div>
            <div class="metric-change neutral-change">error</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        mood_metrics = metrics.get('mood_classification', {})
        mood_acc = mood_metrics.get('accuracy', 0)
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">Akurasi Mood</div>
            <div class="metric-value">{mood_acc:.4f}</div>
            <div class="metric-change {"positive-change" if mood_acc > 0.8 else "neutral-change" if mood_acc > 0.6 else "negative-change"}">
                {mood_acc*100:.1f}%
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        focus_metrics = metrics.get('focus_classification', {})
        focus_acc = focus_metrics.get('accuracy', 0)
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">Akurasi Fokus</div>
            <div class="metric-value">{focus_acc:.4f}</div>
            <div class="metric-change {"positive-change" if focus_acc > 0.8 else "neutral-change" if focus_acc > 0.6 else "negative-change"}">
                {focus_acc*100:.1f}%
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Tab untuk evaluasi detail
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Regresi", 
        "😊 Klasifikasi Mood", 
        "🎯 Klasifikasi Fokus",
        "📈 Feature Importance"
    ])
    
    with tab1:
        st.markdown("### Model Regresi: Prediksi Kualitas Tidur")
        
        # Tampilkan scatter plot actual vs predicted jika ada
        if 'y_test_reg' in data and 'y_pred_reg' in data:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(data['y_test_reg'], data['y_pred_reg'], alpha=0.5)
            ax.plot([data['y_test_reg'].min(), data['y_test_reg'].max()], 
                   [data['y_test_reg'].min(), data['y_test_reg'].max()], 
                   'r--', lw=2)
            ax.set_xlabel('Actual Sleep Quality')
            ax.set_ylabel('Predicted Sleep Quality')
            ax.set_title('Actual vs Predicted Sleep Quality')
            st.pyplot(fig)
        
        # Feature importance
        if 'regression' in feature_importance:
            st.markdown("#### Top 15 Feature Importance - Kualitas Tidur")
            top_features = feature_importance['regression'].head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(top_features['feature'][::-1], top_features['importance'][::-1], 
                          color='skyblue')
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 15 Feature Importance untuk Prediksi Kualitas Tidur', 
                        fontsize=14, fontweight='bold')
            
            # Tambahkan nilai di ujung bar - PERBAIKAN: menambahkan kurung tutup
            for i, (bar, imp) in enumerate(zip(bars, top_features['importance'][::-1])):
                ax.text(imp + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{imp:.4f}', va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tampilkan sebagai tabel
            st.dataframe(top_features.style.background_gradient(subset=['importance'], cmap='Blues'))
    
    with tab2:
        st.markdown("### Model Klasifikasi: Pengaruh pada Mood")
        
        # Confusion matrix
        if 'y_test_mood' in data and 'y_pred_mood' in data:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Confusion Matrix
                cm = confusion_matrix(data['y_test_mood'], data['y_pred_mood'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
                axes[0].set_xlabel('Predicted')
                axes[0].set_ylabel('Actual')
                axes[0].set_title('Confusion Matrix - Mood Classification')
                
                # Classification Report Heatmap
                report = classification_report(data['y_test_mood'], data['y_pred_mood'], 
                                             output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlOrRd', ax=axes[1])
                axes[1].set_title('Classification Report Heatmap')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Tampilkan classification report sebagai tabel
                st.markdown("#### Classification Report Detail")
                st.dataframe(report_df)
                
            except Exception as e:
                st.warning(f"Error dalam visualisasi: {str(e)}")
        
        # Feature importance untuk mood classification
        if 'mood_classification' in feature_importance:
            st.markdown("#### Feature Importance - Mood Classification")
            top_features = feature_importance['mood_classification'].head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_features['feature'][::-1], top_features['importance'][::-1], 
                          color='lightgreen')
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 10 Feature Importance untuk Klasifikasi Mood', 
                        fontsize=14, fontweight='bold')
            st.pyplot(fig)
    
    with tab3:
        st.markdown("### Model Klasifikasi: Pengaruh pada Fokus")
        
        # Confusion matrix untuk fokus
        if 'y_test_focus' in data and 'y_pred_focus' in data:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Confusion Matrix
                cm = confusion_matrix(data['y_test_focus'], data['y_pred_focus'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0])
                axes[0].set_xlabel('Predicted')
                axes[0].set_ylabel('Actual')
                axes[0].set_title('Confusion Matrix - Focus Classification')
                
                # Classification Report Heatmap
                report = classification_report(data['y_test_focus'], data['y_pred_focus'], 
                                             output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlOrBr', ax=axes[1])
                axes[1].set_title('Classification Report Heatmap')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Tampilkan classification report sebagai tabel
                st.markdown("#### Classification Report Detail")
                st.dataframe(report_df)
                
            except Exception as e:
                st.warning(f"Error dalam visualisasi: {str(e)}")
    
    with tab4:
        st.markdown("### Analisis Feature Importance Komparatif")
        
        # Gabungkan feature importance dari ketiga model
        all_features = {}
        
        for model_name in ['regression', 'mood_classification', 'focus_classification']:
            if model_name in feature_importance:
                df_imp = feature_importance[model_name].head(10)
                all_features[model_name] = df_imp
        
        # Buat visualisasi komparatif
        if len(all_features) > 0:
            fig, axes = plt.subplots(len(all_features), 1, figsize=(12, 4*len(all_features)))
            
            for idx, (model_name, df_imp) in enumerate(all_features.items()):
                if len(all_features) == 1:
                    ax = axes
                else:
                    ax = axes[idx]
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][idx % 3]
                bars = ax.barh(df_imp['feature'][::-1], df_imp['importance'][::-1], color=colors)
                ax.set_xlabel('Importance Score')
                ax.set_title(f'Top 10 Feature Importance - {model_name.replace("_", " ").title()}', 
                           fontweight='bold')
                
                # Tambahkan nilai di ujung bar
                for bar, imp in zip(bars, df_imp['importance'][::-1]):
                    ax.text(imp + 0.001, bar.get_y() + bar.get_height()/2, 
                           f'{imp:.4f}', va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Informasi tentang fitur-fitur penting
        st.markdown("#### Analisis Fitur Penting")
        
        st.info("""
        **Insights dari Feature Importance:**
        
        1. **Regresi (Kualitas Tidur):** 
           - Masalah pencernaan memiliki pengaruh tertinggi
           - Efek samping fisik juga signifikan
           - Kafein sebelum tidur mempengaruhi kualitas tidur
        
        2. **Klasifikasi Mood:**
           - Waktu konsumsi kafein paling berpengaruh
           - Jenis minuman kafein favorit penting
           - Perasaan tanpa kafein mempengaruhi mood
        
        3. **Klasifikasi Fokus:**
           - Efek kafein pada stres sangat penting
           - Alasan konsumsi kafein berpengaruh
           - Konsentrasi tanpa kafein menjadi faktor kunci
        """)
    
    # Informasi model
    st.markdown("### Detail Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Parameter Model")
        st.write("""
        - **Algoritma:** Random Forest
        - **n_estimators:** 200
        - **max_depth:** 10
        - **random_state:** 42
        - **Scoring:** 
          - Regresi: R² Score
          - Klasifikasi: Accuracy, Precision, Recall, F1-Score
        """)
    
    with col2:
        st.markdown("#### Data Processing")
        st.write("""
        - **Preprocessing:** 
          - One-hot encoding untuk kategorikal
          - Standard scaling untuk numerik
          - Handling missing values
        - **Split Ratio:** 80% training, 20% testing
        - **Cross-validation:** 5-fold
        - **Feature Selection:** Berdasarkan importance score
        """)

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "dashboard"
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Analisis & Prediksi Konsumsi Kafein")
        st.markdown("---")
        
        pages = [
            ("Dashboard", "dashboard"),
            ("Visualisasi", "visualisasi"),
            ("Prediksi", "prediksi"),
            ("Evaluasi Model", "evaluasi")
        ]
        
        selected_page = st.radio(
            "Navigasi",
            [page[0] for page in pages],
            key="navigation"
        )
        
        for page_name, page_key in pages:
            if page_name == selected_page:
                st.session_state.page = page_key
        
        st.markdown("---")
        
        # Load model results sekali saja
        if 'model_results' not in st.session_state:
            with st.spinner("Memuat model..."):
                st.session_state.model_results = load_model_results()
        
        model_results = st.session_state.model_results
        
        if model_results:
            st.success("✅ Model siap digunakan")
            metrics = model_results.get('metrics', {})
            reg_metrics = metrics.get('regression', {})
            r2 = reg_metrics.get('r2', 0)
            st.metric("R² Score", f"{r2:.3f}")
            
            # Tampilkan informasi tambahan di sidebar
            with st.expander("Detail Model"):
                feature_names = model_results.get('feature_names', [])
                if feature_names:
                    st.write(f"Jumlah fitur: {len(feature_names)}")
                
                models = model_results.get('models', {})
                if models:
                    st.write(f"Jumlah model: {len(models)}")
                    st.write("1. Regression Model")
                    st.write("2. Mood Classifier")
                    st.write("3. Focus Classifier")
        else:
            st.warning("⚠ Model belum dimuat")
        
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            if 'model_results' in st.session_state:
                del st.session_state.model_results
            st.success("Cache dibersihkan")
            st.rerun()
    
    page = st.session_state.page
    
    if page == "dashboard":
        st.markdown('<h1 class="main-header">Analisis & Prediksi Konsumsi Kafein</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>Selamat Datang di Dashboard Analisis Konsumsi Kafein</h3>
            <p>Dashboard ini menganalisis data survei konsumsi kafein dengan fitur prediksi menggunakan Machine Learning.</p>
            <p><strong>Fitur Utama:</strong></p>
            <ul>
                <li>Visualisasi data interaktif</li>
                <li>Prediksi berbasis AI menggunakan Random Forest</li>
                <li>Analisis pola konsumsi kafein</li>
                <li>Rekomendasi personal berdasarkan data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        df = load_data()
        if df is not None:
            # Statistik data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Total Responden</div>
                    <div class="metric-value">{len(df)}</div>
                    <div class="metric-change positive-change">orang</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                if 'age' in df.columns:
                    avg_age = df['age'].mean()
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-title">Rata-rata Umur</div>
                        <div class="metric-value">{avg_age:.1f}</div>
                        <div class="metric-change neutral-change">tahun</div>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with col3:
                if 'sleep_quality' in df.columns:
                    avg_sleep = df['sleep_quality'].mean()
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-title">Kualitas Tidur</div>
                        <div class="metric-value">{avg_sleep:.1f}</div>
                        <div class="metric-change positive-change">/5</div>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with col4:
                missing_total = df.isnull().sum().sum()
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Missing Values</div>
                    <div class="metric-value">{missing_total}</div>
                    <div class="metric-change negative-change">data</div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Model status
            st.markdown('<h3 class="sub-header">Status Model</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.session_state.model_results:
                    st.markdown("""
                    <div class="card">
                        <h4>✅ Model Siap Digunakan</h4>
                        <p>Model Random Forest telah dimuat dari file model_results.pkl:</p>
                        <ul>
                            <li>Regresi: Prediksi kualitas tidur</li>
                            <li>Klasifikasi: Pengaruh kafein pada mood</li>
                            <li>Klasifikasi: Pengaruh kafein pada fokus</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="card">
                        <h4>⚠ Model Belum Tersedia</h4>
                        <p>Pastikan file model_results.pkl ada di folder yang sama dengan aplikasi ini.</p>
                        <p>File ini dihasilkan dari notebook training yang telah dijalankan sebelumnya.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if not st.session_state.model_results:
                    st.markdown("""
                    <div class="card">
                        <h4>Instruksi</h4>
                        <ol>
                            <li>Jalankan notebook training</li>
                            <li>Pastikan model_results.pkl dibuat</li>
                            <li>Refresh halaman ini</li>
                        </ol>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Pratinjau data
            with st.expander("Pratinjau Data"):
                st.dataframe(df.head())
                st.write(f"Dimensi data: {df.shape[0]} baris × {df.shape[1]} kolom")
        
        else:
            st.warning("Data tidak ditemukan. Pastikan file 'Datacleaning.csv' ada di folder yang sama.")
    
    elif page == "visualisasi":
        df = load_data()
        show_visualizations_page(df)
    
    elif page == "prediksi":
        show_prediction_page(st.session_state.model_results)
    
    elif page == "evaluasi":
        show_evaluation_page(st.session_state.model_results)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "Dashboard Analisis Kafein © KELOMPOK 9 | Dibuat dengan Streamlit & Machine Learning"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()