# visualizations.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import textwrap
import warnings

warnings.filterwarnings('ignore')

# Set style untuk konsisten
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'DejaVu Sans'
sns.set_palette("husl")

# ============================================
# FUNGSI UTAMA - HANYA SATU VERSI DARI SETIAP FUNGSI
# ============================================

def plot_age_distribution(df):
    """Plot distribusi usia detail dengan styling yang baik"""
    if 'age' not in df.columns:
        return None
    
    # Pastikan age numerik
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    age_data = df['age'].dropna()
    
    if len(age_data) == 0:
        return None
    
    # Hitung statistik
    mean_age = age_data.mean()
    median_age = age_data.median()
    std_age = age_data.std()
    total_responden = len(age_data)
    
    # Buat figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Histogram dengan gradien warna
    n, bins, patches = ax.hist(age_data, bins=15, alpha=0.85, 
                               color='#4A90E2', edgecolor='black', linewidth=0.5)
    
    # Warna gradien berdasarkan tinggi
    for patch in patches:
        patch.set_facecolor(plt.cm.Blues(patch.get_height()/max(n)))
    
    # KDE plot
    sns.kdeplot(data=age_data, color='#D0021B', linewidth=2.5, ax=ax)
    
    # Garis mean dan median
    ax.axvline(mean_age, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.9, 
               label=f'Mean: {mean_age:.1f}')
    ax.axvline(median_age, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.9, 
               label=f'Median: {median_age:.1f}')
    
    # Judul dan label
    ax.set_title('Distribusi Usia Responden', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Usia (Tahun)', fontsize=12, labelpad=10)
    ax.set_ylabel('Frekuensi', fontsize=12, labelpad=10)
    
    # Tambah nilai di atas bar
    for i, (count, patch) in enumerate(zip(n, patches)):
        if count > 0:  
            ax.text(patch.get_x() + patch.get_width()/2, count + 0.5, 
                    f'{int(count)}', ha='center', va='bottom', fontsize=9)
    
    # Box statistik
    stats_text = f'Rata-rata: {mean_age:.1f} tahun\nMedian: {median_age:.1f} tahun\nStd Dev: {std_age:.1f} tahun'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, fontweight='semibold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     alpha=0.9, edgecolor='#CCCCCC', linewidth=1))
    
    # Legenda
    ax.legend(loc='upper left', fontsize=10, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.9)
    
    # Format axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    
    # Footer
    plt.figtext(0.5, 0.01, f'Total Responden: {total_responden} orang', 
                ha='center', fontsize=10, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def plot_gender_distribution(df):
    """Plot distribusi gender dengan pie chart yang menarik"""
    if 'gender' not in df.columns:
        return None
    
    # Hitung distribusi
    gender_counts = df['gender'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']  # Merah untuk perempuan, biru untuk laki-laki
    
    # Standardize labels
    labels = []
    for idx in gender_counts.index:
        if 'laki' in str(idx).lower():
            labels.append('Laki-laki')
        elif 'perempuan' in str(idx).lower():
            labels.append('Perempuan')
        else:
            labels.append(str(idx))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Pie chart
    wedges, texts, autotexts = ax.pie(
        gender_counts.values, labels=labels,
        autopct=lambda p: f'{p:.1f}%\n({int(p/100*sum(gender_counts.values))})',
        colors=colors[:len(gender_counts)], startangle=90,
        explode=[0.05] * len(gender_counts), shadow=True,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'alpha': 0.9},
        textprops={'fontsize': 11, 'fontweight': 'medium', 'color': '#2C3E50'},
        pctdistance=0.75
    )
    
    # Lingkaran tengah
    centre_circle = plt.Circle((0, 0), 0.65, fc='white', edgecolor='white', linewidth=2)
    ax.add_artist(centre_circle)
    
    # Teks total di tengah
    ax.annotate(
        f'TOTAL\n{sum(gender_counts.values)}', xy=(0, 0), xytext=(0, 0),
        ha='center', va='center', fontsize=14,
        fontweight='bold', color='#2C3E50'
    )
    
    # Judul
    ax.set_title('Distribusi Jenis Kelamin', fontsize=16, fontweight='bold', pad=20)
    
    # Legenda
    legend_labels = [f'{label} ({value})' for label, value in zip(labels, gender_counts.values)]
    ax.legend(
        wedges, legend_labels,
        title="Jenis Kelamin", loc="center left",
        bbox_to_anchor=(1, 0.5), fontsize=11, frameon=True
    )
    
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig

def plot_age_group_distribution(df):
    """Plot distribusi kelompok usia"""
    if 'age' not in df.columns:
        return None
    
    # Buat kelompok usia jika belum ada
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    bins = [0, 18, 25, 35, 45, 100]
    labels = ['<18', '18-24', '25-34', '35-44', '45+']
    
    age_groups = pd.cut(df['age'].dropna(), bins=bins, labels=labels, right=False)
    age_group_counts = age_groups.value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar chart dengan warna pastel
    colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', '#FFD700']
    bars = ax.bar(age_group_counts.index, age_group_counts.values, 
                 color=colors[:len(age_group_counts)])
    
    ax.set_xlabel('Kelompok Usia', fontsize=12)
    ax.set_ylabel('Jumlah Responden', fontsize=12)
    ax.set_title('Distribusi Kelompok Usia', fontsize=14, fontweight='bold')
    
    # Tambahkan nilai di atas bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_favorite_drink(df):
    """Plot minuman kafein favorit"""
    if 'favorite_caffeine_drink' not in df.columns:
        return None
    
    # Hitung frekuensi minuman favorit (top 10)
    drink_counts = df['favorite_caffeine_drink'].value_counts().head(10).sort_values(ascending=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(drink_counts)))
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Horizontal bar chart
    bars = ax.barh(drink_counts.index, drink_counts.values, 
                   color=colors, edgecolor='white', linewidth=1.5, height=0.7)
    
    # Ranking number di kiri
    for i, (drink, value) in enumerate(zip(drink_counts.index, drink_counts.values)):
        ax.text(-max(drink_counts.values)*0.05, i,
                f'{len(drink_counts)-i}',
                ha='right', va='center',
                fontsize=12, fontweight='bold',
                color='#E74C3C')
        
        # Nilai di kanan
        ax.text(value + max(drink_counts.values)*0.01, i,
                f'{int(value)}',
                ha='left', va='center',
                fontsize=11, fontweight='semibold')
    
    ax.set_title(f'Minuman Berkafein Favorit (Top 10)',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.xaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim([-max(drink_counts.values)*0.1, max(drink_counts.values)*1.15])
    
    plt.tight_layout()
    return fig

def plot_caffeine_frequency(df):
    """Plot frekuensi konsumsi kafein"""
    if 'caffeine_frequency' not in df.columns:
        return None
    
    # Urutan frekuensi yang logis
    frequency_order = ['Tidak Pernah', 'Jarang', 'Kadang-kadang', 'Sering', 'Sangat Sering']
    
    # Hitung distribusi dan urutkan
    freq_counts = df['caffeine_frequency'].value_counts()
    freq_counts = freq_counts.reindex([f for f in frequency_order if f in freq_counts.index])
    
    colors = ['#4ECDC4', '#45B7D1', '#FFD166', '#FF6B6B', '#96CEB4']
    
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.bar(freq_counts.index, freq_counts.values, 
                  color=colors[:len(freq_counts)], 
                  edgecolor='white', linewidth=2, alpha=0.85)
    
    # Tambahkan nilai di atas bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{int(height)}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    ax.set_title(f'Frekuensi Konsumsi Kafein', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Frekuensi', fontsize=12, fontweight='semibold', labelpad=10)
    ax.set_ylabel('Jumlah Responden', fontsize=12, fontweight='semibold', labelpad=10)
    
    plt.xticks(rotation=30, ha='right', fontsize=11)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def plot_caffeine_time(df):
    """Plot waktu konsumsi kafein"""
    if 'caffeine_time' not in df.columns:
        return None
    
    time_counts = df['caffeine_time'].value_counts().sort_values(ascending=False)
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#9B59B6', '#3498DB', 
              '#1ABC9C', '#5DADE2', '#52BE80', '#48C9B0', '#76D7C4'][:len(time_counts)]
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    bars = ax.bar(time_counts.index, time_counts.values,
                  color=colors, edgecolor='white', linewidth=2, alpha=0.9)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'{int(height)}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#2C3E50')
    
    ax.set_title(f'Waktu Konsumsi Kafein',
                 fontsize=15, fontweight='bold', pad=15, color='#2C3E50')
    ax.set_xlabel('Waktu', fontsize=12, fontweight='semibold', labelpad=10)
    ax.set_ylabel('Jumlah Responden', fontsize=12, fontweight='semibold', labelpad=10)
    
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def plot_caffeine_before_sleep(df):
    """Plot konsumsi kafein sebelum tidur"""
    if 'caffeine_before_sleep' not in df.columns:
        return None
    
    counts = df['caffeine_before_sleep'].value_counts()
    
    # Warna berdasarkan jumlah kategori
    colors_map = {
        2: ['#FF6B6B','#4ECDC4'],
        3: ['#FF6B6B','#4ECDC4','#95E1D3'],
        4: ['#FF6B6B','#4ECDC4','#FFD166','#06D6A0'],
        5: ['#FF6B6B','#4ECDC4','#FFD166','#06D6A0','#118AB2'],
        6: ['#FF6B6B','#4ECDC4','#FFD166','#06D6A0','#118AB2','#EF476F']
    }
    
    explode_map = {
        2: (0.05, 0),
        3: (0.05, 0, 0),
        4: (0.05, 0, 0, 0),
        5: (0.05, 0, 0, 0, 0),
        6: (0.05, 0, 0, 0, 0, 0)
    }
    
    n_categories = min(len(counts), 6)
    colors = colors_map.get(n_categories, ['#FF6B6B','#4ECDC4','#FFD166','#06D6A0','#118AB2','#EF476F'])[:n_categories]
    explode = explode_map.get(n_categories, (0.05, 0, 0, 0, 0, 0))[:n_categories]
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90,
        colors=colors, explode=explode, shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold', 'color': '#2c3e50'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'alpha': 0.9}
    )
    
    # Format autotext
    for t in autotexts:
        t.set_color('white')
        t.set_fontweight('bold')
        t.set_fontsize(10)
    
    # Legenda
    legend_labels = [f'{l} = {v}' for l, v in zip(counts.index, counts.values)]
    ax.legend(wedges, legend_labels, title="Keterangan:", loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10, frameon=True,
              fancybox=True, shadow=True, borderpad=1)
    
    plt.title("Distribusi Frekuensi Konsumsi Kafein Sebelum Tidur",
              fontsize=16, fontweight='bold', color='#2c3e50', pad=25)
    
    plt.text(0.5, -0.05, f"Total Responden: {len(df)}",
             ha='center', va='center', transform=ax.transAxes,
             fontsize=10, style='italic', color='#7f8c8d')
    
    ax.axis('equal')
    plt.tight_layout(rect=[0, 0.05, 0.85, 0.95])
    return fig

def plot_sleep_quality(df):
    """Plot kualitas tidur"""
    if 'sleep_quality' not in df.columns:
        return None
    
    # Mapping numerik ke label jika perlu
    if df['sleep_quality'].dtype in [np.int64, np.float64]:
        sleep_quality_counts = pd.Series({
            'Sangat Buruk': len(df[df['sleep_quality'] == 1]) if 1 in df['sleep_quality'].values else 0,
            'Buruk': len(df[df['sleep_quality'] == 2]) if 2 in df['sleep_quality'].values else 0,
            'Cukup': len(df[df['sleep_quality'] == 3]) if 3 in df['sleep_quality'].values else 0,
            'Baik': len(df[df['sleep_quality'] == 4]) if 4 in df['sleep_quality'].values else 0,
            'Sangat Baik': len(df[df['sleep_quality'] == 5]) if 5 in df['sleep_quality'].values else 0,
        })
        sleep_quality_counts = sleep_quality_counts[sleep_quality_counts > 0]
    else:
        sleep_quality_counts = df['sleep_quality'].value_counts()
    
    # Urutkan jika ada label standar
    quality_order = ['Sangat Buruk', 'Buruk', 'Cukup', 'Baik', 'Sangat Baik']
    ordered_labels = [q for q in quality_order if q in sleep_quality_counts.index]
    ordered_counts = [sleep_quality_counts[q] for q in ordered_labels]
    
    if len(ordered_counts) != len(sleep_quality_counts):
        ordered_labels = sleep_quality_counts.index.tolist()
        ordered_counts = sleep_quality_counts.values
    
    # Warna berdasarkan jumlah kategori
    if len(ordered_counts) == 5:
        colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
    elif len(ordered_counts) == 3:
        colors = ['#e74c3c', '#f1c40f', '#2ecc71']
    else:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ordered_counts)))
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    bars = ax.bar(ordered_labels, ordered_counts, color=colors, 
                  edgecolor='white', linewidth=2, alpha=0.9)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (max(ordered_counts)*0.01),
                f'{int(h)}\n({h/len(df)*100:.1f}%)',
                ha='center', fontsize=10, fontweight='bold', color='#2c3e50')
    
    plt.xticks(fontsize=11, fontweight='semibold', color='#2c3e50')
    ax.set_xlabel("Tingkat Kualitas Tidur", fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_ylabel("Jumlah Responden", fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_ylim(0, max(ordered_counts) * 1.15)
    
    ax.set_title("Distribusi Kualitas Tidur Responden",
              fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    
    # Hapus border atas dan kanan
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    
    plt.tight_layout()
    return fig

def plot_feeling_without_caffeine(df):
    """Plot perasaan tanpa kafein"""
    if 'feeling_without_caffeine' not in df.columns:
        return None
    
    feeling_counts = df['feeling_without_caffeine'].value_counts().head(10)  # Top 10 saja
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#95E1D3', '#FF9A8B', 
              '#A8E6CF', '#D4A5A5', '#9FD8DF', '#F8B195', '#F67280']
    
    bars = ax.bar(feeling_counts.index, feeling_counts.values, 
                  color=colors[:len(feeling_counts)], 
                  edgecolor='white', linewidth=2)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 5,
            f'{int(h)}\n({h/len(df)*100:.1f}%)',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )
    
    # Wrap label jika panjang
    wrapped_labels = ['\n'.join(textwrap.wrap(str(label), 12)) for label in feeling_counts.index]
    ax.set_xticks(range(len(wrapped_labels)))
    ax.set_xticklabels(wrapped_labels, fontsize=11)
    
    ax.set_title('Perasaan Saat Tidak Mengonsumsi Kafein (Top 10)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Jumlah Responden', fontsize=11)
    ax.set_ylim(0, max(feeling_counts.values) * 1.1)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def plot_mood_without_caffeine(df):
    """Plot mood tanpa kafein"""
    if 'mood_drop_without_caffeine' not in df.columns:
        return None
    
    mood_counts = df['mood_drop_without_caffeine'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    colors = ['#6A5ACD', '#FF8C00', '#20B2AA', '#DC143C', '#2E8B57']
    
    bars = ax.bar(
        mood_counts.index,
        mood_counts.values,
        color=colors[:len(mood_counts)],
        edgecolor='white',
        linewidth=2
    )
    
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 5,
            f'{int(h)}\n({h/len(df)*100:.1f}%)',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )
    
    wrapped_labels = ['\n'.join(textwrap.wrap(str(label), 12)) for label in mood_counts.index]
    ax.set_xticks(range(len(wrapped_labels)))
    ax.set_xticklabels(wrapped_labels, fontsize=11)
    
    ax.set_title("Penurunan Mood Tanpa Kafein", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Kondisi Mood", fontsize=12, fontweight='bold')
    ax.set_ylabel("Jumlah Responden", fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(mood_counts.values) * 1.1)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def plot_focus_boost_caffeine(df):
    """Plot dampak kafein terhadap fokus"""
    if 'focus_boost_caffeine' not in df.columns:
        return None
    
    focus_counts = df['focus_boost_caffeine'].value_counts().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    colors_focus = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#1A73E8', '#B983FF']
    
    ax.barh(
        focus_counts.index,
        focus_counts.values,
        color=colors_focus[:len(focus_counts)],
        edgecolor='white',
        linewidth=2
    )
    
    for i, v in enumerate(focus_counts.values):
        ax.text(
            v + 5,
            i,
            f'{int(v)} ({v/len(df)*100:.1f}%)',
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    wrapped_labels = ['\n'.join(textwrap.wrap(str(label), 20)) for label in focus_counts.index]
    ax.set_yticks(range(len(wrapped_labels)))
    ax.set_yticklabels(wrapped_labels, fontsize=11)
    
    ax.set_title("Peningkatan Fokus akibat Kafein", fontsize=14, fontweight='bold', pad=45)
    ax.set_xlabel("Jumlah Responden", fontsize=12, fontweight='bold')
    ax.set_ylabel("Tingkat Fokus", fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(focus_counts.values) * 1.1)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Total responden di atas
    plt.text(
        0.5, 1.04, f'Total: {len(df)} responden',
        transform=ax.transAxes,
        fontsize=9,
        fontweight='bold',
        ha='center'
    )
    
    ax.invert_yaxis()  # Urutkan dari atas ke bawah
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df):
    """Membuat heatmap korelasi antar variabel numerik"""
    # Pilih kolom numerik saja
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Hitung matriks korelasi
    corr_matrix = df[numeric_cols].corr()
    
    # Buat heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": .8}, ax=ax)
    
    ax.set_title('Matriks Korelasi Antar Variabel Numerik', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_stress_effect(df):
    """Membuat bar chart efek kafein pada stres"""
    if 'stress_effect_caffeine' not in df.columns:
        return None
    
    stress_counts = df['stress_effect_caffeine'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(stress_counts.index, stress_counts.values, 
                 color=plt.cm.RdYlBu(np.linspace(0, 1, len(stress_counts))))
    
    ax.set_xlabel('Efek pada Stres', fontsize=12)
    ax.set_ylabel('Jumlah Responden', fontsize=12)
    ax.set_title('Pengaruh Kafein terhadap Tingkat Stres', 
                fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_digestion_issues(df):
    """Membuat pie chart masalah pencernaan dari kafein"""
    if 'digestion_issues_from_caffeine' not in df.columns:
        return None
    
    digestion_counts = df['digestion_issues_from_caffeine'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB6C1']
    
    wedges, texts, autotexts = ax.pie(digestion_counts.values, 
                                      labels=digestion_counts.index,
                                      autopct='%1.1f%%',
                                      colors=colors[:len(digestion_counts)],
                                      startangle=90,
                                      wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Masalah Pencernaan Akibat Kafein', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_confidence_boost(df):
    """Plot peningkatan kepercayaan diri dari kafein"""
    if 'confidence_boost_from_caffeine' not in df.columns:
        return None
    
    confidence_counts = df['confidence_boost_from_caffeine'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FFD700', '#FFA500', '#FF8C00', '#FF4500', '#FF6347']
    
    bars = ax.barh(confidence_counts.index[::-1], confidence_counts.values[::-1], 
                  color=colors[:len(confidence_counts)])
    
    ax.set_xlabel('Jumlah Responden', fontsize=12)
    ax.set_ylabel('Tingkat Kepercayaan Diri', fontsize=12)
    ax.set_title('Peningkatan Kepercayaan Diri dari Kafein', 
                fontsize=14, fontweight='bold')
    
    for i, v in enumerate(confidence_counts.values[::-1]):
        ax.text(v + 0.5, i, f'{int(v)}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_concentration_without_caffeine(df):
    """Plot konsentrasi tanpa kafein"""
    if 'concentration_without_caffeine' not in df.columns:
        return None
    
    concentration_counts = df['concentration_without_caffeine'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#FFD166', '#4ECDC4', '#45B7D1', '#118AB2']
    
    bars = ax.bar(concentration_counts.index, concentration_counts.values,
                 color=colors[:len(concentration_counts)])
    
    ax.set_xlabel('Tingkat Konsentrasi', fontsize=12)
    ax.set_ylabel('Jumlah Responden', fontsize=12)
    ax.set_title('Konsentrasi Tanpa Mengonsumsi Kafein', 
                fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

# ============================================
# FUNGSI BANTUAN
# ============================================

def plot_all_visualizations(df):
    """Menampilkan semua visualisasi dalam satu list"""
    visualizations = []
    titles = []
    
    # Usia dan demografi
    fig = plot_age_distribution(df)
    if fig:
        visualizations.append(fig)
        titles.append("Distribusi Usia Responden")
    
    fig = plot_gender_distribution(df)
    if fig:
        visualizations.append(fig)
        titles.append("Distribusi Jenis Kelamin")
    
    fig = plot_age_group_distribution(df)
    if fig:
        visualizations.append(fig)
        titles.append("Distribusi Kelompok Usia")
    
    # Minuman dan konsumsi
    fig = plot_favorite_drink(df)
    if fig:
        visualizations.append(fig)
        titles.append("Minuman Berkafein Favorit")
    
    fig = plot_caffeine_frequency(df)
    if fig:
        visualizations.append(fig)
        titles.append("Frekuensi Konsumsi Kafein")
    
    fig = plot_caffeine_time(df)
    if fig:
        visualizations.append(fig)
        titles.append("Waktu Konsumsi Kafein")
    
    fig = plot_caffeine_before_sleep(df)
    if fig:
        visualizations.append(fig)
        titles.append("Konsumsi Kafein Sebelum Tidur")
    
    # Kualitas tidur
    fig = plot_sleep_quality(df)
    if fig:
        visualizations.append(fig)
        titles.append("Kualitas Tidur Responden")
    
    # Efek tanpa kafein
    fig = plot_feeling_without_caffeine(df)
    if fig:
        visualizations.append(fig)
        titles.append("Perasaan Tanpa Kafein")
    
    fig = plot_mood_without_caffeine(df)
    if fig:
        visualizations.append(fig)
        titles.append("Penurunan Mood Tanpa Kafein")
    
    fig = plot_concentration_without_caffeine(df)
    if fig:
        visualizations.append(fig)
        titles.append("Konsentrasi Tanpa Kafein")
    
    # Efek dengan kafein
    fig = plot_focus_boost_caffeine(df)
    if fig:
        visualizations.append(fig)
        titles.append("Peningkatan Fokus dengan Kafein")
    
    fig = plot_stress_effect(df)
    if fig:
        visualizations.append(fig)
        titles.append("Efek pada Stres")
    
    fig = plot_confidence_boost(df)
    if fig:
        visualizations.append(fig)
        titles.append("Peningkatan Kepercayaan Diri")
    