import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Analisis Kualitas Udara di Beijing",
    page_icon="🌤️",
    layout="wide"
)

# CSS Custom: Menyembunyikan Ikon Panah pada st.metric tapi Mengizinkan Warna Merah/Hijau
st.markdown("""
    <style>
    [data-testid="stMetricDeltaIcon"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Function untuk Load dan Preprocessing Data
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard/PRSA_Data_Cleaned.zip")
    
    # Membuat kolom 'date' gabungan dari year, month, day
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    # Menghitung Total Gas Index jika belum ada
    if 'Total_Gas_Index' not in df.columns:
        so2_norm = (df['SO2'] - df['SO2'].min()) / (df['SO2'].max() - df['SO2'].min())
        no2_norm = (df['NO2'] - df['NO2'].min()) / (df['NO2'].max() - df['NO2'].min())
        co_norm = (df['CO'] - df['CO'].min()) / (df['CO'].max() - df['CO'].min())
        o3_norm = (df['O3'] - df['O3'].min()) / (df['O3'].max() - df['O3'].min())
        df['Total_Gas_Index'] = so2_norm + no2_norm + co_norm + o3_norm

    # Pengkategorian Kualitas Udara Gabungan Berdasarkan PM2.5
    def categorize_air_quality(row):
        pm = row['PM2.5']
        if pm <= 35:
            return 'Baik'
        elif pm <= 75:
            return 'Sedang'
        elif pm <= 150:
            return 'Tidak Sehat'
        else:
            return 'Sangat Tidak Sehat / Berbahaya'

    df['Kategori_Gabungan'] = df.apply(categorize_air_quality, axis=1)
    return df

# Load Data Utama
df_raw = load_data()

# Header Dashboard
st.title("🍃 Beijing Air Quality Dashboard 📊")
st.markdown("---")

# ==============================================================================
# SIDEBAR: FITUR INTERAKTIF & MANIPULASI DATA UTAMA
# ==============================================================================
st.sidebar.title("📌 Menu Navigasi & Filter")

# 1. Pilihan Menu Tampilan Grafik
menu = st.sidebar.radio(
    "Pilih Tampilan Grafik:",
    [
        "1. Peringkat & Akumulasi Polusi Per Stasiun (Terendah ke Tertinggi)",
        "2. Pola Distribusi Harian Stasiun Terburuk",
        "3. Distribusi Tingkat Pencemaran Udara Gabungan"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Filter Data Interaktif")

# Filter 1: Pilih Stasiun Pemantau
all_stations = df_raw['station'].unique().tolist()
selected_stations = st.sidebar.multiselect(
    "Pilih Stasiun Pemantau:",
    options=all_stations,
    default=all_stations
)

# Filter 2: Pilih Rentang Tanggal (Tanggal, Bulan, Tahun)
min_date = df_raw['date'].min().date()
max_date = df_raw['date'].max().date()

date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal (Mulai - Selesai):",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Proteksi penanganan input tanggal
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# ==============================================================================
# EKSEKUSI MANIPULASI DATA (df_filtered)
# ==============================================================================
df = df_raw[
    (df_raw['station'].isin(selected_stations)) &
    (df_raw['date'].dt.date >= start_date) &
    (df_raw['date'].dt.date <= end_date)
]

# Proteksi jika hasil filter menghasilkan data kosong
if df.empty:
    st.warning("⚠️ Data kosong! Silakan sesuaikan stasiun atau rentang tanggal pada filter di sidebar.")
    st.stop()


# ==============================================================================
# MENU 1: PERINGKAT & AKUMULASI POLUSI PER STASIUN
# ==============================================================================
if menu == "1. Peringkat & Akumulasi Polusi Per Stasiun (Terendah ke Tertinggi)":
    st.header("Peringkat Konsentrasi Polutan dan Akumulasi Polusi Gas per Stasiun")
    
    st.markdown("""
    #### ❓ **Pertanyaan Bisnis 1**
    > **Stasiun pemantauan mana yang memiliki rata-rata konsentrasi PM2.5, PM10, dan Indeks Gas tertinggi selama periode pemantauan, sehingga dapat menjadi prioritas dalam pengendalian pencemaran udara?**
    """)
    st.write("")

    # Hitung Aggregasi Data dari df yang ter-filter
    tabel_pm25 = df.groupby('station')['PM2.5'].mean().to_frame('PM2.5_mean').sort_values(by='PM2.5_mean', ascending=True)
    tabel_pm10 = df.groupby('station')['PM10'].mean().to_frame('PM10_mean').sort_values(by='PM10_mean', ascending=True)
    tabel_gas_sorted = df.groupby('station')['Total_Gas_Index'].mean().to_frame('Total_Gas_Index').sort_values(by='Total_Gas_Index', ascending=True)

    top_pm25_st, top_pm25_val = tabel_pm25.index[-1], tabel_pm25['PM2.5_mean'].iloc[-1]
    top_pm10_st, top_pm10_val = tabel_pm10.index[-1], tabel_pm10['PM10_mean'].iloc[-1]
    top_gas_st, top_gas_val = tabel_gas_sorted.index[-1], tabel_gas_sorted['Total_Gas_Index'].iloc[-1]

    # Hitung rata-rata gabungan stasiun sebagai baseline pembanding delta
    avg_pm25_all = df['PM2.5'].mean()
    avg_pm10_all = df['PM10'].mean()
    avg_gas_all = df['Total_Gas_Index'].mean()

    diff_pm25 = top_pm25_val - avg_pm25_all
    diff_pm10 = top_pm10_val - avg_pm10_all
    diff_gas = top_gas_val - avg_gas_all

    st.subheader("📊 Grafik Visualisasi Data")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label=f"PM2.5 Tertinggi ({top_pm25_st})", 
            value=f"{top_pm25_val:.2f} µg/m³",
            delta=f"{diff_pm25:+.2f}",
            delta_color="inverse" # Bernilai positif (+) otomatis berwarna MERAH tanpa panah
        )
    with m2:
        st.metric(
            label=f"PM10 Tertinggi ({top_pm10_st})", 
            value=f"{top_pm10_val:.2f} µg/m³",
            delta=f"{diff_pm10:+.2f}",
            delta_color="inverse" # Bernilai positif (+) otomatis berwarna MERAH tanpa panah
        )
    with m3:
        st.metric(
            label=f"Indeks Gas Tertinggi ({top_gas_st})", 
            value=f"{top_gas_val:.3f}",
            delta=f"{diff_gas:+.3f}",
            delta_color="inverse" # Bernilai positif (+) otomatis berwarna MERAH tanpa panah
        )

    st.write("")

    col1, col2, col3 = st.columns(3)

    # GRAFIK 1: PM2.5
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 8))
        stasiun_pm25 = tabel_pm25.index
        rata_rata_pm25 = tabel_pm25['PM2.5_mean']

        sns.barplot(
            ax=ax1,
            x=rata_rata_pm25,
            y=stasiun_pm25,
            palette='RdYlGn_r',
            hue=stasiun_pm25,
            legend=False
        )
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.2f', padding=5, fontsize=10, weight='bold')

        ax1.set_title('Peringkat PM2.5', fontsize=14, pad=12, weight='bold')
        ax1.set_xlabel('Konsentrasi ($\mu g/m^3$)', fontsize=11)
        ax1.set_ylabel('Nama Stasiun Pemantau', fontsize=11)
        ax1.set_xlim(0, max(rata_rata_pm25) * 1.15)
        ax1.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig1)

    # GRAFIK 2: PM10
    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 9.3))
        stasiun_pm10 = tabel_pm10.index
        rata_rata_pm10 = tabel_pm10['PM10_mean']

        sns.barplot(
            ax=ax2,
            x=rata_rata_pm10,
            y=stasiun_pm10,
            palette='RdYlGn_r',
            hue=stasiun_pm10,
            legend=False
        )
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.2f', padding=5, fontsize=10, weight='bold')

        ax2.set_title('Peringkat PM10', fontsize=14, pad=12, weight='bold')
        ax2.set_xlabel('Konsentrasi ($\mu g/m^3$)', fontsize=11)
        ax2.set_ylabel('')  
        ax2.set_xlim(0, max(rata_rata_pm10) * 1.15)
        ax2.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig2)

    # GRAFIK 3: AKUMULASI GAS
    with col3:
        fig3, ax3 = plt.subplots(figsize=(6, 8))
        stasiun_gas = tabel_gas_sorted.index
        index_gas = tabel_gas_sorted['Total_Gas_Index']

        sns.barplot(
            ax=ax3,
            x=index_gas,
            y=stasiun_gas,
            palette='RdYlGn_r',
            hue=stasiun_gas,
            legend=False
        )
        for container in ax3.containers:
            ax3.bar_label(container, fmt='%.3f', padding=5, fontsize=10, weight='bold')

        ax3.set_title('Akumulasi Polusi Gas', fontsize=14, pad=12, weight='bold')
        ax3.set_xlabel('Skor Indeks (SO2, NO2, CO, O3)', fontsize=11)
        ax3.set_ylabel('')  
        ax3.set_xlim(0, max(index_gas) * 1.15)
        ax3.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig3)

    # INSIGHT DINAMIS UNTUK PERTANYAAN 1
    st.subheader("💡 Insight Pertanyaan 1")
    st.markdown(f"""
    * **PM2.5 Tertinggi:** Stasiun **{top_pm25_st}** mencatatkan konsentrasi rata-rata tertinggi yaitu **{top_pm25_val:.2f} $\mu g/m^3$**.
    * **PM10 Tertinggi:** Stasiun **{top_pm10_st}** mendominasi konsentrasi rata-rata partikel kasar sebesar **{top_pm10_val:.2f} $\mu g/m^3$**.
    * **Akumulasi Gas Tertinggi:** Stasiun **{top_gas_st}** menduduki urutan teratas untuk Indeks Polusi Gas Gabungan dengan skor **{top_gas_val:.3f}**.
    """)


# ==============================================================================
# MENU 2: POLA DISTRIBUSI HARIAN STASIUN TERBURUK
# ==============================================================================
elif menu == "2. Pola Distribusi Harian Stasiun Terburuk":
    st.header("Pola Distribusi Harian Konsentrasi Polutan di Stasiun Terburuk")
    
    st.markdown("""
    #### ❓ **Pertanyaan Bisnis 2**
    > **Pada jam berapa konsentrasi rata-rata polutan (PM2.5, PM10, SO₂, NO₂, CO, dan O₃) mencapai nilai tertinggi selama periode pemantauan, sehingga dapat ditentukan waktu prioritas untuk pemantauan dan pengendalian pencemaran udara?**
    """)
    st.write("")

    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    trend_stasiun_jam = df.groupby(['station', 'hour'])[pollutants].mean().reset_index()
    colors = ['#d9534f', '#f0ad4e', '#5cb85c', '#5bc0de', '#9b59b6', '#e67e22']

    # Hitung data puncak
    peak_summary = {}
    for col in pollutants:
        idx_tertinggi = trend_stasiun_jam[col].idxmax()
        peak_summary[col] = {
            'stasiun': trend_stasiun_jam.loc[idx_tertinggi, 'station'],
            'jam': trend_stasiun_jam.loc[idx_tertinggi, 'hour'],
            'nilai': trend_stasiun_jam.loc[idx_tertinggi, col]
        }

    st.subheader("📊 Grafik Visualisasi Data")
    m1, m2 = st.columns(2)
    with m1:
        st.metric(
            label=f"Jam Puncak PM2.5 ({peak_summary['PM2.5']['stasiun']})", 
            value=f"Pukul {peak_summary['PM2.5']['jam']:02d}:00",
            delta=f"+{peak_summary['PM2.5']['nilai']:.1f} µg/m³",
            delta_color="inverse" # Bernilai positif (+) otomatis berwarna MERAH tanpa panah
        )
    with m2:
        st.metric(
            label=f"Jam Puncak PM10 ({peak_summary['PM10']['stasiun']})", 
            value=f"Pukul {peak_summary['PM10']['jam']:02d}:00",
            delta=f"+{peak_summary['PM10']['nilai']:.1f} µg/m³",
            delta_color="inverse" # Bernilai positif (+) otomatis berwarna MERAH tanpa panah
        )

    # SUBHEADER GRAFIK DENGAN LOGO BAR CHART
    st.write("")

    # GRAFIK PERTANYAAN 2
    for i, col in enumerate(pollutants):
        stasiun_terburuk = peak_summary[col]['stasiun']
        data_stasiun_spesifik = trend_stasiun_jam[trend_stasiun_jam['station'] == stasiun_terburuk]

        x_hours = data_stasiun_spesifik['hour'].values
        y_values = data_stasiun_spesifik[col].values

        fig, ax = plt.subplots(figsize=(10, 4.5))

        sns.lineplot(
            ax=ax,
            x=x_hours,
            y=y_values,
            marker='o',
            linewidth=2,
            color=colors[i],
            label=f"Rata-rata {col} di {stasiun_terburuk}"
        )

        ax.fill_between(x_hours, y_values, color=colors[i], alpha=0.15)

        jam_puncak = peak_summary[col]['jam']
        nilai_puncak = peak_summary[col]['nilai']

        teks_label = f"Puncak: Pukul {jam_puncak:02d}:00\nStasiun: {stasiun_terburuk}\nNilai: {nilai_puncak:.1f}"

        ax.annotate(
            teks_label,
            xy=(jam_puncak, nilai_puncak),          
            xytext=(jam_puncak, nilai_puncak + (max(y_values) * 0.12)),
            textcoords='data',
            ha='center',
            va='bottom',
            weight='bold',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="gray", alpha=0.9), 
            arrowprops=dict(
                arrowstyle="->",                    
                connectionstyle="arc3",              
                color="black",                      
                lw=1.5                              
            )
        )

        ax.set_title(f'Pola Distribusi Harian Konsentrasi {col} di Stasiun Terburuk ({stasiun_terburuk})', fontsize=12, pad=12, weight='bold')
        ax.set_xlabel('Jam Dalam Sehari (00:00 - 23:00)', fontsize=10, labelpad=6)
        ax.set_ylabel(f'Konsentrasi {col} ($\mu g/m^3$)', fontsize=10)

        ax.set_xticks(range(0, 24))
        ax.set_xlim(-0.5, 23.5)
        ax.set_ylim(min(y_values) * 0.85, max(y_values) * 1.35) 

        ax.legend(loc='upper left', fontsize=9, shadow=True)
        ax.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # INSIGHT DINAMIS UNTUK PERTANYAAN 2
    st.subheader("💡 Insight Pertanyaan 2")
    st.markdown(f"""
    * **PM2.5:** Puncak tertinggi di stasiun **{peak_summary['PM2.5']['stasiun']}** terjadi pada **pukul {peak_summary['PM2.5']['jam']:02d}:00** ({peak_summary['PM2.5']['nilai']:.1f} $\mu g/m^3$).
    * **PM10:** Puncak tertinggi di stasiun **{peak_summary['PM10']['stasiun']}** terjadi pada **pukul {peak_summary['PM10']['jam']:02d}:00** ({peak_summary['PM10']['nilai']:.1f} $\mu g/m^3$).
    * **$SO_2$:** Memuncak di stasiun **{peak_summary['SO2']['stasiun']}** pada **pukul {peak_summary['SO2']['jam']:02d}:00** ({peak_summary['SO2']['nilai']:.1f} $\mu g/m^3$).
    * **$NO_2$:** Memuncak di stasiun **{peak_summary['NO2']['stasiun']}** pada **pukul {peak_summary['NO2']['jam']:02d}:00** ({peak_summary['NO2']['nilai']:.1f} $\mu g/m^3$).
    * **CO:** Lonjakan tertinggi di stasiun **{peak_summary['CO']['stasiun']}** pada **pukul {peak_summary['CO']['jam']:02d}:00** ({peak_summary['CO']['nilai']:.1f} $\mu g/m^3$).
    * **Ozon ($O_3$):** Memuncak di stasiun **{peak_summary['O3']['stasiun']}** pada **pukul {peak_summary['O3']['jam']:02d}:00** ({peak_summary['O3']['nilai']:.1f} $\mu g/m^3$).
    """)


# ==============================================================================
# MENU 3: DISTRIBUSI TINGKAT PENCEMARAN UDARA GABUNGAN
# ==============================================================================
elif menu == "3. Distribusi Tingkat Pencemaran Udara Gabungan":
    st.header("Distribusi Tingkat Pencemaran Udara Gabungan per Stasiun")
    
    st.markdown("""
    #### ❓ **Pertanyaan Bisnis 3**
    > **Bagaimana pengelompokan data kualitas udara berdasarkan tingkat konsentrasi gabungan polutan ke dalam kategori Baik, Sedang, Tidak Sehat, dan Sangat Tidak Sehat selama periode pemantauan, sehingga dapat diketahui distribusi tingkat pencemaran udara pada setiap stasiun?**
    """)
    st.write("")

    cat_df = pd.crosstab(df['station'], df['Kategori_Gabungan'], normalize='index') * 100
    categories = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat / Berbahaya']
    
    existing_cats = [c for c in categories if c in cat_df.columns]
    cat_df = cat_df.reindex(columns=existing_cats)

    if 'Sangat Tidak Sehat / Berbahaya' in cat_df.columns:
        series_danger = cat_df['Sangat Tidak Sehat / Berbahaya'].dropna()
        if not series_danger.empty:
            max_danger_st, max_danger_val = series_danger.idxmax(), series_danger.max()
            min_danger_st, min_danger_val = series_danger.idxmin(), series_danger.min()
            diff_danger = max_danger_val - min_danger_val

            st.subheader("📊 Grafik Visualisasi Data")
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    label=f"Risiko Berbahaya Tertinggi ({max_danger_st})", 
                    value=f"{max_danger_val:.2f}%",
                    delta=f"+{diff_danger:.2f}%",
                    delta_color="inverse" # Nilai Positif (+) otomatis MERAH tanpa panah
                )
            with m2:
                st.metric(
                    label=f"Risiko Berbahaya Terendah ({min_danger_st})", 
                    value=f"{min_danger_val:.2f}%",
                    delta=f"-{diff_danger:.2f}%",
                    delta_color="inverse" # Nilai Negatif (-) otomatis HIJAU tanpa panah
                )


    # SUBHEADER GRAFIK DENGAN LOGO BAR CHART
    st.write("")

    # GRAFIK PERTANYAAN 3
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    fig3, ax = plt.subplots(figsize=(12, 6))
    cat_df.plot(kind='bar', stacked=True, color=colors[:len(existing_cats)], ax=ax, width=0.5)
    
    ax.set_title("Distribusi Tingkat Pencemaran Udara Gabungan per Stasiun", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Stasiun Pemantau", fontsize=12, labelpad=10)
    ax.set_ylabel("Persentase Distribusi (%)", fontsize=12)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    ax.legend(title="Kategori Kualitas Udara", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    plt.tight_layout()
    st.pyplot(fig3)

    # INSIGHT DINAMIS UNTUK PERTANYAAN 3
    st.subheader("💡 Insight Pertanyaan 3")
    if 'Sangat Tidak Sehat / Berbahaya' in cat_df.columns and not series_danger.empty:
        st.markdown(f"""
        * **Dominasi Kategori Berbahaya:** Pada rentang waktu dan stasiun yang dipilih, persentase kategori **Sangat Tidak Sehat / Berbahaya** berkisar antara **{min_danger_val:.2f}% hingga {max_danger_val:.2f}%**.
        * **Stasiun Terburuk:** Stasiun **{max_danger_st}** mencatatkan persentase kategori berbahaya tertinggi, yaitu **{max_danger_val:.2f}%**.
        * **Stasiun Relatif Aman:** Stasiun **{min_danger_st}** memiliki persentase kategori berbahaya terendah yaitu **{min_danger_val:.2f}%**.
        """)
    else:
        st.info("ℹ️ Tidak ditemukan data kategori 'Sangat Tidak Sehat / Berbahaya' untuk kombinasi filter ini.")
