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

# Function untuk Interpolasi ISPU
def hitung_sub_indeks(nilai, batas_konsentrasi, batas_indeks):
    if pd.isna(nilai):
        return np.nan
    for i in range(len(batas_konsentrasi) - 1):
        if batas_konsentrasi[i] <= nilai <= batas_konsentrasi[i+1]:
            C_low, C_high = batas_konsentrasi[i], batas_konsentrasi[i+1]
            I_low, I_high = batas_indeks[i], batas_indeks[i+1]
            return ((I_high - I_low) / (C_high - C_low)) * (nilai - C_low) + I_low
    return 500  # Batas ekstrem

# 2. Function untuk Load dan Preprocessing Data
@st.cache_data
def load_data():
    df = pd.read_csv("PRSA_Data_Cleaned.zip")
    
    # Membuat kolom 'date' gabungan dari year, month, day
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # Skala Indeks Standar ISPU
    skala_indeks = [0, 50, 100, 200, 300, 500]

    # --- 1. Kalkulasi Sub-Indeks ISPU Tiap Polutan
    df['I_PM25'] = df['PM2.5'].apply(lambda x: hitung_sub_indeks(x, [0, 15, 35, 55, 150, 500], skala_indeks))
    df['I_PM10'] = df['PM10'].apply(lambda x: hitung_sub_indeks(x, [0, 50, 150, 250, 350, 500], skala_indeks))
    df['I_SO2']  = df['SO2'].apply(lambda x: hitung_sub_indeks(x, [0, 40, 80, 380, 800, 1600], skala_indeks))
    df['I_NO2']  = df['NO2'].apply(lambda x: hitung_sub_indeks(x, [0, 40, 80, 180, 280, 565], skala_indeks))
    df['I_O3']   = df['O3'].apply(lambda x: hitung_sub_indeks(x, [0, 50, 100, 168, 208, 748], skala_indeks))
    df['I_CO']   = (df['CO'] / 1000).apply(lambda x: hitung_sub_indeks(x, [0, 2, 4, 9, 15, 32], skala_indeks))

    # --- 2. Menentukan Critical Pollutant (Nilai Maksimum Gabungan)
    df['Indeks_Gabungan'] = df[['I_PM25', 'I_PM10', 'I_SO2', 'I_NO2', 'I_O3', 'I_CO']].max(axis=1)

    # --- 3. Pengkategorian Kualitas Udara Berdasarkan Indeks Gabungan
    bins = [-np.inf, 50, 100, 200, 300, np.inf]
    labels = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
    df['Kategori_Gabungan'] = pd.cut(df['Indeks_Gabungan'], bins=bins, labels=labels)

    # Pengkategorian Kualitas Udara Gabungan Berdasarkan PM2.5
    def categorize_ispu(indeks):
        if pd.isna(indeks):
            return np.nan
        elif indeks <= 50:
            return 'Baik'
        elif indeks <= 100:
            return 'Sedang'
        elif indeks <= 200:
            return 'Tidak Sehat'
        else:
            return 'Sangat Tidak Sehat / Berbahaya'

    df['Kategori_Gabungan'] = df['Indeks_Gabungan'].apply(categorize_ispu)
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

# st.date_input sudah mendukung ketik manual di UI Streamlit.
# Tambahkan 'format' dan 'help' agar pengguna tahu format ketik manualnya (DD/MM/YYYY)
date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal (Mulai - Selesai):",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    format="DD/MM/YYYY",
    help="Anda dapat mengeklik kalender atau mengetik tanggal secara manual dengan format DD/MM/YYYY"
)

# Proteksi penanganan input tanggal (Termasuk saat user sedang mengetik manual di Streamlit)
if isinstance(date_range, (tuple, list)):
    if len(date_range) == 2:
        # Jika pengguna sudah selesai memilih/mengetik rentang tanggal lengkap
        start_date, end_date = date_range
    elif len(date_range) == 1:
        # Jika pengguna baru mengetik/mengeklik 1 tanggal (belum selesai)
        start_date = date_range[0]
        end_date = max_date
    else:
        start_date, end_date = min_date, max_date
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

    # 1. Agregasi Rata-rata Partikulat (PM2.5 & PM10) per Stasiun
    tabel_pm25 = df.groupby('station')['PM2.5'].mean().to_frame('PM2.5_mean').sort_values(by='PM2.5_mean', ascending=True)
    tabel_pm10 = df.groupby('station')['PM10'].mean().to_frame('PM10_mean').sort_values(by='PM10_mean', ascending=True)

    # 2. Agregasi Rata-rata Gas per Stasiun & Hitung Total_Gas_Index (Sesuai Metode Notebook)
    gas_pollutants = ['SO2', 'NO2', 'CO', 'O3']
    tabel_gas_kota = df.groupby('station')[gas_pollutants].mean()

    tabel_gas_kota['Total_Gas_Index'] = (
        (tabel_gas_kota['SO2'] / tabel_gas_kota['SO2'].max()) +
        (tabel_gas_kota['NO2'] / tabel_gas_kota['NO2'].max()) +
        (tabel_gas_kota['CO'] / tabel_gas_kota['CO'].max()) +
        (tabel_gas_kota['O3'] / tabel_gas_kota['O3'].max())
    )
    tabel_gas_sorted = tabel_gas_kota.sort_values(by='Total_Gas_Index', ascending=True)

    # 3. Ambil Nilai Tertinggi untuk Card Metric
    top_pm25_st, top_pm25_val = tabel_pm25.index[-1], tabel_pm25['PM2.5_mean'].iloc[-1]
    top_pm10_st, top_pm10_val = tabel_pm10.index[-1], tabel_pm10['PM10_mean'].iloc[-1]
    top_gas_st, top_gas_val = tabel_gas_sorted.index[-1], tabel_gas_sorted['Total_Gas_Index'].iloc[-1]

    # 4. Hitung Rata-rata Baseline Seluruh Stasiun untuk Pembanding Delta
    avg_pm25_all = df['PM2.5'].mean()
    avg_pm10_all = df['PM10'].mean()
    avg_gas_all = tabel_gas_sorted['Total_Gas_Index'].mean()

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

    # ==============================================================================
    # BARIS 1 (KOLOM 1, 2, 3): PM2.5, PM10, SO2
    # ==============================================================================
    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        st.metric(
            label=f"Jam Puncak PM2.5 ({peak_summary['PM2.5']['stasiun']})", 
            value=f"Pukul {peak_summary['PM2.5']['jam']:02d}:00",
            delta=f"+{peak_summary['PM2.5']['nilai']:.1f} µg/m³",
            delta_color="inverse"
        )

    with row1_col2:
        st.metric(
            label=f"Jam Puncak PM10 ({peak_summary['PM10']['stasiun']})", 
            value=f"Pukul {peak_summary['PM10']['jam']:02d}:00",
            delta=f"+{peak_summary['PM10']['nilai']:.1f} µg/m³",
            delta_color="inverse"
        )

    with row1_col3:
        st.metric(
            label=f"Jam Puncak SO₂ ({peak_summary['SO2']['stasiun']})", 
            value=f"Pukul {peak_summary['SO2']['jam']:02d}:00",
            delta=f"+{peak_summary['SO2']['nilai']:.1f} µg/m³",
            delta_color="inverse"
        )

    # ==============================================================================
    # BARIS 2 (KOLOM 1, 2, 3): NO2, CO, O3
    # ==============================================================================
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.metric(
            label=f"Jam Puncak NO₂ ({peak_summary['NO2']['stasiun']})", 
            value=f"Pukul {peak_summary['NO2']['jam']:02d}:00",
            delta=f"+{peak_summary['NO2']['nilai']:.1f} µg/m³",
            delta_color="inverse"
        )

    with row2_col2:
        st.metric(
            label=f"Jam Puncak CO ({peak_summary['CO']['stasiun']})", 
            value=f"Pukul {peak_summary['CO']['jam']:02d}:00",
            delta=f"+{peak_summary['CO']['nilai']:.1f} µg/m³",
            delta_color="inverse"
        )

    with row2_col3:
        st.metric(
            label=f"Jam Puncak O₃ ({peak_summary['O3']['stasiun']})", 
            value=f"Pukul {peak_summary['O3']['jam']:02d}:00",
            delta=f"+{peak_summary['O3']['nilai']:.1f} µg/m³",
            delta_color="inverse"
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
    st.header("Distribusi Tingkat Pencemaran Udara Gabungan per Stasiun 📊")
    
    st.markdown("""
    #### ❓ **Pertanyaan Bisnis 3**
    > **Bagaimana pengelompokan data kualitas udara berdasarkan tingkat ISPU gabungan ke dalam kategori Baik, Sedang, Tidak Sehat, dan Sangat Tidak Sehat / Berbahaya pada setiap stasiun pemantau?**
    """)
    st.write("")

    # 1. Hitung Persentase Distribusi Kategori per Stasiun
    cat_df = pd.crosstab(df['station'], df['Kategori_Gabungan'], normalize='index') * 100
    
    # Kunci urutan kategori agar rapi dari Baik ke Berbahaya
    categories = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat / Berbahaya']
    existing_cats = [c for c in categories if c in cat_df.columns]
    cat_df = cat_df.reindex(columns=existing_cats)

    # 2. METRIK HIGHLIGHT (Stasiun dengan persentase buruk tertinggi)
    unhealthy_cols = [c for c in ['Tidak Sehat', 'Sangat Tidak Sehat / Berbahaya'] if c in cat_df.columns]
    
    # Inisialisasi status data buruk
    has_bad_data = False

    if unhealthy_cols:
        cat_df['Total_Buruk'] = cat_df[unhealthy_cols].sum(axis=1)
        # Pastikan kolom Total_Buruk tidak kosong / tidak nol semua
        if not cat_df['Total_Buruk'].dropna().empty:
            has_bad_data = True
            max_bad_st, max_bad_val = cat_df['Total_Buruk'].idxmax(), cat_df['Total_Buruk'].max()
            min_bad_st, min_bad_val = cat_df['Total_Buruk'].idxmin(), cat_df['Total_Buruk'].min()
            diff_bad = max_bad_val - min_bad_val

            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    label=f"Risiko Udara Buruk Tertinggi ({max_bad_st})", 
                    value=f"{max_bad_val:.1f}%",
                    delta=f"+{diff_bad:.1f}%",
                    delta_color="inverse"
                )
            with m2:
                st.metric(
                    label=f"Risiko Udara Buruk Terendah ({min_bad_st})", 
                    value=f"{min_bad_val:.1f}%",
                    delta=f"-{diff_bad:.1f}%",
                    delta_color="inverse"
                )

    st.markdown("---")
    st.subheader("📊 Grafik Visualisasi Data")
    st.write("")

    # 3. GRAFIK STACKED BAR CHART (4 WARNA ISPU)
    palette_colors = {
        'Baik': '#2ecc71',
        'Sedang': '#f1c40f',
        'Tidak Sehat': '#e67e22',
        'Sangat Tidak Sehat / Berbahaya': '#e74c3c'
    }
    color_list = [palette_colors[c] for c in existing_cats]

    fig3, ax = plt.subplots(figsize=(12, 6))
    cat_df[existing_cats].plot(kind='bar', stacked=True, color=color_list, ax=ax, width=0.55)
    
    ax.set_title("Distribusi Kategori Kualitas Udara ISPU per Stasiun", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Stasiun Pemantau", fontsize=12, labelpad=10)
    ax.set_ylabel("Persentase Distribusi (%)", fontsize=12)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    ax.legend(title="Kategori ISPU", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig3)
    
    # 4. INSIGHT DINAMIS UNTUK PERTANYAAN 3
    st.subheader("💡 Insight Pertanyaan 3")
    if has_bad_data:
        st.markdown(f"""
        * **Kombinasi Udara Buruk:** Persentase akumulasi kategori **Tidak Sehat** hingga **Sangat Tidak Sehat / Berbahaya** antar stasiun berkisar antara **{min_bad_val:.2f}% hingga {max_bad_val:.2f}%**.
        * **Stasiun Berisiko Tertinggi:** Stasiun **{max_bad_st}** mencatatkan gabungan persentase kategori udara buruk tertinggi, yaitu **{max_bad_val:.2f}%**.
        * **Stasiun Relatif Aman:** Stasiun **{min_bad_st}** memiliki persentase kategori udara buruk terendah yaitu **{min_bad_val:.2f}%**.
        """)
    else:
        st.info("ℹ️ Kualitas udara pada rentang filter ini sangat baik (tidak ditemukan data kategori 'Tidak Sehat' maupun 'Sangat Tidak Sehat / Berbahaya').")
