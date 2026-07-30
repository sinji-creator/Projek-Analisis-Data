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

# 2. Function untuk Load dan Preprocessing Data
@st.cache_data
def load_data():
    df = pd.read_csv("PRSA_Data_Cleaned.zip")
    
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

# Load Data
df = load_data()

# Header Dashboard
st.title("🍃 Beijing Air Quality Dataset 📊")
st.markdown("---")

# Navigation Sidebar
st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Tampilan Grafik:",
    [
        "1. Peringkat & Akumulasi Polusi Per Stasiun (Terendah ke Tertinggi)",
        "2. Pola Distribusi Harian Stasiun Terburuk",
        "3. Distribusi Tingkat Pencemaran Udara Gabungan"
    ]
)


# ==============================================================================
# MENU 1: PERINGKAT & AKUMULASI POLUSI PER STASIUN
# ==============================================================================
if menu == "1. Peringkat & Akumulasi Polusi Per Stasiun (Terendah ke Tertinggi)":
    st.header("Peringkat Konsentrasi Polutan dan Akumulasi Polusi Gas per Stasiun")
    
    # Pertanyaan Bisnis 1 (Formatted Clean & Bold)
    st.markdown("""
    #### ❓ **Pertanyaan Bisnis 1**
    > **Stasiun pemantauan mana yang memiliki rata-rata konsentrasi PM2.5, PM10, dan Indeks Gas tertinggi selama periode Maret 2013–Februari 2017, sehingga dapat menjadi prioritas dalam pengendalian pencemaran udara?**
    """)
    st.write("")

    # Hitung Aggregasi Data
    tabel_pm25 = df.groupby('station')['PM2.5'].mean().to_frame('PM2.5_mean').sort_values(by='PM2.5_mean', ascending=True)
    tabel_pm10 = df.groupby('station')['PM10'].mean().to_frame('PM10_mean').sort_values(by='PM10_mean', ascending=True)
    tabel_gas_sorted = df.groupby('station')['Total_Gas_Index'].mean().to_frame('Total_Gas_Index').sort_values(by='Total_Gas_Index', ascending=True)

    # Buat 3 kolom di Streamlit
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
        tabel_pm10_sorted = tabel_pm10.sort_values(by='PM10_mean', ascending=True)
        stasiun_pm10 = tabel_pm10_sorted.index
        rata_rata_pm10 = tabel_pm10_sorted['PM10_mean']

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
        tabel_gas_final = tabel_gas_sorted.sort_values(by='Total_Gas_Index', ascending=True)
        stasiun_gas = tabel_gas_final.index
        index_gas = tabel_gas_final['Total_Gas_Index']

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

    # Insight & Pemisahan Jawaban
    st.subheader("💡 Insight Pertanyaan 1")
    
    st.markdown("""
    * **PM2.5 Tertinggi:** Stasiun **Dongsi** mencatatkan konsentrasi rata-rata tertinggi yaitu **85,64 $\mu g/m^3$**.
    * **PM10 Tertinggi:** Stasiun **Gucheng** mendominasi konsentrasi rata-rata partikel kasar sebesar **118,70 $\mu g/m^3$**.
    * **Akumulasi Gas Tertinggi:** Stasiun **Nongzhanguan** menduduki urutan teratas untuk Indeks Polusi Gas Gabungan dengan skor **3,709**.
    """)
    

# ==============================================================================
# MENU 2: POLA DISTRIBUSI HARIAN STASIUN TERBURUK
# ==============================================================================
elif menu == "2. Pola Distribusi Harian Stasiun Terburuk":
    st.header("Pola Distribusi Harian Konsentrasi Polutan di Stasiun Terburuk")
    
    # Pertanyaan Bisnis 2 (Formatted Clean & Bold)
    st.markdown("""
    #### ❓ **Pertanyaan Bisnis 2**
    > **Pada jam berapa konsentrasi rata-rata polutan (PM2.5, PM10, SO₂, NO₂, CO, dan O₃) mencapai nilai tertinggi selama periode Maret 2013–Februari 2017, sehingga dapat ditentukan waktu prioritas untuk pemantauan dan pengendalian pencemaran udara?**
    """)
    st.write("")

    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    trend_stasiun_jam = df.groupby(['station', 'hour'])[pollutants].mean().reset_index()
    colors = ['#d9534f', '#f0ad4e', '#5cb85c', '#5bc0de', '#9b59b6', '#e67e22']

    for i, col in enumerate(pollutants):
        idx_tertinggi = trend_stasiun_jam[col].idxmax()
        stasiun_terburuk = trend_stasiun_jam.loc[idx_tertinggi, 'station']
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

        idx_puncak = np.argmax(y_values)
        jam_puncak = x_hours[idx_puncak]
        nilai_puncak = y_values[idx_puncak]

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

    # Insight & Pemisahan Jawaban
    st.subheader("💡 Insight Pertanyaan 2")
    
    st.markdown("""
    * **PM2.5 & PM10:** Mencapai puncak tertinggi secara bersamaan pada malam hari, tepatnya pukul **21.00**.
    * **Partikel Gas ($SO_2$ & $NO_2$):** $SO_2$ memuncak di pagi hari (**pukul 10.00**), sedangkan $NO_2$ memuncak di malam hari (**pukul 22.00**).
    * **Karbon Monoksida (CO):** Mencapai lonjakan di pagi hari pada jam sibuk lalulintas (**pukul 08.00**).
    * **Ozon ($O_3$):** Memiliki pola terbalik dari polutan lain, mencapai puncak di sore hari (**pukul 16.00**) akibat radiasi sinar matahari.
    """)
    
    
# ==============================================================================
# MENU 3: DISTRIBUSI TINGKAT PENCEMARAN UDARA GABUNGAN
# ==============================================================================
elif menu == "3. Distribusi Tingkat Pencemaran Udara Gabungan":
    st.header("Distribusi Tingkat Pencemaran Udara Gabungan per Stasiun (Maret 2013 - Februari 2017)")
    
    # Pertanyaan Bisnis 3 (Formatted Clean & Bold)
    st.markdown("""
    #### ❓ **Pertanyaan Bisnis 3**
    > **Bagaimana pengelompokan data kualitas udara berdasarkan tingkat konsentrasi gabungan polutan ke dalam kategori Baik, Sedang, Tidak Sehat, dan Sangat Tidak Sehat selama periode Maret 2013–Februari 2017, sehingga dapat diketahui distribusi tingkat pencemaran udara pada setiap stasiun?**
    """)
    st.write("")

    # Hitung Persentase Kategori per Stasiun
    cat_df = pd.crosstab(df['station'], df['Kategori_Gabungan'], normalize='index') * 100
    categories = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat / Berbahaya']
    
    existing_cats = [c for c in categories if c in cat_df.columns]
    cat_df = cat_df.reindex(columns=existing_cats)
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    fig3, ax = plt.subplots(figsize=(12, 6))
    cat_df.plot(kind='bar', stacked=True, color=colors[:len(existing_cats)], ax=ax, width=0.5)
    
    ax.set_title("Distribusi Tingkat Pencemaran Udara Gabungan per Stasiun\n(Maret 2013 - Februari 2017)", 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Stasiun Pemantau", fontsize=12, labelpad=10)
    ax.set_ylabel("Persentase Distribusi (%)", fontsize=12)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    ax.legend(title="Kategori Kualitas Udara", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    plt.tight_layout()
    st.pyplot(fig3)

    # Insight & Pemisahan Jawaban
    st.subheader("💡 Insight Pertanyaan 3")
    
    st.markdown("""
    * **Dominasi Kategori Berbahaya:** Seluruh stasiun pemantau di Beijing didominasi oleh kategori **Sangat Tidak Sehat / Berbahaya** dengan persentase berkisar antara **41,76% hingga 54,90%**.
    * **Stasiun Terburuk:** Stasiun **Dongsi** mencatatkan persentase polusi berisiko tinggi tertinggi, yaitu **54,90%**.
    * **Stasiun Relatif Terbaik:** Stasiun **Dingling** memiliki persentase kategori berbahaya terendah (**41,76%**), meskipun posisinya masih tergolong tinggi.
    """)
    
    
