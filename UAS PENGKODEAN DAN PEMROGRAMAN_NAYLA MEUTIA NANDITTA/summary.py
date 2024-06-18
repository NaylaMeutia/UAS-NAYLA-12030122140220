import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tabel 1: Data Pelanggan
data_pelanggan = {
    'Nama Pelanggan': ['Alice', 'Bob', 'Cathy', 'David', 'Eva'],
    'Jenis Kelamin': ['Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan'],
    'Usia': [30, 25, 35, 40, 28],
    'Nomor Telepon': ['08123456789', '08567891234', '08111223344', '08765432100', '08987654321']
}

df_pelanggan = pd.DataFrame(data_pelanggan)

# Tabel 2: Data Layanan yang Digunakan
data_layanan = {
    'Nama Pelanggan': ['Alice', 'Bob', 'Cathy', 'David', 'Eva'],
    'Tanggal Layanan': ['2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05'],
    'Jenis Layanan': ['Potong Rambut', 'Creambath', 'Cat Rambut', 'Facial', 'Manicure'],
    'Biaya': [100000, 150000, 200000, 300000, 120000]
}

df_layanan = pd.DataFrame(data_layanan)

# Tabel 3: Data Produk yang Dibeli
data_produk = {
    'Nama Pelanggan': ['Alice', 'Bob', 'Cathy', 'David', 'Eva'],
    'Tanggal Pembelian': ['2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05'],
    'Nama Produk': ['Shampoo', 'Conditioner', 'Hair Serum', 'Face Mask', 'Nail Polish'],
    'Harga Satuan': [50000, 60000, 75000, 80000, 30000]
}

df_produk = pd.DataFrame(data_produk)

# Tabel 4: Data Keuntungan
data_keuntungan = {
    'Tanggal': ['2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05'],
    'Pemasukan Layanan': [100000, 150000, 200000, 300000, 120000],
    'Pemasukan Produk': [50000, 60000, 75000, 80000, 30000],
    'Total Pemasukan': [150000, 210000, 275000, 380000, 150000]
}

df_keuntungan = pd.DataFrame(data_keuntungan)

# 1. Pembersihan Data: Cek missing values
print("1. Pembersihan Data:")
print("   Tabel Pelanggan:")
print(df_pelanggan.isnull().sum())
print("   Tabel Layanan:")
print(df_layanan.isnull().sum())
print("   Tabel Produk:")
print(df_produk.isnull().sum())
print("   Tabel Keuntungan:")
print(df_keuntungan.isnull().sum())
print()

# 2. Analisis Deskriptif:
print("2. Analisis Deskriptif:")
# - Usia rata-rata pelanggan
rata_usia_pelanggan = df_pelanggan['Usia'].mean()
print(f"   Usia rata-rata pelanggan: {rata_usia_pelanggan:.2f} tahun")
# - Distribusi biaya dari jenis layanan yang tersedia
biaya_layanan = df_layanan.groupby('Jenis Layanan')['Biaya'].describe()
print("   Distribusi biaya dari jenis layanan yang tersedia:")
print(biaya_layanan)
# - Produk dengan harga satuan tertinggi dan terendah
harga_tertinggi = df_produk.loc[df_produk['Harga Satuan'].idxmax()]
harga_terendah = df_produk.loc[df_produk['Harga Satuan'].idxmin()]
print(f"   Produk dengan harga satuan tertinggi:\n{harga_tertinggi}")
print(f"   Produk dengan harga satuan terendah:\n{harga_terendah}")
print()

# 3. Visualisasi Data:
print("3. Visualisasi Data:")
# - Histogram untuk distribusi usia pelanggan
plt.figure(figsize=(8, 5))
plt.hist(df_pelanggan['Usia'], bins=5, edgecolor='black')
plt.title('Histogram Usia Pelanggan')
plt.xlabel('Usia')
plt.ylabel('Frekuensi')
plt.grid(True)
plt.show()

# - Diagram batang untuk jumlah penjualan produk per jenis produk
jumlah_produk_per_jenis = df_produk['Nama Produk'].value_counts()
plt.figure(figsize=(8, 5))
jumlah_produk_per_jenis.plot(kind='bar', color='skyblue')
plt.title('Jumlah Penjualan Produk per Jenis Produk')
plt.xlabel('Nama Produk')
plt.ylabel('Jumlah Penjualan')
plt.grid(axis='y')
plt.show()

# - Boxplot untuk membandingkan biaya layanan berdasarkan jenis layanan
plt.figure(figsize=(8, 5))
plt.boxplot([df_layanan[df_layanan['Jenis Layanan'] == layanan]['Biaya'] for layanan in df_layanan['Jenis Layanan'].unique()],
            labels=df_layanan['Jenis Layanan'].unique())
plt.title('Boxplot Biaya Layanan')
plt.xlabel('Jenis Layanan')
plt.ylabel('Biaya (Rp)')
plt.grid(True)
plt.show()

# Membuat scatterplot untuk hubungan antara usia pelanggan dan biaya layanan
plt.figure(figsize=(8, 5))
plt.scatter(df_pelanggan['Usia'], df_layanan['Biaya'], color='green', alpha=0.7)
plt.title('Hubungan Antara Usia Pelanggan dan Biaya Layanan')
plt.xlabel('Usia Pelanggan')
plt.ylabel('Biaya Layanan (Rp)')
plt.grid(True)
plt.show()

# 4. Analisis Lebih Lanjut:
print("4. Analisis Lebih Lanjut:")
# - Korelasi antara usia pelanggan dan total pengeluaran mereka untuk layanan
df_gabungan = pd.merge(df_pelanggan, df_layanan, on='Nama Pelanggan')
korelasi_usia_biaya = df_gabungan['Usia'].corr(df_gabungan['Biaya'])
print(f"   Korelasi antara usia pelanggan dan biaya layanan: {korelasi_usia_biaya:.2f}")
print()

# - Hubungan antara pemasukan dari layanan dan produk dari bulan ke bulan
korelasi_pemasukan = df_keuntungan['Pemasukan Layanan'].corr(df_keuntungan['Pemasukan Produk'])
print(f"   Korelasi antara pemasukan dari layanan dan produk: {korelasi_pemasukan:.2f}")
print()
