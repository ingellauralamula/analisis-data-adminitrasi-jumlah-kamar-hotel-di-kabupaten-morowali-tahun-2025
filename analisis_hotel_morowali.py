"""
Analisis Data Hotel Kabupaten Morowali 2025
Metode: Regresi Linear & Support Vector Classification (SVC)

Studi Kasus:
- Regresi Linear : Memprediksi total kamar hotel berdasarkan jumlah tipe kamar
- SVC            : Mengklasifikasikan hotel ke kategori KECIL / SEDANG / BESAR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# ===========================================================
# 1. LOAD & PREPROCESSING DATA
# ===========================================================
print("=" * 60)
print("  ANALISIS DATA HOTEL KABUPATEN MOROWALI 2025")
print("=" * 60)

FILE_PATH = "kompilasi-data-administrasi-jumlah-kamar-hotel-di-kabupaten-morowali-tahun-2025-(2025)-397.xlsx"

df_raw = pd.read_excel(FILE_PATH)

# Hapus baris 'Total' (baris terakhir berisi ringkasan)
df_raw = df_raw[df_raw['Nama Usaha'].str.lower() != 'total'].copy()
df_raw['Jumlah kamar'] = pd.to_numeric(df_raw['Jumlah kamar'], errors='coerce')
df_raw.dropna(subset=['Jumlah kamar'], inplace=True)

print(f"\n[INFO] Data mentah: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")

# Agregasi per hotel → satu baris per usaha
df_hotel = (
    df_raw.groupby('Nama Usaha')['Jumlah kamar']
    .agg(Total_Kamar='sum', Jumlah_Tipe_Kamar='count')
    .reset_index()
)
df_hotel['Rata_Kamar_Per_Tipe'] = df_hotel['Total_Kamar'] / df_hotel['Jumlah_Tipe_Kamar']

print(f"[INFO] Jumlah hotel unik: {len(df_hotel)}")
print("\n[PREVIEW DATA HOTEL TERAGREGASI]")
print(df_hotel.head(10).to_string(index=False))

# Buat label kategori untuk SVC
def kategorikan(total):
    if total <= 10:
        return 'KECIL'
    elif total <= 25:
        return 'SEDANG'
    else:
        return 'BESAR'

df_hotel['Kategori'] = df_hotel['Total_Kamar'].apply(kategorikan)

print("\n[DISTRIBUSI KATEGORI HOTEL]")
print(df_hotel['Kategori'].value_counts().to_string())

# ===========================================================
# 2. REGRESI LINEAR
#    Fitur  : Jumlah_Tipe_Kamar, Rata_Kamar_Per_Tipe
#    Target : Total_Kamar
# ===========================================================
print("\n" + "=" * 60)
print("  BAGIAN A: REGRESI LINEAR")
print("=" * 60)

X_reg = df_hotel[['Jumlah_Tipe_Kamar', 'Rata_Kamar_Per_Tipe']].values
y_reg = df_hotel['Total_Kamar'].values

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

model_lr = LinearRegression()
model_lr.fit(X_train_r, y_train_r)
y_pred_r = model_lr.predict(X_test_r)

mae_r  = mean_absolute_error(y_test_r, y_pred_r)
mse_r  = mean_squared_error(y_test_r, y_pred_r)
rmse_r = np.sqrt(mse_r)
r2_r   = r2_score(y_test_r, y_pred_r)

print(f"\nKoefisien Regresi  : {model_lr.coef_}")
print(f"Intercept          : {model_lr.intercept_:.4f}")
print(f"\n[EVALUASI REGRESI LINEAR]")
print(f"  MAE  = {mae_r:.4f}")
print(f"  MSE  = {mse_r:.4f}")
print(f"  RMSE = {rmse_r:.4f}")
print(f"  R²   = {r2_r:.4f}")

# Detail prediksi vs aktual
print("\n[PERBANDINGAN AKTUAL VS PREDIKSI - REGRESI LINEAR]")
hasil_reg = pd.DataFrame({
    'Aktual'  : y_test_r,
    'Prediksi': np.round(y_pred_r, 2),
    'Selisih' : np.round(y_pred_r - y_test_r, 2)
})
print(hasil_reg.to_string(index=False))

# ===========================================================
# 3. SUPPORT VECTOR CLASSIFICATION (SVC)
#    Fitur  : Jumlah_Tipe_Kamar, Rata_Kamar_Per_Tipe
#    Target : Kategori (KECIL / SEDANG / BESAR)
# ===========================================================
print("\n" + "=" * 60)
print("  BAGIAN B: SUPPORT VECTOR CLASSIFICATION (SVC)")
print("=" * 60)

le = LabelEncoder()
y_svc = le.fit_transform(df_hotel['Kategori'])   # encode label

X_svc = df_hotel[['Jumlah_Tipe_Kamar', 'Rata_Kamar_Per_Tipe']].values

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_svc, y_svc, test_size=0.2, random_state=42, stratify=y_svc
)

# Scaling wajib untuk SVM/SVC
scaler = StandardScaler()
X_train_s_sc = scaler.fit_transform(X_train_s)
X_test_s_sc  = scaler.transform(X_test_s)

model_svc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model_svc.fit(X_train_s_sc, y_train_s)
y_pred_s = model_svc.predict(X_test_s_sc)

print(f"\n[EVALUASI SVC]")
print(classification_report(y_test_s, y_pred_s,
                             target_names=le.classes_))

# Detail prediksi vs aktual
print("[PERBANDINGAN AKTUAL VS PREDIKSI - SVC]")
hasil_svc = pd.DataFrame({
    'Aktual'  : le.inverse_transform(y_test_s),
    'Prediksi': le.inverse_transform(y_pred_s),
    'Benar?'  : ['✓' if a == p else '✗'
                 for a, p in zip(y_test_s, y_pred_s)]
})
print(hasil_svc.to_string(index=False))

# ===========================================================
# 4. VISUALISASI
# ===========================================================
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Analisis Data Hotel Kabupaten Morowali 2025\n'
             'Regresi Linear & Support Vector Classification',
             fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

# --- Plot 1: Distribusi Total Kamar per Hotel ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df_hotel['Total_Kamar'], bins=15, color='steelblue',
         edgecolor='white', alpha=0.85)
ax1.set_title('Distribusi Total Kamar per Hotel', fontsize=11)
ax1.set_xlabel('Total Kamar')
ax1.set_ylabel('Frekuensi')
ax1.grid(axis='y', alpha=0.3)

# --- Plot 2: Distribusi Kategori Hotel ---
ax2 = fig.add_subplot(gs[0, 1])
kategori_count = df_hotel['Kategori'].value_counts()
colors = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax2.bar(kategori_count.index, kategori_count.values,
               color=colors[:len(kategori_count)], edgecolor='white', alpha=0.9)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(int(bar.get_height())), ha='center', va='bottom', fontsize=11)
ax2.set_title('Distribusi Kategori Hotel', fontsize=11)
ax2.set_xlabel('Kategori')
ax2.set_ylabel('Jumlah Hotel')
ax2.grid(axis='y', alpha=0.3)

# --- Plot 3: Scatter Jumlah Tipe vs Total Kamar ---
ax3 = fig.add_subplot(gs[0, 2])
color_map = {'KECIL': '#2ecc71', 'SEDANG': '#f39c12', 'BESAR': '#e74c3c'}
for kat, grp in df_hotel.groupby('Kategori'):
    ax3.scatter(grp['Jumlah_Tipe_Kamar'], grp['Total_Kamar'],
                label=kat, color=color_map[kat], alpha=0.8, s=60)
ax3.set_title('Tipe Kamar vs Total Kamar', fontsize=11)
ax3.set_xlabel('Jumlah Tipe Kamar')
ax3.set_ylabel('Total Kamar')
ax3.legend(title='Kategori')
ax3.grid(alpha=0.3)

# --- Plot 4: Regresi Linear — Aktual vs Prediksi ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_test_r, y_pred_r, color='royalblue', alpha=0.8,
            edgecolors='white', s=70, label='Prediksi')
lims = [min(y_test_r.min(), y_pred_r.min()) - 1,
        max(y_test_r.max(), y_pred_r.max()) + 1]
ax4.plot(lims, lims, 'r--', linewidth=1.5, label='Garis Ideal')
ax4.set_title(f'Regresi Linear: Aktual vs Prediksi\nR² = {r2_r:.4f}', fontsize=11)
ax4.set_xlabel('Aktual')
ax4.set_ylabel('Prediksi')
ax4.legend()
ax4.grid(alpha=0.3)

# --- Plot 5: Residual Regresi Linear ---
ax5 = fig.add_subplot(gs[1, 1])
residual = y_pred_r - y_test_r
ax5.bar(range(len(residual)), residual,
        color=['tomato' if r < 0 else 'steelblue' for r in residual],
        alpha=0.85, edgecolor='white')
ax5.axhline(0, color='black', linewidth=1, linestyle='--')
ax5.set_title(f'Residual Regresi Linear\nMAE={mae_r:.2f}, RMSE={rmse_r:.2f}', fontsize=11)
ax5.set_xlabel('Data Uji ke-')
ax5.set_ylabel('Residual (Prediksi − Aktual)')
ax5.grid(axis='y', alpha=0.3)

# --- Plot 6: Confusion Matrix SVC ---
ax6 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(y_test_s, y_pred_s)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=le.classes_)
disp.plot(ax=ax6, colorbar=False, cmap='Blues')
ax6.set_title('Confusion Matrix SVC', fontsize=11)

plt.savefig('analisis_hotel_morowali.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("\n[INFO] Visualisasi disimpan: analisis_hotel_morowali.png")
plt.close()

print("\n" + "=" * 60)
print("  SELESAI")
print("=" * 60)
