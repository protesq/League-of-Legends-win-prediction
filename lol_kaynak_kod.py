import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, matthews_corrcoef, classification_report
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Veri setini yükle
print("Veri seti yükleniyor...")
df = pd.read_csv('lol_matches.csv')

# Veri keşfi
print("Veri seti boyutu:", df.shape)
print("\nWin dağılımı:")
print(df['win'].value_counts())

# Kategorik değişkenleri encode etme
print("\nKategorik değişkenler encode ediliyor...")
categorical_cols = ['champion', 'position', 'game_mode', 'game_type']
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le

# Numerik sütunları seçme
print("Model için öznitelikler seçiliyor...")
numerical_cols = [
    'game_duration_minutes', 'kills', 'deaths', 'assists', 'kda',
    'double_kills', 'champion_level', 'total_damage_to_champions',
    'total_damage_taken', 'gold_earned', 'total_cs', 'cs_per_minute',
    'vision_score', 'damage_per_minute', 'damage_per_gold', 'kill_participation'
]

# Kategorik sütunları ekle
for col in categorical_cols:
    if f'{col}_encoded' in df.columns:
        numerical_cols.append(f'{col}_encoded')

# Sütunların varlığını kontrol etme ve olmayanları kaldırma
available_cols = [col for col in numerical_cols if col in df.columns]
print(f"Kullanılan öznitelikler: {available_cols}")

X = df[available_cols]
y = df['win'].map({True: 1, False: 0})  # True/False değerlerini 1/0'a çevirme

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Eğitim seti boyutu: {X_train.shape}, Test seti boyutu: {X_test.shape}")

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model değerlendirme fonksiyonu
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\n----- {model_name} eğitiliyor -----")
    # Eğitim ve tahmin
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Olasılık tahminleri (ROC AUC için)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = y_pred
    
    # Karmaşıklık matrisi
    cm = confusion_matrix(y_test, y_pred)
    
    # Değerlendirme metrikleri
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    

    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = None
    
    # Sonuçları yazdırma
    print(f"\n----- {model_name} Sonuçları -----")
    print(f"Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"Kesinlik (Precision): {precision:.4f}")
    print(f"Duyarlılık (Recall): {recall:.4f}")
    print(f"F1-Skoru: {f1:.4f}")
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    
    print("\nKarmaşıklık Matrisi:")
    print(cm)
    
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mcc': mcc
    }

print("\nModeller eğitiliyor ve değerlendiriliyor...")

# 1. Karar Ağacı Modeli
dt_model = DecisionTreeClassifier(random_state=42)
dt_results = evaluate_model(dt_model, X_train_scaled, X_test_scaled, y_train, y_test, "Karar Ağacı")

# 2. Lojistik Regresyon Modeli
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_results = evaluate_model(lr_model, X_train_scaled, X_test_scaled, y_train, y_test, "Lojistik Regresyon")

# 3. XGBoost Modeli
print("\n----- XGBoost eğitiliyor -----")
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_results = evaluate_model(xgb_model, X_train_scaled, X_test_scaled, y_train, y_test, "XGBoost")

# Modellerin karşılaştırması
models = [dt_results, lr_results, xgb_results]
model_names = ["Karar Ağacı", "Lojistik Regresyon", "XGBoost"]

# Karşılaştırma tablosu
comparison_df = pd.DataFrame({
    'Model': model_names,
    'Doğruluk (Accuracy)': [model['accuracy'] for model in models],
    'Kesinlik (Precision)': [model['precision'] for model in models],
    'Duyarlılık (Recall)': [model['recall'] for model in models],
    'F1-Skoru': [model['f1'] for model in models],
    'AUC': [model['auc'] for model in models],
    'MCC': [model['mcc'] for model in models]
})

print("\n----- Model Karşılaştırması -----")
print(comparison_df.to_string())

print("\nEn yüksek başarıya sahip model:")
best_model_idx = comparison_df['Doğruluk (Accuracy)'].argmax()
print(f"{model_names[best_model_idx]} - Doğruluk: {comparison_df['Doğruluk (Accuracy)'].max():.4f}")

# Karar Ağacı öznitelik önemliliği
if hasattr(dt_model, 'feature_importances_'):
    print("\nKarar Ağacı Öznitelik Önemliliği:")
    feature_importance = pd.DataFrame({
        'Öznitelik': X.columns,
        'Önem': dt_model.feature_importances_
    }).sort_values('Önem', ascending=False)
    print(feature_importance.head(10).to_string())

# XGBoost öznitelik önemliliği
if hasattr(xgb_model, 'feature_importances_'):
    print("\nXGBoost Öznitelik Önemliliği:")
    feature_importance = pd.DataFrame({
        'Öznitelik': X.columns,
        'Önem': xgb_model.feature_importances_
    }).sort_values('Önem', ascending=False)
    print(feature_importance.head(10).to_string()) 

#Karmaşıklık MATRISI Görselleştirme 
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(dt_model, X_test_scaled, y_test, display_labels=["Kaybetti", "Kazandı"], cmap="Blues")
plt.title("Karar Ağaçlar - Karmaşıklık Matrisi")
plt.show()  

ConfusionMatrixDisplay.from_estimator(lr_model, X_test_scaled, y_test, display_labels=["Kaybetti", "Kazandı"], cmap="Blues")
plt.title("Lojistik Regresyon - Karmaşıklık Matrisi")
plt.show()

ConfusionMatrixDisplay.from_estimator(xgb_model, X_test_scaled, y_test, display_labels=["Kaybetti", "Kazandı"], cmap="Blues")
plt.title("XGBoost - Karmaşıklık Matrisi")
plt.show()

# Modellerin karşılaştırması görselleştirmesi
import matplotlib.pyplot as plt

metrics = ['Doğruluk (Accuracy)', 'Kesinlik (Precision)', 'Duyarlılık (Recall)', 'F1-Skoru', 'AUC', 'MCC']

# Her metrik için bar plot oluştur
for metric in metrics:
    plt.figure(figsize=(8, 4))
    plt.bar(comparison_df['Model'], comparison_df[metric], color='skyblue')
    plt.title(f'Modellere Göre {metric}')
    plt.ylabel(metric)
    plt.xlabel('Algoritma')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

#ROC eğrileri
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

# Lojistik Regresyon ROC Eğrisi
if hasattr(lr_model, "predict_proba"):
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    plt.plot(fpr_lr, tpr_lr, label=f'Lojistik Regresyon (AUC = {roc_auc_lr:.2f})')

# Karar Ağacı ROC Eğrisi
if hasattr(dt_model, "predict_proba"):
    y_pred_proba_dt = dt_model.predict_proba(X_test_scaled)[:, 1]
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
    roc_auc_dt = auc(fpr_dt, tpr_dt)
    plt.plot(fpr_dt, tpr_dt, label=f'Karar Ağacı (AUC = {roc_auc_dt:.2f})')

# XGBoost ROC Eğrisi
if hasattr(xgb_model, "predict_proba"):
    y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('Modellerin ROC Eğrileri Karşılaştırması')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


#Lojistik Regresyon Özellik Önem Düzeyleri
importances = lr_model.coef_[0]
features = X.columns
plt.barh(features, importances)
plt.title("Lojistik Regresyon Öznitelik Önem Düzeyleri")
plt.tight_layout()
plt.show()

#XGBoost Özellik Önem Düzeyleri
importances = xgb_model.feature_importances_
features = X.columns
plt.barh(features, importances)
plt.title("XGBoost Öznitelik Önem Düzeyleri")
plt.tight_layout()
plt.show()

#Karar Ağacı Özellik Önem Düzeyleri
importances = dt_model.feature_importances_
features = X.columns
plt.barh(features, importances)
plt.title("Karar Ağacı Öznitelik Önem Düzeyleri")
plt.tight_layout()
plt.show()
