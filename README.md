# 🎮 League of Legends Maç Kazanma Tahmin Modeli

Bu proje, Gazi Üniversitesi Bilişim Enstitüsü Yönetim Bilişim Sistemleri Tezsiz Yüksek Lisans programı kapsamında dönem projesi olarak geliştirilmiştir.

## 📌 Proje Hakkında

Amaç: League of Legends (LoL) oyununa ait verileri analiz ederek, bir takımın maçı kazanıp kazanamayacağını tahmin eden bir **makine öğrenmesi modeli** geliştirmektir.

Bu çalışmada oyuncuların oyun içi performans verileri kullanılarak üç farklı algoritma ile sınıflandırma modelleri oluşturulmuştur:

- **Lojistik Regresyon**
- **Karar Ağacı**
- **XGBoost**

## 📂 Proje Dosyaları

- `lol_kaynak_kod.py`: Modelleme sürecini, veri analizi ve performans karşılaştırmasını içeren Python kodları.
- `lol_matches.csv`: League of Legends oyununa ait analiz için kullanılan veri seti.

## 📊 Kullanılan Özellikler

Modelleme için kullanılan başlıca özellikler:

- Öldürme / Ölme / Asist sayıları (KDA)
- Toplam altın, verilen/alınan hasar
- Dakika başına minyon skoru (CS)
- Görüş skoru (vision_score)
- Şampiyon, pozisyon, oyun modu gibi kategorik değişkenler (encode edilmiştir)

## 🧠 Modellerin Performans Karşılaştırması

| Model             | Accuracy | Precision | Recall | F1-Score | AUC  | MCC  |
|------------------|----------|-----------|--------|----------|------|------|
| Lojistik Regresyon | **0.8795** | 0.8919    | 0.8462 | 0.8684   | 0.93 | 0.76 |
| XGBoost           | 0.7952   | 0.7895    | 0.7692 | 0.7792   | 0.90 | 0.59 |
| Karar Ağacı       | 0.7470   | 0.7368    | 0.7179 | 0.7273   | 0.75 | 0.49 |

## ⚙️ Nasıl Çalıştırılır?

1. Gerekli Python kütüphanelerini kurun:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
