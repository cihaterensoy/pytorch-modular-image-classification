# Pytorch Modular Image Classification

Bu çalışma, PyTorch kullanılarak geliştirilmiş, modüler yapıya sahip bir görüntü sınıflandırma projesidir. Projenin temel amacı, derin öğrenme süreçlerini (veri hazırlama, model mimarisi, eğitim ve test) birbirinden bağımsız modüller halinde sunarak sürdürülebilir ve temiz bir kod yapısı oluşturmaktır.

## Proje Hakkında

Uygulama, "DesertClassifier" olarak adlandırılan bir Evrişimli Sinir Ağı (CNN) mimarisi üzerine kuruludur. Veri seti üzerinde otomatik istatistik hesaplaması yaparak görüntüleri normalize eder ve eğitim sürecini modüler bir motor üzerinden yürütür.

## Dosya Yapısı ve İşlevleri

Proje, işlevlerine göre şu dosyalara ayrılmıştır:

* **main.py**: Uygulamanın ana giriş noktasıdır. Hiperparametreleri belirler ve tüm süreci (veri yükleme, model oluşturma, eğitim) başlatır.
* **model_creation.py**: Projede kullanılan CNN mimarisinin (DesertClassifier) tanımlandığı sınıftır.
* **setup_data.py**: Ham görüntü verilerini okur, DataLoader nesnelerini oluşturur ve veriyi eğitime hazır hale getirir.
* **training_testing_engine.py**: Eğitim (`train_step`) ve test (`test_step`) döngülerini içeren, modelin öğrenme sürecini yöneten motordur.
* **utils.py**: Modelin kaydedilmesi, yüklenmesi ve veri setine ait ortalama (mean) ve standart sapma (std) değerlerinin hesaplanması gibi yardımcı fonksiyonları içerir.
* **load_model_make_prediction.py**: Eğitilmiş bir model dosyasını yükleyerek belirli bir görsel üzerinde sınıflandırma tahmini yürütür.
* **requirements.txt**: Projenin çalışması için gerekli olan kütüphanelerin listesidir.

## Kurulum

Projeyi çalıştırmak için öncelikle gerekli kütüphaneleri yüklemeniz gerekmektedir:

```bash
pip install -r requirements.txt
```

## Kullanım

### Modelin Eğitilmesi

Eğitim sürecini başlatmak için ana dosyayı çalıştırmanız yeterlidir:

```bash
python main.py
```

### Tahmin Yürütme

Eğitilmiş bir modeli kullanarak bir görseli test etmek için:

```bash
python load_model_make_prediction.py
```

## Teknik Özellikler

* **Modülerlik:** Her bileşen kendi sorumluluk alanına sahiptir, bu sayede kodun test edilmesi ve genişletilmesi kolaydır.
* **Özelleştirilmiş Eğitim:** Hazır kütüphanelerin sunduğu kapalı döngüler yerine, eğitim süreci adım adım kontrol edilebilir.
* **Veri Normalizasyonu:** Veri setinin kendi istatistikleri üzerinden normalizasyon yapılarak model başarımı optimize edilmiştir.
