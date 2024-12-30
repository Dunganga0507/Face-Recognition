import cv2
from deepface import DeepFace
import os
import time

# Yüz tanımlamalarını kaydetmek için listeleri oluşturuyoruz
known_face_names = ["KeremAkgoz"]  # İsimleri buraya ekliyoruz
known_face_images = ["kerem_akgoz.jpg"]  # Resimleri buraya ekliyoruz

# Bilinen yüzlerin bulunduğu dizin
db_path = "known_faces"  # Bilinen yüzlerin veritabanı dizini
os.makedirs(db_path, exist_ok=True)  # Dizin yoksa oluştur

# Bilinen yüzleri işleme
for img_path in known_face_images:
    if not os.path.exists(os.path.join(db_path, img_path)):
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(db_path, img_path), img)

# Kamera akışını başlat
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Kamera açılamadı.")
    exit()

frame_count = 0
last_identity = None  # Son tanınan kimlik
last_update_time = time.time()  # Son güncelleme zamanı

while True:
    # Kameradan kare yakala
    ret, frame = video_capture.read()
    
    if not ret:  # Eğer görüntü alınamazsa döngüyü sonlandır
        print("Kameradan görüntü alınamadı.")
        break

    # Her 5. karede bir yüz tanı
    if frame_count % 24 == 0:
        try:
            # BGR'den RGB'ye dönüştür
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Yüz tanımayı yap
            results = DeepFace.find(frame_rgb, db_path=db_path, model_name="Facenet", enforce_detection=True)
            print(results)  # Sonuçları kontrol et
        except Exception as e:
            print(f"Yüz tanıma hatası: {e}")
            results = []  # Hata durumunda boş sonuç
            
        # Tanınan yüzleri işle
        if isinstance(results, list) and len(results) > 0:  # Eğer sonuçlar bir listeyse ve en az bir eleman varsa
            for result in results:
                if "identity" in result and not result["identity"].empty:  # Sadece identity anahtarının varlığını ve boş olmadığını kontrol et
                    identity_value = result["identity"].iloc[0]  # Series'ten ilk değeri al
                    name = os.path.basename(identity_value)  # Yüz tanıdığında ismi al
                    
                    # Doğru ismi kullanmak için indexleme
                    if name in known_face_images:
                        index = known_face_images.index(name)
                        last_identity = known_face_names[index]  # Son tanınan ismi güncelle
                        last_update_time = time.time()  # Güncelleme zamanını güncelle
                    break  # İlk sonucu bulduktan sonra döngüyü kır
        else:
            # Eğer tanınan yüz yoksa, "Yabancı" olarak etiketle
            last_identity = "Bilinmiyor"
            last_update_time = time.time()  # Güncelleme zamanını güncelle

    # Tanınan kimlik ile metni görüntüye yazdır
    if last_identity is not None and time.time() - last_update_time < 1:  # Eğer tanıma yapıldıysa ve 1 saniyeden az zaman geçtiyse
        if last_identity == "Bilinmiyor":
            cv2.putText(frame, last_identity, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)  # Kırmızı renkte "Yabancı" yazdır
        else:
            cv2.putText(frame, last_identity, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)  # Yeşil renkte isim yazdır

    # Kameradan alınan görüntüyü göster
    cv2.imshow('Video', frame)
    
    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1  # Frame sayısını artır

# Kamerayı ve pencereleri serbest bırak
video_capture.release()
cv2.destroyAllWindows()
