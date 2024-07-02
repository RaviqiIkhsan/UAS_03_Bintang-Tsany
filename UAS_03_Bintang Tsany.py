import cv2
import numpy as np
import pytesseract
import datetime
from prettytable import PrettyTable

pytesseract.pytesseract.tesseract_cmd = r'D:\VSCode\tesseract-ocr\tesseract.exe'

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def detect_license_plate(image):
    preprocessed = preprocess_image(image)
    
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    possible_plates = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2 <= aspect_ratio <= 5 and w > 60 and h > 20:
                possible_plates.append((x, y, w, h))
    
    if not possible_plates:
        return None, None, None
    
    # Get the largest possible plate
    x, y, w, h = max(possible_plates, key=lambda x: x[2] * x[3])
    
    plate_img = image[y:y+h, x:x+w]
    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(plate_thresh, config=config)
    
    return text.strip(), (x, y, w, h), plate_thresh

def identify_region(plate_number):
    region_codes = {
        'A': 'Banten', 'B': 'Jakarta', 'D': 'Jawa Barat', 'E': 'Cirebon',
        'F': 'Bogor', 'G': 'Pekalongan', 'H': 'Semarang', 'K': 'Pati',
        'L': 'Surabaya', 'M': 'Madura', 'N': 'Malang', 'P': 'Banyumas',
        'R': 'Bali', 'S': 'Bojonegoro', 'T': 'Purwakarta', 'W': 'Sidoarjo',
        'Z': 'Sumatera Barat'
    }
    if plate_number and len(plate_number) > 0:
        region_code = plate_number[0]
        return region_codes.get(region_code, 'Wilayah tidak dikenali')
    return 'Wilayah tidak dikenali'

cap = cv2.VideoCapture(0)

# Inisialisasi tabel hasil deteksi
table = PrettyTable()
table.field_names = ["No.", "Waktu", "Nomor Plat", "Asal Daerah"]
detection_count = 0

# Buat file untuk menyimpan hasil
with open('hasil_deteksi_plat.txt', 'a') as file:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        plate_text, plate_box, plate_thresh = detect_license_plate(frame)
        
        if plate_thresh is not None:
            cv2.imshow('Plate', plate_thresh)
        
        if plate_text:
            region = identify_region(plate_text)
            if plate_box:
                x, y, w, h = plate_box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Plat: {plate_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Asal: {region}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Catat hasil ke file dan tabel
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{timestamp} - Plat: {plate_text}, Asal: {region}\n")
            file.flush()  # Memastikan data langsung ditulis ke file
            
            # Tambahkan hasil ke tabel
            detection_count += 1
            table.add_row([detection_count, timestamp, plate_text, region])
        else:
            cv2.putText(frame, "Tidak terdeteksi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('License Plate Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Cetak tabel hasil deteksi
print(table)
