import streamlit as st
import joblib
import numpy as np
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# Memuat semua model dan scaler dari file
models_and_scaler = joblib.load('prediksi_IQ.pkl')

# Mendapatkan model dan scaler dari dictionary
scaler = models_and_scaler['scaler']
model_IQ = models_and_scaler['model_IQ']
model_description = models_and_scaler['model_description']

# Fungsi prediksi
def predict_IQ_and_description(raw_score):
    # Transformasi input skor mentah menggunakan scaler
    scaled_score = scaler.transform([[raw_score]])
    
    # Prediksi nilai IQ dan deskripsi
    predicted_IQ = model_IQ.predict(scaled_score)[0]
    predicted_description = model_description.predict(scaled_score)[0]
    
    return predicted_IQ, predicted_description


# Menambahkan tips per kategori deskripsi
def get_tips(description):
    tips = {
        "Di Bawah Rata-Rata": [
            "🔴 Cobalah lebih banyak latihan soal untuk meningkatkan kemampuan kognitif.",
            "📖 Belajar secara rutin dapat membantu meningkatkan skor Anda.",
            "🧠 Cobalah untuk melakukan latihan otak seperti teka-teki dan puzzle."
        ],
        "Rata-Rata": [
            "📊 Teruskan usaha Anda! Anda berada di jalur yang baik.",
            "💡 Lakukan latihan untuk memperkuat konsep yang sudah dikuasai.",
            "🔍 Tingkatkan fokus dan strategi dalam memecahkan soal."
        ],
        "Di Atas Rata-Rata": [
            "🌟 Sangat baik! Cobalah untuk terus mengasah keterampilan Anda lebih lanjut.",
            "🚀 Tantang diri Anda dengan soal-soal yang lebih sulit.",
            "🧩 Jangan berhenti di sini, kembangkan kemampuan Anda dengan berbagai metode."
        ]
    }
    return tips.get(description, ["💭 Tidak ada tips yang tersedia."])

# Ikon untuk kategori deskripsi IQ
def get_icon(description):
    icons = {
        "Di Bawah Rata-Rata": "🔻",
        "Rata-Rata": "📊",
        "Di Atas Rata-Rata": "🌟",
    }
    return icons.get(description, "⚙️")

def get_bubble_icon(description):
    icons = {
        "Di Bawah Rata-Rata": "😢",
        "Rata-Rata": "🙂",
        "Di Atas Rata-Rata": "😃",
    }
    return icons.get(description, "⚙️")

# pdf
def create_iq_report_pdf(nama, nim, raw_score, predicted_IQ, predicted_description):
    # Buat nama file PDF
    pdf_path = f"IQ_Report_{nama}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=landscape(A4))
    width, height = landscape(A4) 

    # Tambahkan Background Gambar
    try:
        background_image = "background sertifikat.png"  
        c.drawImage(background_image, 0, 0, width=width, height=height)  # Gambar penuh halaman
    except:
        c.setFillColorRGB(1, 1, 1)  # Background putih jika gambar tidak ditemukan
        c.rect(0, 0, width, height, fill=1)

    # Atur warna dan font
    c.setFillColor((255/255, 99/255, 71/255))  # Warna tomato
    
    # Judul Laporan
    c.setFont("Helvetica-Bold", 40)
    c.drawCentredString(width/2, height-120, "IQ Test Report")
    
    # Informasi Pribadi
    c.setFont("Helvetica-Bold", 22)
    c.setFillColorRGB(0, 0, 0)  # Hitam untuk teks
    c.drawCentredString(width/2, height-180, f"{nama}")
    c.drawCentredString(width/2, height-210, f"NIM. {nim}")

    # Memisahkan data ke dalam dua kolom: label (kolom kiri) dan hasil (kolom kanan)
    data = [
    ["Skor Mentah", str(raw_score)],  
    ["Prediksi IQ", f"{predicted_IQ:.2f}"],
    ["Keterangan", predicted_description]  
    ]

    # Tentukan tinggi baris untuk setiap baris
    row_heights = [33, 33, 33]

    # Membuat tabel dengan dua kolom
    table = Table(data, colWidths=[220, 280], rowHeights=row_heights)
    table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (0, -1), colors.Color(1, 1, 0.8)),  # Kolom kiri diberi warna latar belakang light yellow
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Warna teks header
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Teks di tengah semua sel
    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),  # Font tabel
    ('FONTSIZE', (0, 0), (-1, -1), 18),  # Ukuran font tabel
    ('BOTTOMPADDING', (0, 0), (-1, -1), 14),  # Padding sel
    ('WORDWRAP', (0, 0), (-1, -1), True),  # Pembungkusan teks
    ('BOX', (0, 0), (-1, -1), 2, colors.black),  # Border luar tabel
    ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Garis grid tabel
    ]))

    # Menghitung posisi tabel agar berada di tengah
    table_width = sum([220, 280])       # Lebar tabel (total lebar kedua kolom)
    table_height = sum(row_heights)   # Tinggi tabel
    table_x = (width - table_width) / 2  # Posisi horizontal tabel (di tengah)

    # Mengatur posisi vertikal tabel lebih tinggi
    table_y = height - 330  # Sesuaikan nilai ini untuk menaikkan tabel lebih tinggi

    # Gambar tabel pada PDF
    table.wrapOn(c, table_width, table_height)
    table.drawOn(c, table_x, table_y)  # Posisi tabel (x, y)
    
    # Tips Berdasarkan Kategori
    tips = get_tips(predicted_description)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-380, "Tips Pengembangan")
    
    c.setFont("Helvetica", 14)
    for i, tip in enumerate(tips):
        c.drawCentredString(width/2, height-(410 + i*30), tip)

    # Tambahkan Catatan Motivasi di Footer
    c.setFont("Helvetica-Oblique", 14)
    c.setFillColorRGB(1.0, 0.549, 0.0)  # Warna orange
    c.drawCentredString(width/2, 85, "Terus Kembangkan Potensi Anda!")
    
    # Simpan PDF
    c.save()
    return pdf_path

# streamlit app
def main():
    # Desain aplikasi Streamlit
    st.set_page_config(page_title="Test IQ", page_icon="🧠", layout="centered")

    # CSS styling
    st.markdown("""
        <style>
        body {
            background: linear-gradient(90deg, rgba(255,99,71,1) 0%, rgba(255,154,0,1) 100%);
        }
        h1 {
            font-family: 'Arial', sans-serif;
            color: white;
            font-size: 40px;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        h2 {
            font-family: 'Arial', sans-serif;
            color: white;
            text-align: center;
        }
        h4 {
            font-family: 'Arial';
            color: #FF6347;
            text-align: center;
        }
        p {
            font-family: 'Arial', sans-serif;
            color: black,white, orange;
            text-align: center;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .stButton>button {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(90deg, rgba(255,69,0,1) 0%, rgba(255,154,0,1) 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 20px;
            padding: 15px 40px;
            transition: 0.3s ease-in-out;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
            width: 300px;
            display: block;
            margin: 0 auto;
        }
        .stButton>button:hover {
            transform: scale(1.1);
            box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.4);
        }
        table {
            margin: 5px;
            width: 80%;
            border-collapse: collapse;
            font-size: 25px;
            color: #FF6347;
            width: 100%; /* Membuat tabel memenuhi lebar kontainer */
            max-width: 1200px; /* Menambahkan batas maksimum lebar */
            margin: 0 auto; /* Menjaga tabel tetap di tengah */
        }
        table, th, td {
            border: 2px solid #FF6347;
        }
        th, td {
            padding: 12px;
            text-align: center;
        }
        th {
            background-color: #FF6347;
            color: white;
        }
        .stButton>button.download {
            background: linear-gradient(90deg, rgba(255,69,0,1) 0%, rgba(255,154,0,1) 100%);
            margin-top: 20px;
        }
        h2, h4, p {
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.6);
        }
        .tips-container p {
            color: #333333;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }
        </style>
                
        <style>             /* background */
        [data-testid="stAppViewContainer"] {  
            background-color: #e5e5f7;
            opacity: 0.8;
            background-image:  radial-gradient(#f79645 0.9px, transparent 0.9px), radial-gradient(#f79645 0.9px, #e5e5f7 0.9px);
            background-size: 36px 36px;
            background-position: 0 0,18px 18px;
        }
        </style>
        
        <h1 style="text-align: center; font-size: 50px; color: #FF6347;">🧠 Prediksi IQ</h1> 
        <p style="text-align: center; font-size: 20px; color: #000;">Masukkan skor mentah untuk melihat hasil prediksi IQ dan kategorinya.</p> 

        <style>             /* Styling input box */
        .stNumberInput>div>div>input {
            width: 80px;  /* Lebar input box lebih kecil */
            height: 30px;  /* Tinggi input box lebih kecil */
            font-size: 18px;
            text-align: center;
            border-radius: 5px;
            border: 2px solid #FF6347;
            margin: 0 auto;
        }
        .stNumberInput>div>label {
            font-size: 16px;
            font-weight: bold;
            color: #FF6347;
        }
        </style>
                
        <style>             /* Styling untuk tombol prediksi */
        .stButton>button {
            background-color: #FF6347;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 20px;
            width: 150px; /
            margin: 20px auto;
            display: block;
            transition: 0.3s ease;

        }
        .stButton>button:hover {
            background-color: #FF4500;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
        }
        </style>
                
        <style>             /* Styling untuk tombol download pdf */
        .stDownloadButton > button {
            background-color: #FF6347;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 20px;
            width: 400px; 
            margin: 20px auto;
            display: block;
            transition: 0.3s ease;
        }
        .stDownloadButton > button:hover {
            background-color: #FF4500;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
        }
        </style>
                
        <style>
        .container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
        }
        .bubble-container {
            flex: 1;
            position: relative;
            height: 500px;
            display: flex;
            flex-direction: column;
            justify-content: space-evenly;
            align-items: center;
        }
        .bubble {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 50px;
            animation: float 4s ease-in-out infinite;
        }
        .bubble:nth-child(even) {
            animation-duration: 3.5s;
        }
        .bubble:nth-child(odd) {
            animation-duration: 4.5s;
        }
        @keyframes float {
            0%, 100% {
                transform: translateY(0);
            }
            50% {
                transform: translateY(-20px);
            }
        </style>
    """, unsafe_allow_html=True)

    # Input skor mentah
    raw_score_input = st.number_input("Masukkan Skor Mentah:", min_value=0, max_value=100, step=1, value=50)
    # Tambahkan input nama untuk PDF
    nama = st.text_input("Masukkan Nama Anda:")
    # Tambahkan input NIM untuk PDF
    nim = st.text_input("Masukkan NIM Anda:")
    
    # Tombol untuk melakukan prediksi
    if st.button("Prediksi"):
        if raw_score_input and nama and nim:
            predicted_IQ, predicted_description = predict_IQ_and_description(raw_score_input)


            # Menampilkan hasil prediksi
            icon = get_icon(predicted_description)
            tips = get_tips(predicted_description)
            bubble_icon = get_bubble_icon(predicted_description)

            # Layout untuk gelembung dan tabel
            st.markdown(f"""
            <div class="container">
                <div class="bubble-container">
                    <div class="bubble">{bubble_icon}</div>
                </div>
                <div class="table-container">
                    <h2 style="font-size: 40px; color: #FF6347;">Hasil Prediksi IQ</h2>
                    <table>
                        <tr>
                            <th>Prediksi IQ</th>
                            <th>Kategori</th>
                        </tr>
                        <tr>
                            <td>{predicted_IQ:.2f}</td>
                            <td>{predicted_description} {icon}</td>
                        </tr>
                    </table>
                </div>
                <div class="bubble-container">
                    <div class="bubble">{bubble_icon}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Menampilkan tips per kategori
            st.markdown("<h4 >Tips untuk anda :</h4>", unsafe_allow_html=True)
            st.markdown("<div class='tips-container'>" + "".join([f"<p>{tip}</p>" for tip in tips]) + "</div>", unsafe_allow_html=True)

            # Buat dan tampilkan tombol download PDF
            pdf_path = create_iq_report_pdf(nama, nim, raw_score_input, predicted_IQ, predicted_description)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download Laporan IQ.pdf",
                    data=pdf_file,
                    file_name=f"IQ_Report_{nama}.pdf",
                    mime="application/pdf"
                )

        else:
            st.warning("Silakan masukkan nama, NIM, dan skor mentah terlebih dahulu.")

if __name__ == "__main__":
    main()