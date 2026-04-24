import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile

# --- পেজ কনফিগারেশন ---
st.set_page_config(page_title="Handwriting Grid Extractor", layout="wide")

def process_grid_fixed(uploaded_image, cols_count=7, rows_count=9):
    # ইমেজটিকে OpenCV ফরমেটে রূপান্তর
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ১. গ্রিডের বাইরের বর্ডার খুঁজে বের করা
    # এটি পুরো চারকোনা এরিয়াটাকে ডিটেক্ট করবে
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # সবচেয়ে বড় কন্টুরটি হলো আমাদের মেইন গ্রিড
    largest_contour = max(contours, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(largest_contour)

    # ২. গাণিতিকভাবে প্রতিটি বক্সের সাইজ বের করা
    cell_w = gw / cols_count
    cell_h = gh / rows_count

    cropped_images = []

    # ৩. রো এবং কলাম অনুযায়ী লুপ চালিয়ে ক্রপ করা
    for r in range(rows_count):
        for c in range(cols_count):
            # প্রতিটি বক্সের কোঅর্ডিনেট হিসেব করা
            x1 = int(gx + (c * cell_w))
            y1 = int(gy + (r * cell_h))
            x2 = int(x1 + cell_w)
            y2 = int(y1 + cell_h)

            # মূল বক্স থেকে ক্রপ করা
            cell = img[y1:y2, x1:x2]

            # ৪. আউটলাইন রিমুভ করার জন্য 'ইনডোর প্যাডিং' (খুবই গুরুত্বপূর্ণ)
            # আমরা বক্সের চারপাশ থেকে ১০-১২% এরিয়া বাদ দিয়ে দিবো যাতে বর্ডার না আসে
            h_pad = int((y2 - y1) * 0.12) 
            w_pad = int((x2 - x1) * 0.12)
            
            # বর্ডার ছাড়া ফাইনাল ক্রপ
            final_crop = cell[h_pad:-h_pad, w_pad:-w_pad]
            cropped_images.append(final_crop)
            
    return cropped_images

# --- UI ডিজাইন ---
st.title("📝 Handwriting Dataset Creator")
st.write("এই অ্যাপটি আপনার গ্রিড থেকে অটোমেটিক **৬৩টি (৭x৯)** ক্যারেক্টার আলাদা করবে এবং বর্ডার রিমুভ করবে।")

uploaded_file = st.file_uploader("আপনার গ্রিড ইমেজটি আপলোড দিন", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', width=300)
    
    if st.button('এক্সট্রাক্ট শুরু করুন 🚀'):
        with st.spinner('প্রসেসিং হচ্ছে...'):
            uploaded_file.seek(0)
            crops = process_grid_fixed(uploaded_file)
            
            if len(crops) == 63:
                st.success("সফলভাবে ৬৩টি বক্স খুঁজে পাওয়া গেছে এবং আউটলাইন রিমুভ করা হয়েছে!")
                
                # রেজাল্ট প্রিভিউ
                st.subheader("আউটপুট প্রিভিউ (বর্ডার ছাড়া):")
                grid_cols = st.columns(7)
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for i, crop in enumerate(crops):
                        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        
                        # ডিসপ্লে
                        grid_cols[i % 7].image(pil_img, use_container_width=True)
                        
                        # জিপ ফাইলে সেভ (PNG ফরমেটে যাতে কোয়ালিটি ভালো থাকে)
                        buf = io.BytesIO()
                        pil_img.save(buf, format='PNG')
                        zip_file.writestr(f"char_{i+1:02d}.png", buf.getvalue())

                st.divider()
                st.download_button(
                    label="📥 সব ছবি জিপ ফাইলে ডাউনলোড করুন",
                    data=zip_buffer.getvalue(),
                    file_name="handwriting_dataset.zip",
                    mime="application/zip"
                )
            else:
                st.error(f"দুঃখিত, গ্রিড শনাক্তকরণে সমস্যা হয়েছে। পাওয়া গেছে {len(crops)} টি বক্স।")