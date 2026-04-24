import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import io
import zipfile

st.set_page_config(page_title="High-Res Precision Dataset Tool", layout="wide")

def enhance_image_quality(img):
    """ইমেজের শার্পনেস এবং রেজোলিউশন সামান্য উন্নত করার ফাংশন"""
    # সামান্য শার্পনিং ফিল্টার প্রয়োগ
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def get_perfect_intact_crop(warped_img, x1, y1, x2, y2):
    cell = warped_img[y1:y2, x1:x2]
    if cell.size == 0: return None
    
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ২০% বর্ডার মাস্কিং
    h_c, w_c = thresh.shape
    py, px = int(h_c * 0.20), int(w_c * 0.20)
    thresh[:py, :] = 0; thresh[-py:, :] = 0; thresh[:, :px] = 0; thresh[:, -px:] = 0

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        all_pts = np.concatenate(contours)
        ix, iy, iw, ih = cv2.boundingRect(all_pts)
        
        # ১:১ সাইজ + ২৫ পিক্সেল সেফটি মার্জিন
        side = max(iw, ih) + 25 
        
        center_x = x1 + ix + iw // 2
        center_y = y1 + iy + ih // 2
        
        fx1, fy1 = int(center_x - side // 2), int(cy1 := center_y - side // 2)
        fx2, fy2 = fx1 + side, fy1 + side
        
        img_h, img_w = warped_img.shape[:2]
        if fx1 < 0: fx2 -= fx1; fx1 = 0
        if fy1 < 0: fy2 -= fy1; fy1 = 0
        if fx2 > img_w: fx1 -= (fx2 - img_w); fx2 = img_w
        if fy2 > img_h: fy1 -= (fy2 - img_h); fy2 = img_h

        # হাই-কোয়ালিটি ক্রপ
        final_crop = warped_img[fy1:fy2, fx1:fx2]
        
        # রেজোলিউশন এনহ্যান্সমেন্ট: যদি ক্রপ খুব ছোট হয় তবে এটিকে লিনিয়ারলি বড় করা
        target_display_size = 256 # উচ্চ রেজোলিউশনের জন্য ২৫৬x২৫৬ স্ট্যান্ডার্ড ধরা হয়েছে
        if final_crop.shape[0] < target_display_size:
            final_crop = cv2.resize(final_crop, (target_display_size, target_display_size), 
                                    interpolation=cv2.INTER_CUBIC)
            
        return enhance_image_quality(final_crop)
    return None

def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = np.intp(cv2.boxPoints(rect))
    pts = box.reshape(4, 2)
    rect_pts = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect_pts[0] = pts[np.argmin(s)]; rect_pts[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect_pts[1] = pts[np.argmin(diff)]; rect_pts[3] = pts[np.argmax(diff)]
    
    w_max = int(max(np.linalg.norm(rect_pts[2]-rect_pts[3]), np.linalg.norm(rect_pts[1]-rect_pts[0])))
    h_max = int(max(np.linalg.norm(rect_pts[1]-rect_pts[2]), np.linalg.norm(rect_pts[0]-rect_pts[3])))
    
    dst = np.array([[0,0], [w_max-1,0], [w_max-1,h_max-1], [0,h_max-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect_pts, dst)
    # উচ্চ রেজোলিউশন বজায় রাখতে ওয়ার্প করার সময় INTER_CUBIC ব্যবহার
    warped = cv2.warpPerspective(img, M, (w_max, h_max), flags=cv2.INTER_CUBIC)
    
    rows, cols = 9, 7
    ch, cw = h_max / rows, w_max / cols
    
    results = []
    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(c * cw), int(r * ch)
            x2, y2 = int((c + 1) * cw), int((r + 1) * ch)
            crop = get_perfect_intact_crop(warped, x1, y1, x2, y2)
            if crop is not None:
                results.append(crop)
            else:
                results.append(np.ones((256, 256, 3), dtype=np.uint8) * 255)
    return results

# --- UI ---
st.title("🛡️ High-Resolution Zero-Border Extractor")
st.markdown("এই ভার্সনটি **INTER_CUBIC Interpolation** এবং **Sharpening Filter** ব্যবহার করে উচ্চ রেজোলিউশন নিশ্চিত করে।")

file = st.file_uploader("ইমেজ আপলোড করুন", type=['jpg', 'png', 'jpeg'])

if file:
    if st.button("Extract High-Res Dataset"):
        with st.spinner('হাই-রেজোলিউশনে প্রসেসিং হচ্ছে...'):
            results = process_image(file)
            if results and len(results) == 63:
                st.success(f"সফলভাবে ৬৩ টি উচ্চমানের ইমেজ পাওয়া গেছে।")
                st_cols = st.columns(7)
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for i, crop in enumerate(results):
                        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        st_cols[i % 7].image(pil_img, use_container_width=True)
                        
                        buf = io.BytesIO()
                        # কোয়ালিটি ৯৫ এবং অপটিমাইজেশন অন রাখা হয়েছে
                        pil_img.save(buf, format='PNG', quality=95, optimize=True)
                        zip_file.writestr(f"highres_char_{i+1:02d}.png", buf.getvalue())
                st.download_button("Download High-Res ZIP", zip_buffer.getvalue(), "highres_dataset.zip")
            else:
                st.error("গ্রিড শনাক্তকরণে সমস্যা হয়েছে।")
