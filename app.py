import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile

st.set_page_config(page_title="Final Precision Dataset Tool", layout="wide")

def get_perfect_intact_crop(warped_img, x1, y1, x2, y2):
    """
    Kaes-এর চূড়ান্ত রিকোয়ারমেন্ট: 
    ১. বর্ডারের একটি দাগও আসবে না (Zero Tolerance).
    ২. কন্টেন্ট অক্ষত থাকবে (Intact).
    ৩. ১:১ রেশিও এবং ন্যাচারাল টেক্সচার.
    """
    # নির্দিষ্ট সেলের অংশটুকু নেওয়া
    cell = warped_img[y1:y2, x1:x2]
    if cell.size == 0: return None
    
    # গ্রেস্কেল এবং থ্রেশহোল্ড (কালি ডিটেকশন)
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # শক্তিশালী বর্ডার ইরেজার: 
    # আমরা বক্সের চারপাশ থেকে ২০% এরিয়াকে মাস্কে পুরোপুরি মুছে দেব। 
    # এটি নিশ্চিত করবে যে বর্ডারের কোনো মোটা দাগও যেন 'কন্টেন্ট' হিসেবে ডিটেক্ট না হয়।
    h_c, w_c = thresh.shape
    py = int(h_c * 0.20)
    px = int(w_c * 0.20)
    thresh[:py, :] = 0; thresh[-py:, :] = 0
    thresh[:, :px] = 0; thresh[:, -px:] = 0

    # অক্ষরের কন্টুর খোঁজা
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # সব কন্টুর মিলিয়ে অক্ষরের বাউন্ডারি বের করা
        all_pts = np.concatenate(contours)
        ix, iy, iw, ih = cv2.boundingRect(all_pts)
        
        # ১:১ সাইজ নির্ধারণ: অক্ষরের সাইজ + ২৫ পিক্সেল সেফটি মার্জিন
        # এটি নিশ্চিত করবে যে অক্ষরটি কাটা পড়বে না
        side = max(iw, ih) + 25 
        
        # অক্ষরের কেন্দ্রের সাপেক্ষে ক্রপ ফ্রেম তৈরি
        center_x = x1 + ix + iw // 2
        center_y = y1 + iy + ih // 2
        
        fx1 = int(center_x - side // 2)
        fy1 = int(center_y - side // 2)
        fx2 = fx1 + side
        fy2 = fy1 + side
        
        # বাউন্ডারি প্রোটেকশন ও শিফটিং (সাদা প্যাডিং ছাড়াই পেপারের ভেতরে রাখা)
        img_h, img_w = warped_img.shape[:2]
        
        # ফ্রেম যদি ইমেজের বা গ্রিড বর্ডারের বাইরে যেতে চায়, তবে তাকে ভেতরে ঠেলে দেওয়া
        if fx1 < 0: fx2 -= fx1; fx1 = 0
        if fy1 < 0: fy2 -= fy1; fy1 = 0
        if fx2 > img_w: fx1 -= (fx2 - img_w); fx2 = img_w
        if fy2 > img_h: fy1 -= (fy2 - img_h); fy2 = img_h

        # অরিজিনাল পেপার টেক্সচার থেকে ক্রপ
        final_crop = warped_img[fy1:fy2, fx1:fx2]
        
        # একদম নিখুঁত ১:১ নিশ্চিত করা
        if final_crop.shape[0] != final_crop.shape[1] and final_crop.size > 0:
            d = min(final_crop.shape[:2])
            final_crop = final_crop[:d, :d]
            
        return final_crop
    return None

def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # গ্রিড ডিটেকশন ও পারসপেক্টিভ ঠিক করা
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = np.intp(cv2.boxPoints(rect))
    
    # কোণা সাজানো
    pts = box.reshape(4, 2)
    rect_pts = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect_pts[0] = pts[np.argmin(s)]; rect_pts[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect_pts[1] = pts[np.argmin(diff)]; rect_pts[3] = pts[np.argmax(diff)]
    
    w_max = int(max(np.linalg.norm(rect_pts[2]-rect_pts[3]), np.linalg.norm(rect_pts[1]-rect_pts[0])))
    h_max = int(max(np.linalg.norm(rect_pts[1]-rect_pts[2]), np.linalg.norm(rect_pts[0]-rect_pts[3])))
    
    dst = np.array([[0,0], [w_max-1,0], [w_max-1,h_max-1], [0,h_max-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect_pts, dst)
    warped = cv2.warpPerspective(img, M, (w_max, h_max))
    
    # ৯x৭ গ্রিডে ভাগ করে এক্সট্রাক্ট করা
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
                # খালি বক্সের জন্য সাদা ছবি
                results.append(np.ones((100, 100, 3), dtype=np.uint8) * 255)
                
    return results

# --- UI ---
st.title("🛡️ Zero-Border Precision Extractor")
st.markdown("এই ভার্সনটি ২০% বর্ডার মাস্কিং ব্যবহার করে, যা গ্রিড লাইনের শেষ দাগটিও মুছে ফেলবে।")

file = st.file_uploader("ইমেজ আপলোড করুন", type=['jpg', 'png', 'jpeg'])

if file:
    if st.button("Extract Dataset Now"):
        with st.spinner('নিখুঁতভাবে প্রসেসিং হচ্ছে...'):
            results = process_image(file)
            if results and len(results) == 63:
                st.success(f"সফলভাবে ৬৩ টি ইমেজ পাওয়া গেছে।")
                st_cols = st.columns(7)
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for i, crop in enumerate(results):
                        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        st_cols[i % 7].image(img_rgb, use_container_width=True)
                        buf = io.BytesIO()
                        Image.fromarray(img_rgb).save(buf, format='PNG')
                        zip_file.writestr(f"char_{i+1:02d}.png", buf.getvalue())
                st.download_button("Download Perfect ZIP", zip_buffer.getvalue(), "perfect_dataset.zip")
            else:
                st.error("গ্রিড শনাক্তকরণে সমস্যা হয়েছে।")
