import streamlit as st

st.set_page_config(page_title="Trang Kế Tiếp", layout="centered")

st.title("🎉 Bạn đã chuyển trang thành công!")
st.write("Đây là nội dung của trang thứ hai.")

# Tạo một nút để quay lại trang chủ nếu muốn
if st.button("⬅️ Quay lại Trang Chủ"):
    st.switch_page("app.py")
