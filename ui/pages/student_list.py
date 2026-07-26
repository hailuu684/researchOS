import gradio as gr
from database.users import user_profile

def render_student_list_page():
    gr.Markdown("## 👥 Danh sách Thành viên Lab")
    gr.Markdown("Quản lý hồ sơ năng lực của các sinh viên và nghiên cứu viên để AI tự động phân việc.")

    # Duyệt qua database để vẽ UI động
    for user in user_profile:
        # gr.Group() giúp tạo một cái khung (card) bọc xung quanh thông tin
        with gr.Group():
            with gr.Row():
                # CỘT TRÁI: Hiển thị Avatar (min_width nhỏ để khung ảnh gọn gàng)
                with gr.Column(scale=1, min_width=120):
                    gr.Image(
                        value=user.get("avatar"),
                        show_label=False,           # Ẩn chữ label
                        interactive=False,          # Không cho người dùng up ảnh đè lên
                        container=False,            # Xóa viền xám mặc định của Gradio Image
                        width=100,
                        height=100
                    )
                
                # CỘT PHẢI: Hiển thị Tên, Tuổi và Thông tin
                with gr.Column(scale=5):
                    # Xử lý text kỹ năng
                    skills_text = ", ".join(user.get("skills", []))
                    
                    # Markdown hỗ trợ render HTML cơ bản, giúp chữ đẹp hơn
                    info_html = f"""
                    <h3 style='margin-top: 0; margin-bottom: 5px;'>{user['name']}</h3>
                    <p style='margin: 2px 0;'><b>Tuổi:</b> {user.get('age', 'N/A')} | <b>Level:</b> {user.get('level', 'N/A')}</p>
                    <p style='margin: 2px 0;'><b>Giới hạn năng lực:</b> Độ khó {user.get('max_difficulty')} - Quan trọng {user.get('max_criticality')}</p>
                    <p style='margin: 2px 0;'><b>Kỹ năng:</b> {skills_text}</p>
                    <p style='margin: 2px 0;'><b>Tự chủ:</b> {user.get('autonomy')}</p>
                    """
                    gr.HTML(info_html)