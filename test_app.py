"""research_os_demo/
│
├── app.py                    # File chạy chính (Gom các page Gradio lại bằng Tab)
├── requirements.txt          # Khai báo thư viện (gradio, pydantic, openai, v.v.)
├── .env                      # Lưu API Keys (OPENAI_API_KEY, v.v.)
│
├── ui/                       # LỚP GIAO DIỆN (Frontend)
│   ├── __init__.py
│   ├── components.py         # Các UI block dùng chung (vd: Form nộp bài, Bảng Rubric)
│   └── pages/                # Các trang theo Role như đã thiết kế
│       ├── 01_pi_topic.py
│       ├── 02_pi_dashboard.py
│       ├── 03_student_task.py
│       └── 04_mentor_review.py
│
├── agents/                   # LỚP XỬ LÝ AI (Backend Logic)
│   ├── __init__.py
│   ├── contracts.py          # (Quan trọng) Pydantic schemas định nghĩa Input/Output chuẩn của Agent
│   ├── topic_agent.py        # Logic phân tích Prior-art & chia Roadmap
│   ├── tutor_agent.py        # Logic tạo câu hỏi Understanding Gate
│   └── review_agent.py       # Logic tự động chấm điểm Format & Reproducibility
│
├── database/                 # LỚP DỮ LIỆU GIẢ LẬP (Mock DB)
│   ├── db_handler.py         # Script đọc/ghi dữ liệu từ file JSON
│   ├── mock_graph.json       # Giả lập Claim-Evidence Graph thay cho Neo4j
│   ├── mock_users.json       # Giả lập profile sinh viên/PI (quy định Role/Năng lực)
│   └── packages.json         # Lưu các Learning-and-Research Package đã tạo
│
└── evidence_repo/            # KHO LƯU TRỮ ARTIFACTS (File cứng)
    ├── pending/              # Các task_report.md, file code sinh viên vừa nộp
    ├── revised/              # Các file bị Mentor đánh rớt cần sửa lại
    └── accepted/             # Accepted research evidence sẵn sàng để ghép paper
"""

import gradio as gr

# Import các hàm render UI từ thư mục pages
from ui.pages.pi_topic import render_topic_page
from ui.pages.pi_dashboard import render_dashboard_page
from ui.pages.student_task import render_student_page
from ui.pages.mentor_review import render_mentor_page
from ui.pages.student_list import render_student_list_page

# Cấu hình giao diện tổng thể
with gr.Blocks(title="Multi-Agent Research OS") as demo:
    gr.Markdown(
        """
        # 🚀 Multi-Agent Research Training & Supervision OS
        Hệ thống multi-agent hỗ trợ sinh viên học nghiên cứu khoa học, giảm tải cho PI và duy trì chuẩn publication-oriented.
        """
    )
    
    # Sử dụng Tabs để chia phân hệ (Role-based views)
    with gr.Tabs():
        with gr.Tab("1. Topic & Novelty (PI)"):
            render_topic_page()
            
        with gr.Tab("2. PI Dashboard"):
            render_dashboard_page()
            
        with gr.Tab("3. Student Workspace"):
            render_student_page()
            
        with gr.Tab("4. Mentor Review"):
            render_mentor_page()

        with gr.Tab("5. Quản lý Lab (Users)"):
            render_student_list_page()

demo.launch(share=True, server_port=7860, theme=gr.themes.Soft())
