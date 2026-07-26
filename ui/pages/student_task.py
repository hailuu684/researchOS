import gradio as gr

def render_student_page():
    gr.Markdown("## 🎓 Student Workspace")
    gr.Markdown("Không gian để sinh viên nhận Learning-and-Research Package, nộp Artifact và thực hiện Understanding Check[cite: 1].")
    
    with gr.Accordion("📦 Package Hiện Tại: Implement Topological Distillation Loss (D3, C3)", open=True):
        gr.Markdown("""
        **Learning Objectives:** Biết cách tích hợp persistent homology vào pipeline distillation.
        **Expected Outputs:** `loss_module.py`, `reproducibility.md`, `results.json`.
        """)
    
    with gr.Tabs():
        with gr.Tab("1. Nộp Artifacts"):
            file_upload = gr.File(label="Upload files (Code, logs, markdown)", file_count="multiple")
            txt_report = gr.Textbox(label="Task Report Summary", lines=4)
            btn_submit_task = gr.Button("Gửi Artifact cho Agent kiểm tra Format")
            out_format_status = gr.Markdown()
            
            def mock_submit(files, report):
                return "✅ **Format Gate Pass**: `artifact_manifest.json` hợp lệ. Vui lòng chuyển sang tab Understanding Gate."
            
            btn_submit_task.click(fn=mock_submit, inputs=[file_upload, txt_report], outputs=out_format_status)
            
        with gr.Tab("2. Understanding Gate"):
            gr.Markdown("⚠️ **Chú ý:** Artifact không được duyệt chỉ vì code chạy được. Bạn phải giải thích được kết quả[cite: 1].")
            
            bot_question = gr.Textbox(
                label="Understanding Examiner Agent", 
                value="Câu hỏi: Việc dùng Wasserstein distance trong loss module này hỗ trợ claim nghiên cứu nào của bài báo? Nếu baseline A cao hơn method của ta ở class thiểu số, bạn giải thích sao?", 
                interactive=False
            )
            txt_answer = gr.Textbox(label="Câu trả lời của bạn", lines=5)
            btn_submit_answer = gr.Button("Nộp câu trả lời", variant="primary")
            
            out_understanding = gr.Markdown()
            
            def mock_understanding(answer):
                if len(answer) > 20:
                    return "⏳ **Agent đánh giá:** Câu trả lời hợp lý. Đã chuyển trạng thái sang `Pass learning`. Đang chờ Mentor duyệt vòng cuối."
                return "❌ **Agent đánh giá:** `Revise explanation`. Bạn chưa giải thích rõ liên kết với claim gốc. Vui lòng đọc lại tài liệu."
                
            btn_submit_answer.click(fn=mock_understanding, inputs=txt_answer, outputs=out_understanding)