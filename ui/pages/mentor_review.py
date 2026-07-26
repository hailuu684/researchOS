import gradio as gr

def render_mentor_page():
    gr.Markdown("## 🧑‍🏫 Mentor Review Gate")
    gr.Markdown("Mentor kiểm tra các submission đã qua sơ duyệt của AI, quyết định accept, yêu cầu sửa, hoặc can thiệp nếu sinh viên không hiểu bài[cite: 1].")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Thông tin Submission")
            gr.Markdown("**Student:** Quan Manh \n**Task:** Run baseline PointTransformer \n**AI Pre-check:** ✅ Format Pass, ✅ Reproducibility Pass.")
            gr.File(label="Artifacts đính kèm")
            gr.Textbox(label="Student's Understanding Explanation", lines=3, interactive=False, value="Em thấy baseline model gặp overfit nhanh vì tập validation bị rò rỉ dữ liệu (leakage).")
            
        with gr.Column():
            gr.Markdown("### Quyết định của Mentor")
            radio_decision = gr.Radio(
                choices=[
                    "Pass learning (Đạt mục tiêu học tập)", 
                    "Accepted research evidence (Đưa vào PI-ready repo)", 
                    "Revise artifact (Yêu cầu sửa code/report)", 
                    "Mentor intervention (Sinh viên hổng kiến thức nền)"
                ],
                label="Review Decision[cite: 1]"
            )
            txt_feedback = gr.Textbox(label="Feedback cho sinh viên", lines=3)
            btn_submit_review = gr.Button("Xác nhận & Gửi Feedback", variant="primary")
            out_msg = gr.Markdown()
            
            def mock_review(decision, feedback):
                return f"Đã lưu quyết định: **{decision}** và thông báo cho sinh viên."
            
            btn_submit_review.click(fn=mock_review, inputs=[radio_decision, txt_feedback], outputs=out_msg)