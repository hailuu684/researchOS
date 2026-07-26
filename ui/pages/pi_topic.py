import gradio as gr
import pandas as pd
from agents.topic_agents import analyze_topic_and_assign

def render_topic_page():
    gr.Markdown("## 🎯 Khởi tạo Topic & Phân công tự động (AI Agent)")
    gr.Markdown("PI nhập một topic. LLM sẽ tự động bóc tách thành các task nhỏ và map với profile của các members trong lab.")
    
    with gr.Row():
        txt_topic = gr.Textbox(
            label="Research Topic", 
            placeholder="VD: Nghiên cứu phương pháp Topological Knowledge Distillation cho Point Cloud...",
            scale=4
        )
        btn_analyze = gr.Button("Phân tích & Lên Task", variant="primary", scale=1)
    
    # UI Elements (Ẩn đi cho đến khi có dữ liệu)
    out_charter = gr.Markdown(visible=False)
    gr.Markdown("### 📋 Bảng Phân Công Nhiệm Vụ (Task Assignment)")
    table_tasks = gr.Dataframe(interactive=False, visible=False, wrap=True)
    
    # Hàm này đóng vai trò chốt chặn trước khi gọi xuống Agent
    def handle_analyze(topic):
        # 1. Chặn gọi API nếu input rỗng
        if not topic or not topic.strip():
            return gr.update(value="⚠️ Vui lòng nhập topic!", visible=True), gr.update(visible=False)
            
        try:
            # 2. CHỈ GỌI API KHI ĐÃ QUA CHỐT CHẶN NÀY
            roadmap_data = analyze_topic_and_assign(topic)
            
            # Xử lý format Markdown cho Charter
            charter_md = f"**Project Charter:**\n\n{roadmap_data.project_charter}"
            
            # Xử lý Pydantic object thành List of Dictionaries cho Pandas
            task_list = []
            for task in roadmap_data.tasks:
                task_list.append({
                    "Task Name": task.task_name,
                    "Mức độ": f"{task.difficulty} - {task.criticality}",
                    "Người thực hiện": task.assigned_member,
                    "Lý do phân công": task.assignment_reason,
                    "Chi tiết": task.description
                })
            
            df_tasks = pd.DataFrame(task_list)
            
            # Trả dữ liệu lên giao diện và hiển thị chúng ra
            return gr.update(value=charter_md, visible=True), gr.update(value=df_tasks, visible=True)
            
        except Exception as e:
            # Catch mọi lỗi mạng/parse json để ứng dụng không bị crash
            return gr.update(value=f"❌ Lỗi khi gọi API: {str(e)}", visible=True), gr.update(visible=False)

    # Đăng ký sự kiện Click của Gradio
    btn_analyze.click(
        fn=handle_analyze, 
        inputs=txt_topic, 
        outputs=[out_charter, table_tasks]
    )