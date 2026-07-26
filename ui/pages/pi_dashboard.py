import gradio as gr
import pandas as pd

def render_dashboard_page():
    gr.Markdown("## 📊 PI Dashboard & Evidence Repository")
    gr.Markdown("Quản lý tiến độ tổng thể và các bằng chứng nghiên cứu (Claim-Evidence Matrix).")
    
    with gr.Tabs():
        with gr.Tab("Triage Queue (Cần PI duyệt)"):
            gr.Markdown("Các artifact quan trọng (Criticality C3-C4) ảnh hưởng đến main claim cần PI ra quyết định cuối cùng[cite: 1].")
            mock_queue = pd.DataFrame({
                "Package ID": ["M4", "E6"],
                "Student": ["Team 4", "Team 5"],
                "Artifact": ["LLM Planner Controller", "Ablation Tables (Robustness)"],
                "Mentor Note": ["Code chạy ổn, cần PI check logic tool-use.", "Baseline fair, metrics khớp."]
            })
            gr.Dataframe(value=mock_queue, interactive=False)
            gr.Button("Vào chi tiết duyệt bài", variant="primary")
            
        with gr.Tab("Claim-Evidence Matrix"):
            gr.Markdown("Theo dõi claim nào đã có bằng chứng, claim nào còn thiếu[cite: 1].")
            mock_matrix = pd.DataFrame({
                "Scientific Claim": ["Topology loss improves semantic boundary", "Geometry tools reduce hallucination"],
                "Required Evidence": ["Ablation on boundary metrics", "Failure analysis on metric reasoning"],
                "Status": ["✅ Accepted", "❌ Missing Evidence"],
                "Linked Artifact": ["ablation_table.csv", "N/A"]
            })
            gr.Dataframe(value=mock_matrix, interactive=False)