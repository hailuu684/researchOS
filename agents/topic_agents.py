import os
import json
from openai import OpenAI
from agents.contracts import TopicRoadmap # Giả định bạn đã import từ file chứa Pydantic model ở trên
from researchos_backend.modal_llm import ModalOpenAIChat
from researchos_backend.local_llm import LocalLLMChat
from database.users import user_profile
from dotenv import load_dotenv

load_dotenv()

# Mock database: Profile của các thành viên trong lab (Đa dạng level, chuyên môn)


def analyze_topic_and_assign(topic_description: str) -> TopicRoadmap:
    """
    Agent phân tích topic lớn và tự động gán task cho các thành viên trong lab.
    """
   
    local_max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS"))

    profiles_str = json.dumps(user_profile, ensure_ascii=False, indent=2)
    
    # ENHANCED SYSTEM PROMPT
    system_prompt = f"""
    Bạn là một PI (Principal Investigator) xuất sắc quản lý một phòng Lab nghiên cứu AI. 
    Nhiệm vụ của bạn là nhận một Topic từ user, lên roadmap và chia thành các tasks nhỏ (Learning-and-Research Packages), sau đó phân công cho các thành viên dựa trên năng lực.

    --- ĐỊNH NGHĨA THANG ĐIỂM (RUBRIC) ---
    [Độ Khó - Difficulty]
    D1: Rất dễ. Công việc tay chân, lặp đi lặp lại (gán nhãn, chạy script có sẵn).
    D2: Cơ bản. Yêu cầu kiến thức nền tảng (chạy baseline, EDA cơ bản, format tài liệu).
    D3: Trung bình. Cần hiểu biết chuyên môn hẹp (setup pipeline, fine-tune model nhỏ, code app cơ bản).
    D4: Khó. Yêu cầu tùy chỉnh thuật toán, giải quyết bug hệ thống, đọc hiểu paper phức tạp, deploy scale lớn.
    D5: Rất khó/Nghiên cứu sâu. Thiết kế kiến trúc mới, chứng minh toán học, viết paper nộp hội nghị top-tier.

    [Độ Quan Trọng - Criticality]
    C1: Thấp. Các task râu ria, bổ sung, không làm chậm tiến độ chính.
    C2: Trung bình. Quan trọng nhưng có thể chờ, có workaround nếu fail.
    C3: Cao. Task thuộc critical path, ảnh hưởng trực tiếp tới các task khác.
    C4: Blocker. Nhiệm vụ sống còn của dự án, sai sót sẽ làm hỏng toàn bộ project.

    --- DANH SÁCH THÀNH VIÊN LAB ---
    {profiles_str}
    
    --- THUẬT TOÁN PHÂN CÔNG (BẮT BUỘC TUÂN THỦ 100%) ---
    Với mỗi task bạn tạo ra, hãy thực hiện đánh giá trong đầu theo các bước:
    Bước 1: Xác định mức D (D1-D5) và mức C (C1-C4) khách quan dựa trên định nghĩa trên.
    Bước 2: Lọc ra các ứng viên có kỹ năng (skills) phù hợp với nội dung task.
    Bước 3: Từ danh sách ở Bước 2, LOẠI BỎ những người có max_difficulty < D của task, hoặc max_criticality < C của task. (Ví dụ: Task D4 thì không thể giao cho người max_difficulty D3).
    Bước 4: Chọn người phù hợp nhất (khớp skill và đáp ứng đủ năng lực).
    Bước 5: Nếu Bước 4 không còn ai, BẮT BUỘC ghi vào assigned_member chính xác chuỗi: "chưa tìm thấy ứng viên phù hợp cho task này". KHÔNG ĐƯỢC ÉP BUỘC PHÂN CÔNG VƯỢT QUÁ NĂNG LỰC.

    Mục tiêu: Đảm bảo tiến độ dự án thực tế, không giao phó rủi ro cho người thiếu năng lực.
    """

    user_prompt = f"Hãy phân tích và lên roadmap cho topic sau: {topic_description}"
    raw_json_dict = None

    # try:
    #     client = ModalOpenAIChat()

    #     raw_json_dict = client.complete_json(system=system_prompt, user=user_prompt)
    
    # except Exception as api_error:
    #     print(f"⚠️ Gọi API thất bại: {api_error}")
        
    #     local_model_name = os.getenv("LOCAL_LLM_MODEL", "Local Model")
    #     print(f"🔄 Đang chuyển sang Local Fallback Model ({local_model_name})... (Max tokens: {local_max_tokens})")

    #     try:
    #         local_client = LocalLLMChat()
    #         raw_json_dict = local_client.complete_json(
    #             system=system_prompt, 
    #             user=user_prompt,
    #             max_new_tokens=local_max_tokens # Truyền tham số max tokens cho model local
    #         )
    #         print("✅ Xử lý bằng Local Model thành công.")
            
    #     except Exception as local_error:
    #         raise RuntimeError(f"Cả API và Local Model đều thất bại. Lỗi Local: {local_error}")


    local_client = LocalLLMChat()
    raw_json_dict = local_client.complete_json(
        system=system_prompt, 
        user=user_prompt,
        max_new_tokens=local_max_tokens # Truyền tham số max tokens cho model local
    )
    
    return TopicRoadmap(**raw_json_dict)