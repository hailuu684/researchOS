from pydantic import BaseModel, Field
from typing import List, Literal

class AssignedTask(BaseModel):
    task_name: str = Field(
        description="Tên công việc ngắn gọn, BẮT ĐẦU BẰNG MỘT ĐỘNG TỪ HÀNH ĐỘNG (ví dụ: Thiết kế, Viết, Kiểm thử, Triển khai)."
    )
    description: str = Field(
        description="Mô tả chi tiết: Bao gồm input cần thiết, các bước thực hiện chính và output mong đợi của task này."
    )
    difficulty: Literal["D1", "D2", "D3", "D4", "D5"] = Field(
        description="Mức độ khó của task. Chỉ chọn từ D1 (Rất dễ/Cơ bản) đến D5 (Cực kỳ phức tạp/Nghiên cứu sâu)."
    )
    criticality: Literal["C1", "C2", "C3", "C4"] = Field(
        description="Mức độ quan trọng đối với tiến độ dự án. Chỉ chọn từ C1 (Thấp/Bổ sung) đến C4 (Blocker/Bắt buộc phải có ngay)."
    )
    assigned_member: str = Field(
        description="Tên thành viên được phân công (CHỈ ĐƯỢC CHỌN TỪ DANH SÁCH THÀNH VIÊN ĐÃ CUNG CẤP TRONG CONTEXT). NẾU KHÔNG CÓ AI PHÙ HỢP, BẮT BUỘC TRẢ VỀ CHUỖI: 'chưa tìm thấy ứng viên phù hợp cho task này'"
    )
    assignment_reason: str = Field(
        description="Phân tích sự phù hợp giữa kỹ năng của ứng viên (hoặc sự thiếu hụt kỹ năng của team) với yêu cầu kỹ thuật của task."
    )

class TopicRoadmap(BaseModel):
    project_charter: str = Field(
        description="Roadmap tổng thể (từ 3-5 câu): Nêu rõ mục tiêu cuối cùng, các giai đoạn chính (phases), và key deliverables của topic này."
    )
    tasks: List[AssignedTask] = Field(
        description="Danh sách các task theo trình tự thời gian hợp lý để hoàn thành roadmap. Các task không được trùng lặp phạm vi (MECE)."
    )