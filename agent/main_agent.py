"""
Main Agent — RAG Agent V1 (Base) và V2 (Optimized).
Dùng context từ golden_set.jsonl để trả lời câu hỏi.

V1 (Base): Trả lời đơn giản, chỉ dùng 1 context, dễ sai
V2 (Optimized): Prompt tốt hơn, dùng nhiều context, xử lý edge cases
"""

import asyncio
import os
import random
import time
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Knowledge Base — Trích xuất từ ChromaDB chunks trong golden_set
# ============================================================
KNOWLEDGE_BASE = {
    # === ACCESS CONTROL SOP ===
    "access_control_sop_0": {
        "doc_id": "access_control_sop",
        "content": "Ghi chú: Tài liệu này trước đây có tên \"Approval Matrix for System Access\".",
        "keywords": ["tên tài liệu", "approval matrix", "system access", "trước đây"]
    },
    "access_control_sop_1": {
        "doc_id": "access_control_sop",
        "content": "Tài liệu này quy định quy trình cấp phép truy cập vào các hệ thống nội bộ của công ty. Áp dụng cho tất cả nhân viên, contractor, và third-party vendor.",
        "keywords": ["quy trình", "cấp phép", "truy cập", "hệ thống nội bộ", "nhân viên", "contractor", "vendor", "đối tượng"]
    },
    "access_control_sop_2": {
        "doc_id": "access_control_sop",
        "content": "Level 1 — Read Only: Áp dụng cho: Tất cả nhân viên mới trong 30 ngày đầu. Phê duyệt: Line Manager. Thời gian xử lý: 1 ngày làm việc. Level 2 — Standard Access: Áp dụng cho: Nhân viên chính thức đã qua thử việc. Phê duyệt: Line Manager + IT Admin. Thời gian xử lý: 2 ngày làm việc. Level 3 — Elevated Access: Áp dụng cho: Team Lead, Senior Engineer, Manager. Phê duyệt: Line Manager + IT Admin + IT Security. Thời gian xử lý: 3 ngày làm việc. Level 4 — Admin Access: Áp dụng cho: DevOps, SRE, IT Admin. Phê duyệt: IT Manager + CISO. Thời gian xử lý: 5 ngày làm việc. Yêu cầu thêm: Training bắt buộc về security policy.",
        "keywords": ["level", "read only", "standard", "elevated", "admin", "phê duyệt", "line manager", "it admin", "it security", "ciso", "thời gian xử lý", "cấp 1", "cấp 2", "cấp 3", "cấp 4", "quyền truy cập"]
    },
    "access_control_sop_3": {
        "doc_id": "access_control_sop",
        "content": "Bước 1: Nhân viên tạo Access Request ticket trên Jira (project IT-ACCESS). Bước 2: Line Manager phê duyệt yêu cầu trong 1 ngày làm việc. Bước 3: IT Admin kiểm tra compliance và cấp quyền. Bước 4: IT Security review với Level 3 và Level 4. Bước 5: Nhân viên nhận thông báo qua email khi quyền được cấp.",
        "keywords": ["bước", "access request", "ticket", "jira", "it-access", "phê duyệt", "compliance", "cấp quyền", "yêu cầu quyền"]
    },
    "access_control_sop_4": {
        "doc_id": "access_control_sop",
        "content": "Escalation chỉ áp dụng khi cần thay đổi quyền hệ thống ngoài quy trình thông thường. Ví dụ: Khẩn cấp trong sự cố P1, cần cấp quyền tạm thời để fix incident. Quy trình escalation khẩn cấp: 1. On-call IT Admin có thể cấp quyền tạm thời (max 24 giờ) sau khi được Tech Lead phê duyệt bằng lời. 2. Sau 24 giờ, phải có ticket chính thức hoặc quyền bị thu hồi tự động. 3. Mọi quyền tạm thời phải được ghi log vào hệ thống Security Audit.",
        "keywords": ["escalation", "khẩn cấp", "p1", "tạm thời", "24 giờ", "incident", "on-call", "thu hồi", "security audit"]
    },
    "access_control_sop_5": {
        "doc_id": "access_control_sop",
        "content": "Quyền phải được thu hồi trong các trường hợp: - Nhân viên nghỉ việc: Thu hồi ngay trong ngày cuối. - Hết hạn contract: Thu hồi đúng ngày hết hạn. - Chuyển bộ phận: Điều chỉnh trong 3 ngày làm việc.",
        "keywords": ["thu hồi", "nghỉ việc", "hết hạn", "contract", "chuyển bộ phận", "ngày cuối", "điều chỉnh"]
    },
    "access_control_sop_6": {
        "doc_id": "access_control_sop",
        "content": "IT Security thực hiện access review mỗi 6 tháng. Mọi bất thường phải được báo cáo lên CISO trong vòng 24 giờ.",
        "keywords": ["access review", "6 tháng", "bất thường", "báo cáo", "ciso", "24 giờ"]
    },
    "access_control_sop_7": {
        "doc_id": "access_control_sop",
        "content": "Ticket system: Jira (project IT-ACCESS). IAM system: Okta. Audit log: Splunk. Email: it-access@company.internal",
        "keywords": ["jira", "it-access", "okta", "splunk", "it-access@company.internal", "email", "ticket", "hệ thống"]
    },

    # === HR LEAVE POLICY ===
    "hr_leave_policy_0": {
        "doc_id": "hr_leave_policy",
        "content": "1.1 Nghỉ phép năm (Annual Leave): Số ngày: 12 ngày/năm cho nhân viên dưới 3 năm kinh nghiệm. 15 ngày/năm cho nhân viên từ 3-5 năm. 18 ngày/năm cho nhân viên trên 5 năm. Chuyển năm sau: Tối đa 5 ngày phép năm chưa dùng được chuyển sang năm tiếp theo. 1.2 Nghỉ ốm (Sick Leave): 10 ngày/năm có trả lương. Yêu cầu: Thông báo cho Line Manager trước 9:00 sáng ngày nghỉ. Nếu nghỉ trên 3 ngày liên tiếp: Cần giấy tờ y tế từ bệnh viện. 1.3 Nghỉ thai sản: 6 tháng theo quy định Luật Lao động. 1.4 Nghỉ lễ tết: Theo lịch nghỉ lễ quốc gia do HR công bố hàng năm vào tháng 12.",
        "keywords": ["nghỉ phép", "annual leave", "12 ngày", "15 ngày", "18 ngày", "nghỉ ốm", "sick leave", "10 ngày", "thai sản", "6 tháng", "chuyển năm", "giấy tờ y tế", "kinh nghiệm"]
    },
    "hr_leave_policy_1": {
        "doc_id": "hr_leave_policy",
        "content": "Bước 1: Nhân viên gửi yêu cầu nghỉ phép qua hệ thống HR Portal ít nhất 3 ngày làm việc trước ngày nghỉ. Bước 2: Line Manager phê duyệt hoặc từ chối trong vòng 1 ngày làm việc. Bước 3: Nhân viên nhận thông báo qua email sau khi được phê duyệt. Trường hợp khẩn cấp: Có thể gửi yêu cầu muộn hơn nhưng phải được Line Manager đồng ý qua tin nhắn trực tiếp.",
        "keywords": ["nghỉ phép", "hr portal", "3 ngày", "phê duyệt", "khẩn cấp", "tin nhắn", "yêu cầu", "trước"]
    },
    "hr_leave_policy_2": {
        "doc_id": "hr_leave_policy",
        "content": "3.1 Điều kiện làm thêm: Làm thêm giờ phải được Line Manager phê duyệt trước bằng văn bản. 3.2 Hệ số lương làm thêm: Ngày thường: 150% lương giờ tiêu chuẩn. Ngày cuối tuần: 200%. Ngày lễ: 300%.",
        "keywords": ["làm thêm", "overtime", "150%", "200%", "300%", "lương", "hệ số", "ngày lễ", "cuối tuần", "văn bản"]
    },
    "hr_leave_policy_3": {
        "doc_id": "hr_leave_policy",
        "content": "4.1 Điều kiện remote: Nhân viên sau probation period có thể làm remote tối đa 2 ngày/tuần. Team Lead phải phê duyệt lịch remote qua HR Portal. Ngày onsite bắt buộc: Thứ 3 và Thứ 5. 4.2 Yêu cầu kỹ thuật khi remote: Kết nối VPN bắt buộc. Camera bật trong các cuộc họp team.",
        "keywords": ["remote", "2 ngày", "probation", "team lead", "onsite", "vpn", "camera", "hr portal"]
    },
    "hr_leave_policy_4": {
        "doc_id": "hr_leave_policy",
        "content": "Email: hr@company.internal. Hotline: ext. 2000. HR Portal: https://hr.company.internal. Giờ làm việc: Thứ 2 - Thứ 6, 8:30 - 17:30",
        "keywords": ["hr", "email", "hotline", "ext. 2000", "hr portal", "giờ làm việc", "8:30", "17:30", "liên hệ", "phòng nhân sự"]
    },

    # === IT HELPDESK FAQ ===
    "it_helpdesk_faq_0": {
        "doc_id": "it_helpdesk_faq",
        "content": "Q: Tôi quên mật khẩu, phải làm gì? A: Truy cập https://sso.company.internal/reset hoặc liên hệ Helpdesk qua ext. 9000. Mật khẩu mới sẽ được gửi qua email công ty trong vòng 5 phút. Q: Tài khoản bị khóa sau bao nhiêu lần đăng nhập sai? A: 5 lần liên tiếp. Để mở khóa, liên hệ IT Helpdesk hoặc tự reset qua portal SSO. Q: Mật khẩu cần thay đổi định kỳ không? A: Có. Mỗi 90 ngày. Hệ thống nhắc nhở 7 ngày trước khi hết hạn.",
        "keywords": ["mật khẩu", "quên", "reset", "sso", "ext. 9000", "khóa", "5 lần", "90 ngày", "đăng nhập"]
    },
    "it_helpdesk_faq_1": {
        "doc_id": "it_helpdesk_faq",
        "content": "Q: Phần mềm VPN nào công ty dùng? A: Cisco AnyConnect. Download tại https://vpn.company.internal/download. Q: VPN có giới hạn số thiết bị không? A: Mỗi tài khoản được kết nối VPN trên tối đa 2 thiết bị cùng lúc.",
        "keywords": ["vpn", "cisco", "anyconnect", "download", "thiết bị", "2 thiết bị", "kết nối"]
    },
    "it_helpdesk_faq_2": {
        "doc_id": "it_helpdesk_faq",
        "content": "Q: Tôi cần cài phần mềm mới, phải làm gì? A: Gửi yêu cầu qua Jira project IT-SOFTWARE. Line Manager phải phê duyệt trước khi IT cài đặt. Q: Ai chịu trách nhiệm gia hạn license phần mềm? A: IT Procurement team. Nhắc nhở 30 ngày trước khi hết hạn.",
        "keywords": ["phần mềm", "cài đặt", "it-software", "jira", "license", "gia hạn", "procurement"]
    },
    "it_helpdesk_faq_3": {
        "doc_id": "it_helpdesk_faq",
        "content": "Q: Laptop mới được cấp sau bao lâu khi vào công ty? A: Laptop được cấp trong ngày onboarding đầu tiên. Q: Laptop bị hỏng phải báo cáo như thế nào? A: Tạo ticket P2 hoặc P3 tùy mức độ. Mang thiết bị đến IT Room (tầng 3).",
        "keywords": ["laptop", "cấp", "onboarding", "hỏng", "ticket", "p2", "p3", "it room", "tầng 3"]
    },
    "it_helpdesk_faq_4": {
        "doc_id": "it_helpdesk_faq",
        "content": "Q: Hộp thư đến đầy, phải làm gì? A: Xóa email cũ hoặc yêu cầu tăng dung lượng qua ticket IT-ACCESS. Dung lượng tiêu chuẩn là 50GB. Q: Tôi không nhận được email từ bên ngoài? A: Kiểm tra thư mục Spam trước. Nếu vẫn không có, tạo ticket P2 kèm địa chỉ email gửi và thời gian gửi.",
        "keywords": ["email", "hộp thư", "đầy", "50gb", "dung lượng", "spam", "không nhận được"]
    },
    "it_helpdesk_faq_5": {
        "doc_id": "it_helpdesk_faq",
        "content": "Hotline: ext. 9000 (8:00 - 18:00, Thứ 2 - Thứ 6). Email: helpdesk@company.internal. Jira: project IT-SUPPORT. Slack: #it-helpdesk. Emergency (ngoài giờ): ext. 9999",
        "keywords": ["hotline", "ext. 9000", "helpdesk", "it-support", "slack", "emergency", "ext. 9999", "ngoài giờ", "cuối tuần"]
    },

    # === REFUND POLICY ===
    "policy_refund_v4_0": {
        "doc_id": "policy_refund_v4",
        "content": "Chính sách này áp dụng cho tất cả các đơn hàng được đặt trên hệ thống nội bộ kể từ ngày 01/02/2026. Các đơn hàng đặt trước ngày có hiệu lực sẽ áp dụng theo chính sách hoàn tiền phiên bản 3.",
        "keywords": ["chính sách", "hoàn tiền", "01/02/2026", "hiệu lực", "phiên bản 3", "đơn hàng"]
    },
    "policy_refund_v4_1": {
        "doc_id": "policy_refund_v4",
        "content": "Khách hàng được quyền yêu cầu hoàn tiền khi: Sản phẩm bị lỗi do nhà sản xuất, không phải do người dùng. Yêu cầu được gửi trong vòng 7 ngày làm việc kể từ thời điểm xác nhận đơn hàng. Đơn hàng chưa được sử dụng hoặc chưa bị mở seal.",
        "keywords": ["hoàn tiền", "lỗi", "nhà sản xuất", "7 ngày", "seal", "điều kiện", "người dùng"]
    },
    "policy_refund_v4_2": {
        "doc_id": "policy_refund_v4",
        "content": "Ngoại lệ không được hoàn tiền: Sản phẩm thuộc danh mục hàng kỹ thuật số (license key, subscription). Đơn hàng đã áp dụng mã giảm giá đặc biệt Flash Sale. Sản phẩm đã được kích hoạt hoặc đăng ký tài khoản.",
        "keywords": ["ngoại lệ", "kỹ thuật số", "license", "subscription", "flash sale", "kích hoạt", "không được hoàn"]
    },
    "policy_refund_v4_3": {
        "doc_id": "policy_refund_v4",
        "content": "Bước 1: Khách hàng gửi yêu cầu qua hệ thống ticket nội bộ với category \"Refund Request\". Bước 2: CS Agent xem xét trong vòng 1 ngày làm việc. Bước 3: Nếu đủ điều kiện, chuyển sang Finance Team. Bước 4: Finance Team xử lý trong 3-5 ngày làm việc.",
        "keywords": ["hoàn tiền", "ticket", "refund request", "cs agent", "finance team", "1 ngày", "3-5 ngày", "quy trình"]
    },
    "policy_refund_v4_4": {
        "doc_id": "policy_refund_v4",
        "content": "Hoàn tiền qua phương thức thanh toán gốc: áp dụng trong 100% trường hợp đủ điều kiện. Hoàn tiền qua credit nội bộ (store credit): khách hàng có thể chọn nhận store credit thay thế với giá trị 110% so với số tiền hoàn.",
        "keywords": ["phương thức", "thanh toán gốc", "store credit", "110%", "hoàn tiền"]
    },
    "policy_refund_v4_5": {
        "doc_id": "policy_refund_v4",
        "content": "Email: cs-refund@company.internal. Hotline nội bộ: ext. 1234. Giờ làm việc: Thứ 2 - Thứ 6, 8:00 - 17:30",
        "keywords": ["cs-refund", "email", "ext. 1234", "hotline", "giờ làm việc", "liên hệ"]
    }
}


class MainAgent:
    """
    RAG Agent hỗ trợ nội bộ công ty.
    V1 (Base): Retrieval đơn giản, prompt cơ bản, dễ sai
    V2 (Optimized): Retrieval tốt hơn, prompt chi tiết, xử lý edge cases
    """

    def __init__(self, version: str = "v1"):
        self.version = version
        self.name = f"InternalSupportAgent-{version}"
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("AGENT_MODEL", "gpt-4o-mini")
        self.use_real_api = bool(self.api_key and not self.api_key.startswith("sk-your"))
        self.total_tokens = 0

    def _retrieve(self, question: str, top_k: int = None) -> tuple:
        """
        Keyword-based retrieval.
        V1: top_k=1, chỉ keyword matching đơn giản
        V2: top_k=3, matching tốt hơn (keyword + bigram)
        """
        if top_k is None:
            top_k = 1 if self.version == "v1" else 3

        question_lower = question.lower()
        q_words = set(question_lower.split())
        scores = {}

        for chunk_id, chunk in KNOWLEDGE_BASE.items():
            score = 0
            for keyword in chunk["keywords"]:
                kw_lower = keyword.lower()
                if kw_lower in question_lower:
                    score += 2  # exact substring match
                elif any(w in kw_lower for w in q_words if len(w) > 2):
                    score += 1  # partial word match

            # V2 bonus: bigram matching — check consecutive word pairs
            if self.version == "v2" and score > 0:
                content_lower = chunk["content"].lower()
                q_bigrams = [f"{question_lower.split()[i]} {question_lower.split()[i+1]}"
                             for i in range(len(question_lower.split()) - 1)]
                for bigram in q_bigrams:
                    if bigram in content_lower:
                        score += 1

            scores[chunk_id] = score

        # Sort by score descending
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        retrieved_ids = []
        contexts = []
        for chunk_id, score in sorted_chunks[:top_k]:
            retrieved_ids.append(chunk_id)
            contexts.append(KNOWLEDGE_BASE[chunk_id]["content"])

        return retrieved_ids, contexts

    def _generate_v1(self, question: str, contexts: List[str]) -> str:
        """
        V1: Trả lời đơn giản, lấy 1 câu đầu từ context.
        Nhược điểm cố ý:
        - Không xử lý adversarial
        - Trả lời cụt, thiếu chi tiết
        - Không check nếu context không liên quan
        """
        if not question.strip():
            return "Vui lòng nhập câu hỏi."

        if contexts:
            # Chỉ lấy 1-2 câu đầu tiên của context
            first_ctx = contexts[0]
            sentences = [s.strip() for s in first_ctx.replace(". ", ".\n").split("\n") if s.strip()]
            # Trả lời cụt — chỉ lấy câu đầu
            answer = sentences[0] if sentences else first_ctx[:200]
            return f"Theo tài liệu: {answer}"
        else:
            return f"Tôi không tìm thấy thông tin liên quan đến câu hỏi: {question}"

    def _generate_v2(self, question: str, contexts: List[str]) -> str:
        """
        V2: Trả lời chi tiết, tổng hợp từ nhiều context.
        Cải tiến:
        - Xử lý adversarial (từ chối lịch sự)
        - Trả lời đầy đủ, có cấu trúc
        - Kiểm tra context có liên quan không
        """
        question_lower = question.lower()

        # Edge case: empty question
        if not question.strip():
            return "Xin chào! Tôi là trợ lý hỗ trợ nội bộ. Bạn cần giúp gì về quy trình công ty?"

        # Adversarial detection
        adversarial_patterns = [
            "bỏ qua", "ignore", "system prompt", "giả vờ", "pretend",
            "api key", "viết thơ", "hack", "jailbreak", "password của người khác"
        ]
        if any(p in question_lower for p in adversarial_patterns):
            return ("Xin lỗi, tôi chỉ có thể hỗ trợ các câu hỏi liên quan đến quy trình "
                    "và chính sách nội bộ công ty. Yêu cầu này nằm ngoài phạm vi hỗ trợ của tôi.")

        # Out-of-scope detection
        out_of_scope = ["thời tiết", "bóng đá", "chính trị", "tỷ giá", "chứng khoán"]
        if any(p in question_lower for p in out_of_scope):
            return ("Câu hỏi này nằm ngoài phạm vi hỗ trợ của tôi. Tôi chỉ có thể giúp "
                    "về các quy trình nội bộ: Access Control, HR, IT Helpdesk, và Refund Policy.")

        if contexts:
            # Tổng hợp từ nhiều context, trích xuất câu liên quan
            relevant_parts = []
            for ctx in contexts:
                sentences = [s.strip() for s in ctx.replace(". ", ".\n").split("\n") if s.strip()]
                for sent in sentences:
                    sent_lower = sent.lower()
                    # Check if sentence is relevant to question
                    q_words = [w for w in question_lower.split() if len(w) > 2]
                    if any(w in sent_lower for w in q_words):
                        relevant_parts.append(sent)

            if relevant_parts:
                # Remove duplicates, keep order
                seen = set()
                unique_parts = []
                for p in relevant_parts:
                    if p not in seen:
                        seen.add(p)
                        unique_parts.append(p)

                answer = ". ".join(unique_parts[:5])
                if not answer.endswith("."):
                    answer += "."
                return f"Dựa trên tài liệu nội bộ: {answer}"
            else:
                # Context found but not specifically relevant
                return f"Dựa trên tài liệu: {contexts[0][:300]}"
        else:
            return ("Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở tri thức. "
                    "Vui lòng liên hệ bộ phận liên quan hoặc gửi ticket qua Jira.")

    async def query(self, question: str) -> Dict:
        """
        Main entry point — RAG pipeline.
        """
        start = time.perf_counter()

        # Step 1: Retrieval
        retrieved_ids, contexts = self._retrieve(question)

        # Step 2: Generation
        if self.use_real_api:
            answer = await self._generate_with_api(question, contexts)
        else:
            if self.version == "v1":
                answer = self._generate_v1(question, contexts)
            else:
                answer = self._generate_v2(question, contexts)
            await asyncio.sleep(random.uniform(0.02, 0.08))  # Simulate latency

        latency = time.perf_counter() - start

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": self.model if self.use_real_api else f"simulated-{self.version}",
                "tokens_used": 150,
                "latency": round(latency, 3),
                "version": self.version
            }
        }

    async def _generate_with_api(self, question: str, contexts: List[str]) -> str:
        """Gọi OpenAI API thật."""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key)

        system = (
            "Bạn là trợ lý hỗ trợ nội bộ công ty. Trả lời dựa trên context được cung cấp. "
            "Nếu không tìm thấy thông tin, nói rõ và hướng dẫn liên hệ bộ phận phù hợp. "
            "Trả lời ngắn gọn, chính xác, chuyên nghiệp bằng tiếng Việt."
        )
        if self.version == "v2":
            system += (" Từ chối các yêu cầu ngoài phạm vi hỗ trợ. "
                       "Không tiết lộ thông tin nội bộ nhạy cảm.")

        context_text = "\n---\n".join(contexts) if contexts else "Không có context."

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nCâu hỏi: {question}"}
                ],
                temperature=0.3,
                max_tokens=400
            )
            self.total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Lỗi khi xử lý yêu cầu. Vui lòng thử lại. ({str(e)[:50]})"


if __name__ == "__main__":
    async def test():
        for ver in ["v1", "v2"]:
            agent = MainAgent(version=ver)
            print(f"\n=== Agent {ver.upper()} ===")
            questions = [
                "Ai là người phê duyệt quyền truy cập cấp 1?",
                "Nhân viên có dưới 3 năm kinh nghiệm được nghỉ phép năm bao nhiêu ngày?",
                "Tôi cần làm gì nếu quên mật khẩu?",
                "Thời tiết hôm nay thế nào?",
            ]
            for q in questions:
                resp = await agent.query(q)
                print(f"Q: {q}")
                print(f"A: {resp['answer'][:120]}")
                print(f"Retrieved: {resp['retrieved_ids']}")
                print()
    asyncio.run(test())
