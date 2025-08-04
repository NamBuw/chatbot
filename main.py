import sounddevice as sd
import numpy as np
from silero_vad import load_silero_vad
import whisper
import torch
import queue
import time
import sys
import select
import noisereduce as nr
import os
import json
import google.generativeai as genai
import re
from datetime import datetime
import threading
import unicodedata
# ==================== CONFIGURATION ====================

API_KEY = "YOUR_API_KEY_HERE"
genai.configure(api_key=API_KEY)

USER_INFO_FILE = 'user_info.json'
TOPICS_DIR = 'topics'
file_lock = threading.Lock()

# ==================== ENHANCED TOPICS CONFIGURATION ====================

TOPICS = {
    'que_huong': {
        'name': '🏠 Quê hương và hoài niệm',
        'description': 'Ký ức về quê nhà, món ăn truyền thống, ca dao tục ngữ, âm nhạc quê hương',
        'folder': 'que_huong',
        'emoji': '🏠',
        'keywords': ['quê', 'nhà', 'cơm', 'phở', 'làng', 'sông', 'cây', 'hoài niệm'],
        'greeting': 'Chào bác! Cháu rất vui được nghe bác kể về quê hương. Bác muốn chia sẻ gì về quê nhà?'
    },
    'gia_dinh': {
        'name': '👨‍👩‍👧‍👦 Gia đình',
        'description': 'Liên lạc với người thân, truyền dạy văn hóa cho con cháu, kể chuyện gia đình',
        'folder': 'gia_dinh',
        'emoji': '👨‍👩‍👧‍👦',
        'keywords': ['con', 'cháu', 'vợ', 'chồng', 'anh em', 'gia đình', 'họp mặt'],
        'greeting': 'Chào bác! Cháu thích nghe bác kể về gia đình. Gia đình bác có gì vui không?'
    },
    'suc_khoe': {
        'name': '💊 Sức khỏe',
        'description': 'Thuốc nam, chế độ ăn uống, tập thể dục cho người cao tuổi',
        'folder': 'suc_khoe',
        'emoji': '💊',
        'keywords': ['khỏe', 'thuốc', 'bệnh', 'tập', 'ăn', 'uống', 'ngủ'],
        'greeting': 'Chào bác! Sức khỏe là vàng đúng không? Bác có muốn tâm sự về sức khỏe không?'
    },
    'lich_su': {
        'name': '📚 Lịch sử',
        'description': 'Các triều đại, kháng chiến, nhân vật lịch sử, sự kiện đã trải qua',
        'folder': 'lich_su',
        'emoji': '📚',
        'keywords': ['lịch sử', 'chiến tranh', 'vua', 'anh hùng', 'cách mạng'],
        'greeting': 'Chào bác! Cháu muốn nghe bác kể về những câu chuyện lịch sử thú vị.'
    },
    'tam_linh': {
        'name': '🙏 Tâm linh',
        'description': 'Phật giáo, thờ cúng tổ tiên, lễ hội truyền thống, phong thủy',
        'folder': 'tam_linh',
        'emoji': '🙏',
        'keywords': ['phật', 'cúng', 'tổ tiên', 'chùa', 'lễ', 'phong thủy'],
        'greeting': 'Chào bác! Cháu sẵn sàng tâm sự với bác về tâm linh và tín ngưỡng.'
    }
}

# ==================== HALLUCINATION FILTER ====================

class HallucinationFilter:
    """Lọc hallucination cho Whisper STT"""
    
    def __init__(self):
        # Vietnamese hallucination patterns
        # Câu spam cần match chính xác (exact match)
        self.exact_hallucinations = [
        "hẹn gặp lại các bạn trong những video tiếp theo.",
        "cảm ơn các bạn đã theo dõi và hẹn gặp lại.",
        "hãy đăng ký kênh để không bỏ lỡ video mới.",
        "hãy like và subscribe để ủng hộ kênh.",
        "hãy đăng ký kênh để ủng hộ",
        "hãy đăng ký kênh để xem thêm nhiều video hay",
]

# Các cụm spam ngắn, chỉ cần xuất hiện trong câu (substring match)
        self.partial_hallucinations = [
            "đăng ký", "đăng ký kênh", "subscribe", "like và subscribe",
            "lala", "lalala", "ghiền mì gõ", "ơ ơ ơ", "à à à", "ờ ờ ờ",
            "ừ ừ ừ", "ử ử ử", "e e e", "i i i", "u u u", "o o o",
]
        
        # English hallucination patterns
        self.english_patterns = [
            "thank you for watching", "thanks for watching", "like and subscribe",
            "subscribe to", "www.", "http", ".com", "please subscribe",
            "hit the bell", "notification bell"
        ]
        
        # Music/sound patterns
        self.music_patterns = [
            "♪", "♫", "music", "âm nhạc", "nhạc nền",
            "[âm nhạc]", "[music]", "(music)", "(âm nhạc)"
        ]
        
        # Quality thresholds
        self.min_length = 3
        self.max_length = 200
        self.min_confidence = 0.00


    def is_hallucination(self, text, confidence=None):
        """Kiểm tra xem text có phải là hallucination hay không"""
        text_clean = text.strip().lower()

        # Check length
        if len(text_clean) < self.min_length or len(text_clean) > self.max_length:
            return True

        # Check confidence if provided
        if confidence is not None and confidence < self.min_confidence:
            return True

        # Check exact match hallucinations
        if text_clean in self.exact_hallucinations:
            return True

        # Check partial (substring) hallucinations
        for pattern in self.partial_hallucinations:
            if pattern in text_clean:
                return True

        # Check English hallucinations
        for pattern in self.english_patterns:
             if pattern in text_clean:
                return True

        # Check music patterns
        for pattern in self.music_patterns:
             if pattern in text_clean:
                return True

        # Check repetitive patterns
        if self._is_repetitive(text_clean):
            return True

         # Check if mostly non-alphabetic
        if self._is_mostly_symbols(text_clean):
            return True

        return False

    def _is_repetitive(self, text):
        """Kiểm tra pattern lặp lại"""
        words = text.split()
        if len(words) < 3:
            return False
            
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
            if word_count[word] >= 3:
                return True
        return False

    def _is_mostly_symbols(self, text):
        """Kiểm tra xem có phải chủ yếu là ký tự đặc biệt không"""
        if len(text) == 0:
            return True
        alpha_count = sum(1 for c in text if c.isalpha())
        return alpha_count / len(text) < 0.5

    def filter_text(self, text, confidence=None):
        """Lọc text, trả về text clean hoặc empty string"""
        if self.is_hallucination(text, confidence):
            return ""
        
        # Clean up text
        text_clean = text.strip()
        text_clean = re.sub(r'\s+', ' ', text_clean)
        return text_clean

# ==================== ADVANCED EMOTION DETECTION ====================

def detect_emotion_and_optimize_response(user_message):
    """Phân tích cảm xúc và đưa ra gợi ý phản hồi"""
    emotion_keywords = {
        'buồn': ['buồn', 'khóc', 'cô đơn', 'một mình', 'chán nản', 'tủi thân', 'u uất'],
        'nhớ_quê': ['nhớ', 'quê', 'xa nhà', 'nước ngoài', 'hoài niệm', 'hương', 'làng'],
        'lo_lắng': ['lo', 'sợ', 'băn khoăn', 'không biết', 'thế nào', 'làm sao', 'tìm đâu'],
        'vui': ['vui', 'hạnh phúc', 'tốt', 'khỏe', 'hài lòng', 'sung sướng', 'phấn khích'],
        'bệnh_tật': ['đau', 'ốm', 'bệnh', 'mệt', 'yếu', 'thuốc', 'khó chịu'],
        'gia_đình': ['con', 'cháu', 'vợ', 'chồng', 'anh em', 'họ hàng', 'thăm']
    }
    
    detected_emotions = []
    message_lower = user_message.lower()
    
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            detected_emotions.append(emotion)
    
    # Tạo response optimization hint
    optimization_hint = ""
    if 'buồn' in detected_emotions:
        optimization_hint += """
PHÁT HIỆN CẢM XÚC BUỒN - ÁP DỤNG CHIẾN LƯỢC AN ỦI:
• Bắt đầu bằng việc thừa nhận cảm xúc: "Cháu hiểu bác đang buồn..."
• Đồng hành: "Bác không một mình đâu, có cháu ở đây"
• Chuyển hướng nhẹ nhàng về điều tích cực
• TRÁNH: Khuyên giải ngay, bỏ qua cảm xúc
"""
    
    if 'nhớ_quê' in detected_emotions:
        optimization_hint += """
PHÁT HIỆN TÌNH CẢM NHỚ QUÊ - ÁP DỤNG CHIẾN LƯỢC HOÀI NIỆM:
• Chia sẻ cảm xúc: "Xa quê lòng nao nao, cháu hiểu lắm..."
• Gợi mở ký ức: "Món gì ở quê bác thích nhất?"
• Kết nối văn hóa: "Mình cùng tìm cách làm món đó ở đây"
• TRÁNH: Nói về tương lai, bỏ qua nỗi nhớ
"""
    
    return detected_emotions, optimization_hint

# ==================== DIALECT SUPPORT ====================

def get_dialect_style(hometown):
    """Xác định giọng địa phương với Chain of Thought và Few-shot Prompting"""
    # Mapping các tỉnh về đại diện
    province_mapping = {
        # Miền Bắc
        "Hà Nội": "Hà Nội", "Hà Tây": "Hà Nội", "Bắc Ninh": "Hà Nội",
        "Nam Định": "Nam Định", "Thái Bình": "Nam Định", "Hà Nam": "Nam Định",
        
        # Miền Trung
        "Huế": "Huế", "Thừa Thiên Huế": "Huế", "Quảng Trị": "Huế",
        "Nghệ An": "Nghệ An", "Hà Tĩnh": "Nghệ An", "Thanh Hóa": "Nghệ An",
        
        # Miền Nam
        "TP.HCM": "TP.HCM", "Hồ Chí Minh": "TP.HCM", "Sài Gòn": "TP.HCM",
        "Cần Thơ": "Cần Thơ", "An Giang": "Cần Thơ", "Kiên Giang": "Cần Thơ"
    }
    
    representative = province_mapping.get(hometown, "Hà Nội")
    
    dialect_examples = {
        "Hà Nội": "Giọng lịch sự, trang trọng, dùng 'ạ', 'thưa', 'dạ'",
        "Nam Định": "Chân chất, mộc mạc, dùng 'nhỉ', 'đó', 'này'",
        "Huế": "Nhẹ nhàng, ngọt ngào, dùng 'mình', 'rứa', 'nì'", 
        "Nghệ An": "Giọng 'gi' thành 'di', 'r' thành 'z', chân chất",
        "TP.HCM": "Thoải mái, phóng khoáng, dùng 'nhé', 'nha', 'dzậy'",
        "Cần Thơ": "Đậm chất miền Tây, dùng 'mầy', 'tui', 'dzậy'"
    }
    
    return f"""
GIỌNG {representative.upper()}:
Đặc điểm: {dialect_examples.get(representative, 'Giọng chung của người Việt')}

HƯỚNG DẪN ÁP DỤNG:
- Sử dụng từ ngữ đặc trưng một cách TỰ NHIÊN
- Giữ giọng điệu gần gũi, thân mật
- Lồng ghép văn hóa, món ăn địa phương
- Đảm bảo giọng nói tự nhiên, phù hợp người cao tuổi
"""

# ==================== FILE MANAGEMENT ====================

def ensure_topic_folders():
    """Tạo các thư mục chủ đề nếu chưa có"""
    if not os.path.exists(TOPICS_DIR):
        os.makedirs(TOPICS_DIR)
        print(f"Đã tạo thư mục chính: {TOPICS_DIR}")
    
    for topic_key, topic_info in TOPICS.items():
        topic_path = os.path.join(TOPICS_DIR, topic_info['folder'])
        if not os.path.exists(topic_path):
            os.makedirs(topic_path)
            print(f"Đã tạo thư mục: {topic_path}")

def get_topic_file_path(topic_key, file_type):
    """Lấy đường dẫn file theo chủ đề"""
    if topic_key not in TOPICS:
        raise ValueError(f"Chủ đề không hợp lệ: {topic_key}")
    
    topic_folder = TOPICS[topic_key]['folder']
    file_names = {
        'history': 'chat_history.json',
        'context': 'chat_context.json',
        'summary': 'chat_summary.json',
        'backup': 'full_conversation_backup.json'
    }
    
    if file_type not in file_names:
        raise ValueError(f"Loại file không hợp lệ: {file_type}")
    
    return os.path.join(TOPICS_DIR, topic_folder, file_names[file_type])

def load_user_info():
    """
    Đọc thông tin người dùng từ file JSON và luôn reload từ file.
    Nếu file không tồn tại, tạo file mẫu với thông tin mặc định.
    """
    try:
        # # In ra working directory để debug (tuỳ chọn)
        # print("Current working dir:", os.getcwd())
        
        if os.path.exists(USER_INFO_FILE):
            with open(USER_INFO_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # print("✅ Loaded user_info:", data)
                return data
        
        # Nếu file chưa tồn tại, tạo file mẫu
        default_info = {
            "name": "Bác Tâm",
            "age": 70,
            "gender": "Nam",
            "hometown": "Nam Định",
            "location": "Hà Nội",
            "occupation": "Giáo viên về hưu",
            "family": "Có vợ, 2 con, 3 cháu",
            "health": "Khỏe mạnh",
            "call_style": "bác"
        }
        with open(USER_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_info, f, ensure_ascii=False, indent=2)
        print(f"⚠️ File {USER_INFO_FILE} không tồn tại. Đã tạo file mẫu với thông tin mặc định.")
        return default_info
    
    except Exception as e:
        # Bắt mọi lỗi đọc/ghi và trả về dict rỗng
        print(f"❌ Error loading {USER_INFO_FILE}: {e}")
        return {}

# ==================== ADVANCED PROMPTING SYSTEM ====================

def get_enhanced_system_prompt(topic_key, user_input=None, user_info=None):
    """Tạo prompt nâng cao với đầy đủ tính năng từ GitHub repo"""
    # LUÔN reload user_info mới nhất từ file
    user_info = load_user_info()
    
    prompt_parts = []
    
    # Phần 1: Vai trò và nguyên tắc cơ bản
    prompt_parts.append("""
Bạn là một người bạn thân thiết, luôn lắng nghe, chia sẻ và tâm sự với người lớn tuổi Việt Nam.
Hãy trò chuyện như một người bạn đồng hành, không phải chuyên gia hay trợ lý AI.

NGUYÊN TẮC VÀNG:
- TRẢ LỜI NGẮN GỌN: TỐI ĐA 4-5 CÂU, tránh dài dòng
- SÁNG TẠO TRONG CÁCH TRẢ LỜI: Không dùng từ ngữ máy móc, tránh lặp từ
- Luôn lắng nghe, đồng cảm, chia sẻ cảm xúc, động viên nhẹ nhàng
- Sử dụng giọng điệu tự nhiên, gần gũi, thân mật
- Không dùng markdown, không in đậm, không ký tự đặc biệt
- NÓI NHƯ NGƯỜI THẬT: Câu chuyện tự nhiên như nói chuyện hàng ngày
""")
    
    # Phần 2: Thông tin cá nhân
    if user_info:
        call_style = user_info.get('call_style', 'bác')
        prompt_parts.append(f"\nQUAN TRỌNG: Luôn gọi người dùng là '{call_style}' trong mọi câu trả lời.\n")
        
        # Thêm thông tin cá nhân
        personal_info = []
        if user_info.get('name'): personal_info.append(f"Tên: {user_info['name']}")
        if user_info.get('age'): personal_info.append(f"Tuổi: {user_info['age']}")
        if user_info.get('hometown'): personal_info.append(f"Quê quán: {user_info['hometown']}")
        if user_info.get('location'): personal_info.append(f"Nơi ở: {user_info['location']}")
        if user_info.get('occupation'): personal_info.append(f"Nghề nghiệp: {user_info['occupation']}")
        
        if personal_info:
            prompt_parts.append("THÔNG TIN CÁ NHÂN CẦN SỬ DỤNG: " + ", ".join(personal_info) + "\n")
        
        # Giọng địa phương
        if user_info.get('hometown'):
            dialect_style = get_dialect_style(user_info['hometown'])
            prompt_parts.append(f"\nGIỌNG ĐỊA PHƯƠNG: {dialect_style}\n")
    
    # Phần 3: Chủ đề cụ thể
    topic_prompts = {
        'que_huong': """
BẠN LÀ CHUYÊN GIA VỀ QUÊ HƯƠNG VÀ HOÀI NIỆM:
- Chia sẻ về món ăn quê hương, cách nấu truyền thống
- Kể về phong cảnh, con người, làng xóm quê nhà
- Nhớ về ca dao, tục ngữ, truyện cổ tích
- Mô tả lễ hội, tết cổ truyền, phong tục tập quán
""",
        'gia_dinh': """
BẠN LÀ CHUYÊN GIA VỀ GIA ĐÌNH:
- Đưa ra cách giữ liên lạc với người thân
- Hướng dẫn truyền dạy tiếng Việt, văn hóa cho con cháu
- Kể chuyện về gia đình, tổ tiên với giọng điệu ấm áp
- Hỗ trợ giáo dục con cháu về văn hóa Việt
""",
        'suc_khoe': """
BẠN LÀ CHUYÊN GIA VỀ SỨC KHỎE:
- Giới thiệu thuốc nam, bài thuốc dân gian
- Đề xuất chế độ ăn uống bổ dưỡng cho người cao tuổi
- Gợi ý bài tập thể dục phù hợp (thái cực quyền, yoga)
- Chia sẻ cách phòng ngừa bệnh tật
""",
        'lich_su': """
BẠN LÀ CHUYÊN GIA VỀ LỊCH SỬ VIỆT NAM:
- Kể về các triều đại, vua chúa nổi tiếng
- Mô tả các cuộc kháng chiến chống Pháp, chống Mỹ
- Chia sẻ về nhân vật lịch sử với góc nhìn gần gũi
- Truyền đạt bài học lịch sử cho thế hệ trẻ
""",
        'tam_linh': """
BẠN LÀ CHUYÊN GIA VỀ VĂN HÓA TÂM LINH:
- Giải thích về Phật giáo, tín ngưỡng Việt Nam
- Hướng dẫn cách thờ cúng tổ tiên ở nước ngoài
- Mô tả lễ hội, tết cổ truyền với ý nghĩa tâm linh
- Chia sẻ triết lý sống, tu dưỡng đạo đức
"""
    }
    
    if topic_key in topic_prompts:
        prompt_parts.append(topic_prompts[topic_key])
    
    return '\n'.join(prompt_parts)

# ==================== CONVERSATION MANAGEMENT ====================

def load_chat_history(topic_key):
    """Load lịch sử chat theo chủ đề"""
    try:
        file_path = get_topic_file_path(topic_key, 'history')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('messages', [])
        return []
    except Exception as e:
        print(f"Lỗi đọc file lịch sử {topic_key}: {e}")
        return []

def save_chat_history(topic_key, messages):
    """Lưu lịch sử chat theo chủ đề"""
    try:
        with file_lock:
            file_path = get_topic_file_path(topic_key, 'history')
            chat_data = {
                'topic': topic_key,
                'topic_name': TOPICS[topic_key]['name'],
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'total_messages': len(messages),
                'messages': messages
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Lỗi ghi file lịch sử {topic_key}: {e}")

def add_message_to_history(topic_key, user_message, bot_response):
    """Thêm tin nhắn vào lịch sử"""
    new_message = {
        'timestamp': datetime.now().isoformat(),
        'user': user_message,
        'bot': bot_response,
        'emotions_detected': detect_emotion_and_optimize_response(user_message)[0]
    }
    
    messages = load_chat_history(topic_key)
    messages.append(new_message)
    save_chat_history(topic_key, messages)

# ==================== ENHANCED LLM CLASS ====================

class EnhancedVietnameseLLM:
    """Enhanced Vietnamese LLM with full features from GitHub repo"""
    
    def __init__(self, api_key=None, default_topic='que_huong'):
        self.api_key = api_key or API_KEY
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
        except Exception as e:
            print(f"❌ Lỗi cấu hình Gemini API: {e}")
            self.model = None
            return
        
        self.current_topic = default_topic
        self.chat_session = None
        
        # Initialize session
        self._initialize_session(default_topic)
    
    def _initialize_session(self, topic):
        """Khởi tạo chat session với prompt nâng cao - LUÔN reload user_info"""
        if not self.model:
            print("❌ Model chưa được khởi tạo")
            return
        
        try:
            self.current_topic = topic
            
            # LUÔN tải lại user_info mới nhất từ file
            current_user_info = load_user_info()
            
            # Tạo system prompt nâng cao với user_info mới nhất
            system_prompt = get_enhanced_system_prompt(topic, user_info=current_user_info)
            
            # Tạo lời chào thân thiện
            topic_info = TOPICS[topic]
            greeting = topic_info['greeting']
            
            # Khởi tạo chat session
            self.chat_session = self.model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [system_prompt]
                    },
                    {
                        "role": "model", 
                        "parts": [greeting]
                    }
                ]
            )
            
            # print(f"🤖 Khởi tạo session cho chủ đề: {topic}")
            # print(f"🤖 Bot: {greeting}")
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo LLM session: {e}")
            self.chat_session = None
    
    def update_user_info(self):
        self._initialize_session(self.current_topic)
    
    def chat(self, text, topic_key=None):
        """Chat với LLM với emotion detection và user_info update"""
        if not self.chat_session:
            return "Xin lỗi bác, cháu gặp chút vấn đề kỹ thuật. Vui lòng kiểm tra API key."
        
        # Chuyển topic nếu cần
        if topic_key and topic_key != self.current_topic:
            self._initialize_session(topic_key)
        
        # Luôn reload user_info trước khi chat để đảm bảo cập nhật
        self.update_user_info()
        
        try:
            # Phân tích cảm xúc
            detected_emotions, optimization_hint = detect_emotion_and_optimize_response(text)
            
            # Thêm optimization hint nếu có
            enhanced_message = text
            if optimization_hint:
                enhanced_message = f"{optimization_hint}\n\nTin nhắn từ người dùng: {text}"
            
            # Tạo response
            response = self.chat_session.send_message(enhanced_message, stream=False)
            
            if response and response.text:
                response_text = response.text.strip()
                
                # Làm sạch markdown formatting
                response_text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', response_text)
                response_text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', response_text)
                
                # Lưu vào lịch sử
                if topic_key:
                    add_message_to_history(topic_key, text, response_text)
                
                return response_text
            else:
                return "Bác ơi, cháu không nghe rõ. Bác nói lại được không?"
                
        except Exception as e:
            print(f"❌ Lỗi LLM: {e}")
            return "Xin lỗi bác, cháu đang bận một tí. Bác đợi cháu một chút nhé."

# ==================== SPEECH PROCESSING WITH TIMING ====================

def process_audio_with_llm(frames, stt_model, llm, hallucination_filter, device, sample_rate, topic_key='que_huong'):
    """Xử lý audio với LLM integration và đo thời gian"""
    try:
        # Bắt đầu đo tổng thời gian
        start_total_time = time.time()
        
        # Concatenate and denoise audio
        audio_np = np.concatenate(frames)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            audio_denoised = nr.reduce_noise(y=audio_np.astype(np.float32), sr=sample_rate)
        
        # STT với đo thời gian
        print("🔄 Đang nhận dạng tiếng nói...")
        start_stt_time = time.time()
        result = stt_model.transcribe(audio_denoised, language='vi', fp16=(device == "cuda"))
        end_stt_time = time.time()
        stt_processing_time = end_stt_time - start_stt_time
        
        raw_text = result["text"].strip()
        confidence = result.get("avg_logprob", 0)
        
        # print(f"🗣️ Raw STT: '{raw_text}' (confidence: {confidence:.3f})")
        # print(f"⏱️ Thời gian STT: {stt_processing_time:.3f} giây")
        
        # Apply hallucination filter
        filtered_text = hallucination_filter.filter_text(raw_text, confidence)
        
        if not filtered_text:
            print("🔍 Filtered as hallucination")
            return None
        
        # print(f"✅ Filtered text: '{filtered_text}'")
        print(f"👤 Bạn: {filtered_text}")
        
        # LLM Response với đo thời gian
        # print("🔄 Đang suy nghĩ...")
        start_llm_time = time.time()
        bot_response = llm.chat(filtered_text, topic_key)
        end_llm_time = time.time()
        llm_processing_time = end_llm_time - start_llm_time
        
        # Tính tổng thời gian
        end_total_time = time.time()
        total_processing_time = end_total_time - start_total_time
        
        print(f"🤖 Bot: {bot_response}")
        # print(f"⏱️ Thời gian LLM: {llm_processing_time:.3f} giây")
        # print(f"⏱️ Tổng thời gian xử lý: {total_processing_time:.3f} giây")
        print(f"📊 Tổng thời gian xử lý: STT({stt_processing_time:.3f}s) + LLM({llm_processing_time:.3f}s) = Total({total_processing_time:.3f}s)")
        
        return {"user": filtered_text, "bot": bot_response}
        
    except Exception as e:
        print(f"❌ Lỗi xử lý audio: {e}")
        return None

# ==================== MAIN TERMINAL MODE ====================

def run_terminal_mode():
    """Chạy chế độ terminal với đầy đủ tính năng"""
    # Đảm bảo các thư mục tồn tại
    ensure_topic_folders()
    
    print("🔄 Đang tải models...")
    
    # Load models
    vad_model = load_silero_vad()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stt_model = whisper.load_model("large-v3", device=device)
    
    # Initialize components
    print("🔄 Đang khởi tạo Vietnamese LLM...")
    llm = EnhancedVietnameseLLM()
    if not llm.model:
        print("❌ Không thể khởi tạo LLM. Vui lòng kiểm tra API key.")
        return
    
    print("🔄 Đang khởi tạo Hallucination Filter...")
    hallucination_filter = HallucinationFilter()
    
    print("✅ Tất cả components đã sẵn sàng!")
    
    # Audio settings
    sample_rate = 16000
    block_duration = 0.3
    block_size = int(sample_rate * block_duration)
    silence_threshold = 0.7
    q = queue.Queue()
    
    # Statistics
    stats = {
        'total_transcriptions': 0,
        'filtered_hallucinations': 0,
        'successful_responses': 0
    }
    
    def audio_callback(indata, frames, time_info, status):
        """Callback thu âm"""
        q.put(indata.copy())
    
    def is_speech_block(audio_block, vad_model, sample_rate, chunk_size=512):
        """Kiểm tra có tiếng nói không"""
        for start in range(0, len(audio_block), chunk_size):
            chunk = audio_block[start:start+chunk_size]
            if len(chunk) < chunk_size:
                break
            tensor_chunk = torch.from_numpy(chunk).float()
            prob = vad_model(tensor_chunk, sample_rate)
            if prob.item() > 0.5:
                return True
        return False
    
    # Chọn chủ đề
    print("\n" + "="*70)
    print("        🎤 VIETNAMESE COMPANION CHATBOT - TERMINAL MODE")
    print("="*70)
    print("Chọn chủ đề tâm sự:")
    for i, (key, info) in enumerate(TOPICS.items(), 1):
        print(f"{i}. {info['name']}: {info['description']}")
    
    while True:
        try:
            choice = input("\nNhập số thứ tự chủ đề (1-5): ").strip()
            topic_keys = list(TOPICS.keys())
            if choice.isdigit() and 1 <= int(choice) <= len(topic_keys):
                selected_topic = topic_keys[int(choice) - 1]
                llm._initialize_session(selected_topic)
                break
            else:
                print("Vui lòng nhập số từ 1-5")
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            return
    
    print("\n" + "="*70)
    print("📋 Hướng dẫn:")
    print("   • Nói bình thường, hệ thống tự nhận diện")
    print("   • Nhấn Enter để dừng đoạn ghi âm hiện tại")
    print("   • Nhấn 'u' + Enter để cập nhật thông tin cá nhân")
    print("   • Ctrl+C để thoát chương trình")
    print("="*70)
    
    frames = []
    recording = False
    last_speech_time = None
    
    with sd.InputStream(samplerate=sample_rate,
                       channels=1,
                       dtype='float32',
                       blocksize=block_size,
                       callback=audio_callback):
        
        while True:
            try:
                # Check for Enter key press
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    user_input = sys.stdin.readline().strip()
                    
                    # Nếu user nhập 'u' thì cập nhật user info
                    if user_input.lower() == 'u':
                        print("🔄 Đang cập nhật thông tin cá nhân từ user_info.json...")
                        llm.update_user_info()
                        continue
                    
                    if recording:
                        print("\n⏹️ Dừng ghi âm hiện tại.")
                        result = process_audio_with_llm(frames, stt_model, llm,
                                                      hallucination_filter, device,
                                                      sample_rate, selected_topic)
                        if result:
                            stats['successful_responses'] += 1
                        
                        print("-" * 60)
                        frames = []
                        recording = False
                        last_speech_time = None
                    else:
                        print("\n⚠️ Chưa có đoạn nào đang ghi âm.")
                        print("🎤 Tiếp tục nói, nhấn 'u' để update user info, hoặc nhấn Enter...")
                
                audio_block = q.get()
                
                # Noise reduction
                audio_float32 = audio_block.flatten()
                # SỬA LỖI: Thêm dòng reduce_noise trước khi sử dụng audio_denoised_block
                audio_denoised_block = nr.reduce_noise(y=audio_float32, sr=sample_rate, stationary=False)
                audio_denoised_block = np.nan_to_num(audio_denoised_block, nan=0.0, posinf=0.0, neginf=0.0)
                audio_int16 = (audio_denoised_block * 32768).astype(np.int16)
                
                # VAD check
                speech_detected = is_speech_block(audio_int16, vad_model, sample_rate)
                
                if speech_detected:
                    if not recording:
                        print("\n🎤 Phát hiện tiếng nói, bắt đầu ghi âm...")
                        recording = True
                    
                    frames.append(audio_denoised_block)
                    last_speech_time = time.time()
                else:
                    if recording:
                        silence_time = time.time() - last_speech_time
                        if silence_time > silence_threshold:
                            print(f"\n⏹️ Dừng ghi âm sau {silence_time:.2f}s im lặng.")
                            result = process_audio_with_llm(frames, stt_model, llm,
                                                          hallucination_filter, device,
                                                          sample_rate, selected_topic)
                            if result:
                                stats['successful_responses'] += 1
                            else:
                                stats['filtered_hallucinations'] += 1
                            
                            stats['total_transcriptions'] += 1
                            print(f"📊 Stats: {stats['successful_responses']}/{stats['total_transcriptions']} successful")
                            print("-" * 60)
                            
                            frames = []
                            recording = False
                            last_speech_time = None
                            
            except KeyboardInterrupt:
                print("\n\n👋 Cảm ơn bạn đã sử dụng Enhanced Vietnamese Companion Chatbot!")
                print(f"\n📊 Final Stats:")
                print(f"   Total transcriptions: {stats['total_transcriptions']}")
                print(f"   Successful responses: {stats['successful_responses']}")
                print(f"   Filtered hallucinations: {stats['filtered_hallucinations']}")
                
                if stats['total_transcriptions'] > 0:
                    success_rate = (stats['successful_responses'] / stats['total_transcriptions']) * 100
                    filter_rate = (stats['filtered_hallucinations'] / stats['total_transcriptions']) * 100
                    print(f"   Success rate: {success_rate:.1f}%")
                    print(f"   Filter rate: {filter_rate:.1f}%")
                break
                
            except Exception as e:
                print(f"❌ Lỗi: {e}")
                continue

# ==================== MAIN FUNCTION ====================

def main():
    print("=" * 70)
    print("                 🤖 VIETNAMESE COMPANION CHATBOT")
    print("                            Terminal Mode ")
    # print("   ✅ Đã cập nhật user_info từ file JSON")
    # print("   ✅ Hiển thị thời gian xử lý STT, LLM và tổng thời gian")
    print("=" * 70)
    
    try:
        run_terminal_mode()
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()