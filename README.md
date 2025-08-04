# Vietnamese Companion Chatbot

## 🎯 Mô tả

**Vietnamese Companion Chatbot** là một ứng dụng trò chuyện AI tiên tiến được thiết kế đặc biệt để trò chuyện và đồng hành cùng người lớn tuổi Việt Nam. Ứng dụng sử dụng công nghệ Speech-to-Text (STT) của OpenAI Whisper, Voice Activity Detection (VAD) của Silero, và Google Gemini AI để tạo ra những cuộc trò chuyện tự nhiên, ấm áp và có cảm xúc.

## ✨ Tính năng chính

### 🎤 Giao tiếp giọng nói thông minh
- **Voice Activity Detection (VAD)**: Tự động phát hiện và bắt đầu/kết thúc ghi âm
- **Speech-to-Text nâng cao**: Sử dụng Whisper large-v3 với độ chính xác cao cho tiếng Việt
- **Lọc Hallucination**: Loại bỏ các câu không liên quan từ YouTube, âm nhạc, v.v.
- **Khử nhiễu âm thanh**: Cải thiện chất lượng âm thanh đầu vào

### 🧠 AI thông minh và cảm xúc
- **Phân tích cảm xúc**: Tự động nhận diện trạng thái cảm xúc (buồn, vui, nhớ quê, lo lắng...)
- **Tối ưu phản hồi**: Điều chỉnh cách trả lời phù hợp với cảm xúc được phát hiện
- **Hỗ trợ giọng địa phương**: Nhận diện và sử dụng từ ngữ đặc trưng các vùng miền
- **Cập nhật thông tin tự động**: Học và ghi nhớ thông tin cá nhân từ cuộc trò chuyện

### 🎭 Chủ đề đa dạng
- **🏠 Quê hương và hoài niệm**: Chia sẻ về món ăn, phong cảnh, truyền thống
- **👨‍👩‍👧‍👦 Gia đình**: Kể chuyện gia đình, truyền dạy văn hóa con cháu
- **💊 Sức khỏe**: Thuốc nam, chế độ ăn uống, bài tập cho người cao tuổi
- **📚 Lịch sử**: Các triều đại, kháng chiến, nhân vật lịch sử
- **🙏 Tâm linh**: Phật giáo, thờ cúng tổ tiên, lễ hội truyền thống

### 📊 Theo dõi hiệu suất
- **Timing chi tiết**: Hiển thị thời gian xử lý STT, LLM và tổng thời gian
- **Thống kê cuộc trò chuyện**: Theo dõi số lượng transcription thành công
- **Lưu trữ lịch sử**: Ghi lại toàn bộ cuộc trò chuyện theo từng chủ đề

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- Microphone (để ghi âm)
- RAM: Tối thiểu 8GB (khuyến nghị 16GB)
- Disk: ~5GB trống (cho models)
- GPU: CUDA-compatible GPU (tùy chọn, để tăng tốc)

### Bước 1: Clone repository
```bash
git clone https://github.com/NamBuw/chatbot.git
cd chatbot
```

### Bước 2: Tạo virtual environment
```bash
python -m venv venv

# Windows
venv\\Scripts\\activate

# macOS/Linux
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình API Key
1. Lấy Google Gemini API key từ [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Mở file `enhanced_main_final.py`
3. Thay thế `API_KEY` bằng key của bạn:
```python
API_KEY = "your-gemini-api-key-here"
```

### Bước 5: Chạy ứng dụng
```bash
python main.py
```

## 📋 Hướng dẫn sử dụng

### Khởi động
1. Chạy `main.py`
2. Chọn chủ đề tâm sự (1-5)
3. Chờ hệ thống tải models (có thể mất vài phút lần đầu)

### Tương tác
- **Nói chuyện tự nhiên**: Hệ thống tự động phát hiện tiếng nói
- **Nhấn Enter**: Dừng ghi âm hiện tại
- **Nhấn 'u' + Enter**: Cập nhật thông tin cá nhân thủ công
- **Ctrl+C**: Thoát chương trình

### Cập nhật thông tin cá nhân
Hệ thống tự động cập nhật thông tin khi bạn chia sẻ:
- "Tôi là Nguyễn Văn A" → Cập nhật tên
- "Tôi 65 tuổi" → Cập nhật tuổi
- "Quê tôi ở Huế" → Cập nhật quê quán
- "Tôi làm bác sĩ" → Cập nhật nghề nghiệp

## 📁 Cấu trúc thư mục

```
chatbot/
├── main.py                    # File chính
├── requirements.txt          # Dependencies
├── README.md                # Tài liệu này
├── user_info.json           # Thông tin cá nhân (tự động tạo)
└── topics/                  # Thư mục lưu lịch sử chat
    ├── que_huong/
    ├── gia_dinh/
    ├── suc_khoe/
    ├── lich_su/
    └── tam_linh/
```

## ⚙️ Tùy chỉnh

### Thay đổi thông tin cá nhân
Chỉnh sửa file `user_info.json`:
```json
{
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
```

### Thay đổi cài đặt âm thanh
Trong file `main.py`, tìm và chỉnh sửa:
```python
# Audio settings
sample_rate = 16000          # Tần số lấy mẫu
block_duration = 0.3         # Thời gian mỗi block (giây)
silence_threshold = 0.7      # Thời gian im lặng để dừng ghi âm (giây)
```

## 🔧 Troubleshooting

### Lỗi thường gặp

**1. Lỗi không tìm thấy microphone**
```bash
# Kiểm tra thiết bị âm thanh
python -c "import sounddevice; print(sounddevice.query_devices())"
```

**2. Lỗi CUDA out of memory**
- Giảm batch size hoặc sử dụng CPU
- Trong file chính, thay đổi:
```python
device = "cpu"  # Thay vì "cuda"
```

**3. Lỗi Gemini API**
- Kiểm tra API key đã đúng chưa
- Kiểm tra kết nối internet
- Kiểm tra quota API key

**4. Lỗi models không tải được**
```bash
# Tải lại models
pip uninstall whisper
pip install whisper==1.1.10
```

### Tối ưu hiệu suất

**1. Sử dụng GPU (khuyến nghị)**
```bash
# Cài đặt PyTorch với CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. Giảm model size nếu RAM thấp**
Trong file chính, thay đổi:
```python
stt_model = whisper.load_model("base", device=device)  # Thay vì "large-v3"
```

## 📊 Thông số kỹ thuật

### Models sử dụng
- **STT**: OpenAI Whisper large-v3 (~5GB)
- **VAD**: Silero VAD v5.1 (~100MB)
- **LLM**: Google Gemini 2.0 Flash (API)

### Hiệu suất
- **Latency**: 2-5 giây (tùy thuộc hardware)
- **Accuracy**: >95% cho tiếng Việt
- **RAM usage**: 6-8GB
- **VRAM usage**: 4-6GB (nếu dùng GPU)

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Để đóng góp:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add some amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

## 📝 License

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - STT model
- [Silero Models](https://github.com/snakers4/silero-models) - VAD model
- [Google Gemini](https://ai.google.dev/) - LLM API
- [SoundDevice](https://python-sounddevice.readthedocs.io/) - Audio I/O
- [NoiseReduce](https://github.com/timsainb/noisereduce) - Noise reduction

## 📞 Liên hệ

- **Issues**: [GitHub Issues](https://github.com/NamBuw/chatbot/issues)
- **Email**: trungnam0708qwert@gmail.com
---

**Made with ❤️ for Vietnamese elderly community**
