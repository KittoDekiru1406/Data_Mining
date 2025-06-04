# 🔬 Ứng dụng Phân cụm FCM (Fuzzy C-Means)

Một ứng dụng phân cụm hoàn chỉnh sử dụng thuật toán Fuzzy C-Means cho cả dữ liệu CSV và ảnh viễn thám, với giao diện web thân thiện được xây dựng bằng Streamlit.

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Cài đặt](#️-cài-đặt)
- [Cấu trúc Project](#-cấu-trúc-project)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [API Reference](#-api-reference)
- [Ví dụ](#-ví-dụ)
- [Kết quả](#-kết-quả)
- [Đóng góp](#-đóng-góp)

## ✨ Tính năng

### 🎯 Phân cụm Dữ liệu CSV
- ✅ Hỗ trợ các dataset chuẩn: Iris, Wine, Dry Bean
- ✅ Tính toán 8 chỉ số đánh giá chất lượng phân cụm
- ✅ Trực quan hóa kết quả bằng biểu đồ 2D và 3D
- ✅ Xuất báo cáo LaTeX

### 🛰️ Phân cụm Ảnh Viễn thám
- ✅ Xử lý ảnh đa kênh phổ (multi-band)
- ✅ Phân đoạn ảnh thành các vùng đồng nhất
- ✅ Hiển thị ảnh trước và sau phân cụm
- ✅ Thống kê phân bố các cụm

### 🌐 Giao diện Web
- ✅ Giao diện thân thiện với Streamlit
- ✅ Tùy chỉnh tham số FCM realtime
- ✅ Hiển thị metrics và biểu đồ tương tác
- ✅ Responsive design

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- RAM: Tối thiểu 4GB (8GB khuyến nghị cho ảnh lớn)
- Dung lượng: ~500MB

### Bước 1: Clone repository
```bash
git clone <repository-url>
cd f1_teams
```

### Bước 2: Tạo môi trường ảo
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
# hoặc
env\Scripts\activate     # Windows
```

### Bước 3: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 4: Kiểm tra cài đặt
```bash
python -c "import streamlit; print('Streamlit installed successfully!')"
```

## 📁 Cấu trúc Project

```
f1_teams/
├── 📱 clustering_app.py          # Ứng dụng web Streamlit chính
├── 🧮 fcm.py                     # Lớp Fuzzy C-Means
├── 📊 fcm_all_data.py           # Script phân cụm dữ liệu CSV
├── 🛰️ fcm_image.py              # Script phân cụm ảnh viễn thám
├── 🖼️ img2data.py               # Xử lý ảnh đa kênh
├── 🎨 imgsegment.py             # Tạo ảnh phân đoạn
├── 🔧 utility.py                # Các hàm tiện ích
├── 📈 validity.py               # Các chỉ số đánh giá
├── 📋 requirements.txt          # Dependencies
├── 📖 README.md                 # Hướng dẫn này
├── 📂 data/                     # Dữ liệu
│   ├── csv/                     # Dữ liệu CSV
│   │   ├── Iris.csv
│   │   ├── Wine.csv
│   │   └── DryBean.csv
│   └── images/                  # Ảnh viễn thám
│       └── Anh-ve-tinh/
│           ├── Anh-da-pho/      # Ảnh đa kênh phổ
│           └── Anh-mau/         # Ảnh tổng hợp màu
└── 📤 outputs/                  # Kết quả xuất
    ├── images/                  # Ảnh đã phân cụm
    └── logs/                    # Báo cáo LaTeX
```

## 🚀 Hướng dẫn sử dụng

### 1. Chạy ứng dụng web (Khuyến nghị)

```bash
streamlit run clustering_app.py
```

Mở trình duyệt tại: `http://localhost:8501`

#### Giao diện chính:
1. **Sidebar**: Chọn loại dữ liệu và tham số FCM
2. **Main panel**: Hiển thị kết quả và biểu đồ
3. **Metrics**: Các chỉ số đánh giá chất lượng

### 2. Phân cụm dữ liệu CSV (Command line)

```bash
python fcm_all_data.py
```

**Kết quả:**
- File báo cáo: `outputs/logs/fcm_data.txt`
- Chứa metrics cho 3 datasets: Iris, Wine, DryBean

### 3. Phân cụm ảnh viễn thám (Command line)

```bash
python fcm_image.py
```

**Kết quả:**
- Ảnh phân đoạn: `outputs/images/fcm.jpg`
- File báo cáo: `outputs/logs/remote_sensing_clustering_report.txt`

## 📚 API Reference

### Lớp `Dfcm` (Fuzzy C-Means)

```python
from fcm import Dfcm

# Khởi tạo
fcm = Dfcm(n_clusters=3, m=2.0, epsilon=1e-5, max_iter=1000)

# Phân cụm
U, V, steps = fcm.fit(data, seed=42)
```

**Tham số:**
- `n_clusters`: Số cụm (2-10)
- `m`: Hệ số mờ (1.1-3.0, mặc định 2.0)
- `epsilon`: Ngưỡng hội tụ (1e-6 đến 1e-3)
- `max_iter`: Số vòng lặp tối đa

**Trả về:**
- `U`: Ma trận độ thuộc (membership matrix)
- `V`: Ma trận tâm cụm (centroids)
- `steps`: Số bước lặp để hội tụ

### Lớp `Dimg2data` (Xử lý ảnh)

```python
from img2data import Dimg2data

img_processor = Dimg2data()

# Đọc ảnh đa kênh từ thư mục
X, height, width = img_processor.read_multi_band_directory(
    directory="data/images/Anh-ve-tinh/Anh-da-pho/HaNoi",
    normalize=True
)
```

### Các chỉ số đánh giá

```python
from validity import *

# Davies-Bouldin Index (càng thấp càng tốt)
db = davies_bouldin(X, labels)

# Partition Coefficient (0-1, càng cao càng tốt)
pc = partition_coefficient(U)

# Silhouette Score (-1 đến 1, càng cao càng tốt)
si = silhouette(X, labels)
```

## 💡 Ví dụ

### Ví dụ 1: Phân cụm Iris dataset

```python
import pandas as pd
from fcm import Dfcm
from utility import extract_labels
from validity import davies_bouldin, silhouette

# Đọc dữ liệu
df = pd.read_csv('data/csv/Iris.csv')
X = df.iloc[:, :-1].values

# Phân cụm
fcm = Dfcm(n_clusters=3, m=2.0)
U, V, steps = fcm.fit(X, seed=42)
labels = extract_labels(U)

# Đánh giá
db_score = davies_bouldin(X, labels)
sil_score = silhouette(X, labels)

print(f"Davies-Bouldin: {db_score:.3f}")
print(f"Silhouette: {sil_score:.3f}")
```

### Ví dụ 2: Phân cụm ảnh viễn thám

```python
from img2data import Dimg2data
from fcm import Dfcm
from imgsegment import DimgSegment
from utility import extract_labels

# Đọc ảnh đa kênh
img_processor = Dimg2data()
X, height, width = img_processor.read_multi_band_directory(
    "data/images/Anh-ve-tinh/Anh-da-pho/HaNoi"
)

# Phân cụm
fcm = Dfcm(n_clusters=6, m=2.0)
U, V, steps = fcm.fit(X, seed=42)
labels = extract_labels(U)

# Tạo ảnh phân đoạn
img_segment = DimgSegment()
img_segment.save_label2image(labels, height, width, "output.jpg")
```

## 📊 Kết quả

### Chỉ số đánh giá chất lượng phân cụm

| Chỉ số | Ý nghĩa | Giá trị tốt |
|--------|---------|-------------|
| **Davies-Bouldin Index** | Độ phân tách giữa các cụm | Càng thấp càng tốt (< 1.0) |
| **Partition Coefficient** | Độ rõ ràng của phân cụm | Càng cao càng tốt (0.7-1.0) |
| **Classification Entropy** | Độ không chắc chắn | Càng thấp càng tốt (< 1.0) |
| **Silhouette Score** | Chất lượng tổng thể | Càng cao càng tốt (0.5-1.0) |
| **Calinski-Harabasz** | Tỷ lệ phân tán | Càng cao càng tốt |

### Ví dụ kết quả trên Iris dataset

```
📊 Kết quả phân cụm Iris (3 cụm):
- Davies-Bouldin Index: 0.667 ✅
- Partition Coefficient: 0.823 ✅  
- Silhouette Score: 0.681 ✅
- Thời gian xử lý: 0.045s
- Số bước lặp: 23
```

### Ví dụ kết quả ảnh viễn thám

```
🛰️ Kết quả phân cụm ảnh Hà Nội (6 cụm):
- Kích thước: 1024x1024 pixels
- Số kênh phổ: 4 (B1, B2, B3, B4)
- Thời gian xử lý: 224.362s
- Số bước lặp: 155
- Davies-Bouldin: 1.049
- Partition Coefficient: 0.461
```

## ⚙️ Tùy chỉnh nâng cao

### Thay đổi tham số FCM

```python
# Phân cụm "cứng" hơn (m thấp)
fcm_hard = Dfcm(n_clusters=3, m=1.5)

# Phân cụm "mềm" hơn (m cao)  
fcm_soft = Dfcm(n_clusters=3, m=3.0)

# Hội tụ nhanh hơn (epsilon cao)
fcm_fast = Dfcm(n_clusters=3, epsilon=1e-3)

# Chính xác hơn (epsilon thấp)
fcm_precise = Dfcm(n_clusters=3, epsilon=1e-6)
```

### Xử lý ảnh lớn

```python
# Giảm kích thước ảnh để xử lý nhanh hơn
from skimage.transform import resize

def downsample_image(X, height, width, factor=2):
    """Giảm kích thước ảnh đi factor lần"""
    new_height = height // factor
    new_width = width // factor
    
    # Reshape và resize
    image = X.reshape(height, width, -1)
    resized = resize(image, (new_height, new_width), anti_aliasing=True)
    
    return resized.reshape(-1, X.shape[1]), new_height, new_width
```

## 🐛 Troubleshooting

### Lỗi thường gặp

#### 1. Lỗi thiếu module
```bash
ModuleNotFoundError: No module named 'xxx'
```
**Giải pháp:**
```bash
pip install -r requirements.txt
```

#### 2. Lỗi bộ nhớ với ảnh lớn
```
MemoryError: Unable to allocate array
```
**Giải pháp:**
- Giảm kích thước ảnh
- Tăng RAM hệ thống
- Sử dụng `max_iter` thấp hơn

#### 3. Phân cụm không hội tụ
```
UserWarning: FCM did not converge
```
**Giải pháp:**
- Tăng `max_iter`
- Giảm `epsilon` 
- Thay đổi `seed`

#### 4. Streamlit không chạy
```bash
streamlit: command not found
```
**Giải pháp:**
```bash
pip install streamlit
# hoặc
python -m streamlit run clustering_app.py
```

### Tips tối ưu hiệu suất

1. **Cho dữ liệu CSV:**
   - Chuẩn hóa dữ liệu trước khi phân cụm
   - Sử dụng PCA nếu có quá nhiều features

2. **Cho ảnh viễn thám:**
   - Resize ảnh xuống 512x512 cho test nhanh
   - Sử dụng `normalize=True`
   - Chọn số cụm phù hợp (4-8 cụm)

## 📈 Roadmap

### Phiên bản tương lai

- [ ] 🚀 Hỗ trợ GPU acceleration
- [ ] 📱 Mobile-responsive UI
- [ ] 🔄 Batch processing nhiều ảnh
- [ ] 🎯 Thuật toán phân cụm khác (K-means, DBSCAN)
- [ ] 📊 Xuất báo cáo PDF/Word
- [ ] 🌐 Deploy lên cloud (Heroku, AWS)

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! 

### Cách đóng góp:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

### Guidelines:

- Tuân thủ PEP 8 style guide
- Thêm docstring cho functions/classes
- Viết unit tests cho code mới
- Cập nhật README nếu cần

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Authors

- **NCKH Team** - *Initial work*
- Liên hệ: [email@example.com]

## 🙏 Acknowledgments

- Thuật toán FCM dựa trên paper của Bezdek (1981)
- Streamlit framework cho giao diện web
- Scikit-learn cho các metrics đánh giá
- Plotly cho visualization

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng:

1. Kiểm tra [Troubleshooting](#-troubleshooting)
2. Tạo [Issue](https://github.com/your-repo/issues)
3. Liên hệ team qua email

---

⭐ **Nếu project hữu ích, đừng quên star repo!** ⭐
