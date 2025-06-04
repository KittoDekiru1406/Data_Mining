import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from PIL import Image
import io

# Import your clustering modules
from fcm import Dfcm
from img2data import Dimg2data
from imgsegment import DimgSegment
from utility import round_float, extract_labels, extract_clusters, TEST_CASES
from validity import (davies_bouldin, partition_coefficient, classification_entropy, 
                     separation, calinski_harabasz, silhouette, hypervolume, cs)

# Page configuration
st.set_page_config(
    page_title="🔬 Ứng dụng Phân cụm FCM",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(45deg, #f0f8ff, #e6f3ff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background: linear-gradient(45deg, #f0fff0, #e6ffe6);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background: linear-gradient(45deg, #fff8dc, #ffeaa7);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def load_csv_data(csv_file, data_id):
    """Load CSV data for clustering"""
    try:
        df = pd.read_csv(csv_file)
        return {
            'X': df.iloc[:, :-1].values,
            'Y': df.iloc[:, -1:].values,
            'df': df,
            'feature_names': df.columns[:-1].tolist(),
            'target_name': df.columns[-1]
        }
    except Exception as e:
        st.error(f"Lỗi khi đọc file CSV: {e}")
        return None

def run_fcm_clustering(data, n_clusters, m=2, epsilon=1e-5, max_iter=10000, seed=42):
    """Run FCM clustering"""
    with st.spinner("🔄 Đang thực hiện phân cụm FCM..."):
        start_time = time.time()
        
        fcm = Dfcm(n_clusters=n_clusters, m=m, epsilon=epsilon, max_iter=max_iter)
        U, V, steps = fcm.fit(data, seed=seed)
        
        processing_time = time.time() - start_time
        labels = extract_labels(U)
        
        return {
            'U': U,
            'V': V,
            'labels': labels,
            'steps': steps,
            'time': processing_time
        }

def calculate_metrics(X, U, V, labels, m=2):
    """Calculate clustering metrics"""
    try:
        clusters = extract_clusters(X, labels, len(V))
        
        metrics = {
            'Davies-Bouldin Index': round_float(davies_bouldin(X, labels)),
            'Partition Coefficient': round_float(partition_coefficient(U)),
            'Classification Entropy': round_float(classification_entropy(U)),
            'Separation': round_float(separation(X, U, V, m)),
            'Calinski-Harabasz Index': round_float(calinski_harabasz(X, labels)),
            'Silhouette Score': round_float(silhouette(X, labels)),
            'Fuzzy Hypervolume': round_float(hypervolume(U, m)),
            'Compactness & Separation': round_float(cs(X, U, V, m))
        }
        return metrics
    except Exception as e:
        st.error(f"Lỗi khi tính toán metrics: {e}")
        return {}

def display_metrics(metrics, processing_time, steps):
    """Display clustering metrics in a beautiful format"""
    st.markdown('<div class="sub-header">📊 Kết quả Phân cụm</div>', unsafe_allow_html=True)
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⏱️ Thời gian xử lý", f"{processing_time:.3f}s")
    with col2:
        st.metric("🔄 Số bước lặp", f"{steps}")
    with col3:
        st.metric("✅ Trạng thái", "Hoàn thành")
    
    # Clustering quality metrics
    st.markdown("### 🎯 Chỉ số Đánh giá Chất lượng Phân cụm")
    
    metric_cols = st.columns(2)
    metric_items = list(metrics.items())
    
    for i, (metric_name, value) in enumerate(metric_items):
        col = metric_cols[i % 2]
        with col:
            # Create metric card
            if "Davies-Bouldin" in metric_name or "Classification Entropy" in metric_name:
                trend = "📉 Càng thấp càng tốt"
                color = "#dc3545" if value > 1 else "#28a745"
            else:
                trend = "📈 Càng cao càng tốt"
                color = "#28a745" if value > 0.5 else "#dc3545"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: {color}; margin: 0;">{metric_name}</h4>
                <h2 style="margin: 0.5rem 0;">{value}</h2>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">{trend}</p>
            </div>
            """, unsafe_allow_html=True)

def plot_clustering_results(data_dict, labels, centroids):
    """Plot clustering results"""
    X = data_dict['X']
    df = data_dict['df']
    
    if X.shape[1] >= 2:
        # 2D scatter plot
        fig = px.scatter(
            x=X[:, 0], y=X[:, 1], 
            color=labels.astype(str),
            title="🎯 Kết quả Phân cụm (2D)",
            labels={'x': data_dict['feature_names'][0], 'y': data_dict['feature_names'][1]},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Add centroids
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x', line=dict(width=2, color='black')),
            name='Centroids'
        ))
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # If more than 2 features, show 3D plot
        if X.shape[1] >= 3:
            fig_3d = px.scatter_3d(
                x=X[:, 0], y=X[:, 1], z=X[:, 2],
                color=labels.astype(str),
                title="🎯 Kết quả Phân cụm (3D)",
                labels={
                    'x': data_dict['feature_names'][0], 
                    'y': data_dict['feature_names'][1],
                    'z': data_dict['feature_names'][2]
                },
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_3d.add_trace(go.Scatter3d(
                x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
                mode='markers',
                marker=dict(size=10, color='red', symbol='x'),
                name='Centroids'
            ))
            
            fig_3d.update_layout(height=600)
            st.plotly_chart(fig_3d, use_container_width=True)

def process_remote_sensing_image(image_dir, n_clusters=6, m=2, epsilon=1e-5, max_iter=10000):
    """Process remote sensing images"""
    try:
        img_processor = Dimg2data()
        
        # Read multi-band images
        with st.spinner("📡 Đang đọc ảnh viễn thám..."):
            X, height, width = img_processor.read_multi_band_directory(image_dir, normalize=True)
            st.success(f"✅ Đã đọc ảnh thành công: {height}x{width} pixels, {X.shape[1]} kênh phổ")
        
        # Run FCM clustering
        result = run_fcm_clustering(X, n_clusters, m, epsilon, max_iter)
        
        # Calculate metrics
        metrics = calculate_metrics(X, result['U'], result['V'], result['labels'], m)
        
        # Create segmented image
        with st.spinner("🎨 Đang tạo ảnh phân đoạn..."):
            img_segment = DimgSegment()
            
            # Save segmented image to memory
            temp_path = "/tmp/segmented_image.jpg"
            img_segment.save_label2image(result['labels'], height, width, temp_path)
            
            # Load the image
            segmented_img = Image.open(temp_path)
        
        return {
            'original_shape': (height, width, X.shape[1]),
            'segmented_image': segmented_img,
            'metrics': metrics,
            'processing_time': result['time'],
            'steps': result['steps'],
            'labels': result['labels'],
            'centroids': result['V']
        }
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh viễn thám: {e}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">🔬 Ứng dụng Phân cụm FCM</div>', unsafe_allow_html=True)
    st.markdown("### 🎯 Phân cụm Dữ liệu và Ảnh Viễn thám với Fuzzy C-Means")
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Cấu hình")
    
    # Data type selection
    data_type = st.sidebar.selectbox(
        "📂 Chọn loại dữ liệu:",
        ["📊 Dữ liệu CSV", "🛰️ Ảnh viễn thám"]
    )
    
    # FCM parameters
    st.sidebar.markdown("### 🔧 Tham số FCM")
    n_clusters = st.sidebar.slider("Số cụm", 2, 10, 3)
    m = st.sidebar.slider("Hệ số mờ (m)", 1.1, 3.0, 2.0, 0.1)
    epsilon = st.sidebar.select_slider("Ngưỡng hội tụ", [1e-6, 1e-5, 1e-4, 1e-3], 1e-5)
    max_iter = st.sidebar.slider("Số vòng lặp tối đa", 100, 10000, 1000, 100)
    
    if data_type == "📊 Dữ liệu CSV":
        st.markdown("## 📊 Phân cụm Dữ liệu CSV")
        
        # Dataset selection
        dataset_options = {
            "🌸 Iris Dataset": ("data/csv/Iris.csv", 53),
            "🍷 Wine Dataset": ("data/csv/Wine.csv", 109),
            "🫘 Dry Bean Dataset": ("data/csv/DryBean.csv", 602)
        }
        
        selected_dataset = st.selectbox("Chọn bộ dữ liệu:", list(dataset_options.keys()))
        csv_file, data_id = dataset_options[selected_dataset]
        
        if st.button("🚀 Bắt đầu Phân cụm", type="primary"):
            # Load data
            data_dict = load_csv_data(csv_file, data_id)
            
            if data_dict is not None:
                # Show data info
                st.markdown("### 📋 Thông tin Dữ liệu")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📏 Số mẫu", data_dict['X'].shape[0])
                with col2:
                    st.metric("🔢 Số đặc trưng", data_dict['X'].shape[1])
                with col3:
                    st.metric("🎯 Số cụm", n_clusters)
                
                # Show sample data
                st.markdown("### 👀 Xem trước Dữ liệu")
                st.dataframe(data_dict['df'].head(10), use_container_width=True)
                
                # Run clustering
                result = run_fcm_clustering(data_dict['X'], n_clusters, m, epsilon, max_iter)
                
                # Calculate metrics
                metrics = calculate_metrics(data_dict['X'], result['U'], result['V'], result['labels'], m)
                
                # Display results
                display_metrics(metrics, result['time'], result['steps'])
                
                # Plot results
                st.markdown("### 📈 Biểu đồ Phân cụm")
                plot_clustering_results(data_dict, result['labels'], result['V'])
                
                # Cluster distribution
                st.markdown("### 📊 Phân bố Cụm")
                cluster_counts = pd.Series(result['labels']).value_counts().sort_index()
                fig_bar = px.bar(
                    x=cluster_counts.index, 
                    y=cluster_counts.values,
                    title="Số lượng điểm trong mỗi cụm",
                    labels={'x': 'Cụm', 'y': 'Số lượng điểm'},
                    color=cluster_counts.values,
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
    
    else:  # Remote sensing images
        st.markdown("## 🛰️ Phân cụm Ảnh Viễn thám")
        
        # Image dataset selection
        image_options = {
            "🏙️ Hà Nội": "data/images/Anh-ve-tinh/Anh-da-pho/HaNoi",
            "🌾 Đông Nam Bộ": "data/images/Anh-ve-tinh/Anh-da-pho/DongNamBo"
        }
        
        selected_location = st.selectbox("Chọn khu vực:", list(image_options.keys()))
        image_dir = image_options[selected_location]
        
        if st.button("🚀 Bắt đầu Phân cụm Ảnh", type="primary"):
            # Check if directory exists
            if not os.path.exists(image_dir):
                st.error(f"❌ Thư mục {image_dir} không tồn tại!")
                return
            
            # Process image
            result = process_remote_sensing_image(image_dir, n_clusters, m, epsilon, max_iter)
            
            if result is not None:
                # Display results
                display_metrics(result['metrics'], result['processing_time'], result['steps'])
                
                # Display images
                st.markdown("### 🖼️ So sánh Ảnh Trước và Sau Phân cụm")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📷 Ảnh Gốc")
                    # For original image, we'll show the first band or create a composite
                    try:
                        # Try to load a composite image if available
                        composite_path = f"data/images/Anh-ve-tinh/Anh-mau/{selected_location.split()[1]}.tif"
                        if os.path.exists(composite_path):
                            original_img = Image.open(composite_path)
                            st.image(original_img, caption="Ảnh viễn thám gốc", use_container_width=True)
                        else:
                            st.info("💡 Ảnh gốc tổng hợp không có sẵn. Hiển thị ảnh phân đoạn.")
                    except:
                        st.info("💡 Không thể hiển thị ảnh gốc.")
                
                with col2:
                    st.markdown("#### 🎨 Ảnh Sau Phân cụm")
                    st.image(
                        result['segmented_image'], 
                        caption=f"Ảnh phân đoạn thành {n_clusters} cụm",
                        use_container_width=True
                    )
                
                # Image info
                st.markdown("### ℹ️ Thông tin Ảnh")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📐 Kích thước", f"{result['original_shape'][0]}x{result['original_shape'][1]}")
                with col2:
                    st.metric("🌈 Số kênh phổ", result['original_shape'][2])
                with col3:
                    st.metric("🎯 Số cụm", n_clusters)
                
                # Cluster statistics
                st.markdown("### 📊 Thống kê Phân cụm")
                cluster_counts = pd.Series(result['labels']).value_counts().sort_index()
                total_pixels = len(result['labels'])
                
                cluster_stats = pd.DataFrame({
                    'Cụm': cluster_counts.index,
                    'Số pixel': cluster_counts.values,
                    'Tỷ lệ (%)': (cluster_counts.values / total_pixels * 100).round(2)
                })
                
                st.dataframe(cluster_stats, use_container_width=True)
                
                # Cluster distribution chart
                fig_pie = px.pie(
                    values=cluster_counts.values,
                    names=[f"Cụm {i}" for i in cluster_counts.index],
                    title="Phân bố tỷ lệ các cụm",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔬 **Ứng dụng Phân cụm FCM** - Phát triển bởi NCKH Team | "
        "🚀 Powered by Streamlit & Fuzzy C-Means Algorithm"
    )

if __name__ == "__main__":
    main()
