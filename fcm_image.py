from validity import partition_coefficient, classification_entropy, separation, calinski_harabasz, hypervolume, cs, davies_bouldin
from fcm import Dfcm
from img2data import Dimg2data
from utility import extract_labels, round_float, name_slug, export_to_latex_image
import time
import os
import numpy as np

# ==============================================================================================================

def fuzzy_draw_image(X: np.ndarray, V: np.ndarray, U: np.ndarray, height: int, width: int, title: str, output_folder: str) -> str:
    labels = extract_labels(U)
    _img_fileio = os.path.join(output_folder, f'{name_slug(title)}.jpg')
    os.makedirs(output_folder, exist_ok=True)
    from imgsegment import DimgSegment
    DimgSegment().save_label2image(labels=labels, height=height, width=width, fileio=_img_fileio)
    metrics = {
        'DB': round_float(davies_bouldin(X, labels), n=ROUND_FLOAT),
        'PC': round_float(partition_coefficient(U), n=ROUND_FLOAT),
        'CE': round_float(classification_entropy(U), n=ROUND_FLOAT),
        'S': round_float(separation(X, U, V, M), n=0),
        'CH': round_float(calinski_harabasz(X, labels), n=0),
        'FHV': round_float(hypervolume(U, M), n=ROUND_FLOAT),
        'CS': round_float(cs(X, U, V, M), n=ROUND_FLOAT)
    }
    return metrics


if __name__ == '__main__':
    IMAGE_DATA_PATH = 'data/images'
    OUTPUT_FOLDER = 'outputs/images'
    REPORT_FILE = 'outputs/logs/remote_sensing_clustering_report.txt'
    DIR_NAME = 'Anh-ve-tinh/Anh-da-pho/HaNoi'
    ROUND_FLOAT = 3
    M = 2
    N_CLUSTERS = 6
    EPSILON = 1e-5
    MAXITER = 10000
    dir_path = os.path.join(IMAGE_DATA_PATH, DIR_NAME)

    # ==================================
    img_processor = Dimg2data()
    start_time = time.time()
    print("So cum=", N_CLUSTERS)
    # -------------------
    try:
        X, height, width = img_processor.read_multi_band_directory(dir_path, normalize=True)
        print(f"Kích thước dữ liệu tổng hợp: {height}x{width}, Số kênh: {X.shape[1]}")
        print(f"Thời gian đọc dữ liệu: {round_float(time.time() - start_time)}s")
    except Exception as e:
        print(f"Lỗi khi đọc thư mục {dir_path}: {e}")
        exit()

    # ===============================================
    start_time = time.time()
    fcm = Dfcm(n_clusters=N_CLUSTERS, m=M, epsilon=EPSILON, max_iter=MAXITER)
    U, V, steps = fcm.fit(X, seed=42)
    processing_time = round_float(time.time() - start_time)
    print(f"Thời gian phân cụm FCM: {processing_time}s, {steps} bước lặp")

    title = f'FCM'
    base_metrics = fuzzy_draw_image(X=X, V=V, U=U, height=height, width=width, title=title, output_folder=OUTPUT_FOLDER)   
    print('\t', base_metrics)
    print('U', U.shape, U[0])
    print('V', V.shape, V[0])
    
    # Prepare metrics for LaTeX export (add missing keys)
    full_metrics = {
        'FCM': f'FCM C={N_CLUSTERS}',
        'Time': processing_time,
        **base_metrics  # Merge the base metrics
    }
    
    # export_to_latex_image expects a list of dictionaries
    export_to_latex_image([full_metrics], REPORT_FILE)

    