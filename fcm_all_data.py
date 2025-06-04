import time
import os 
from fcm import Dfcm 
from utility import round_float, extract_labels, extract_clusters, TEST_CASES, export_to_latex_data
from validity import davies_bouldin, partition_coefficient, classification_entropy, separation, calinski_harabasz, silhouette, hypervolume, cs


if __name__ == "__main__":
    # ------------------------------------------
    maxiter = 10000
    m = 2
    epsilon = 1e-5
    seed = 42
    DATA_ID = [53, 109, 602]
    metrics = []
    num_processes = 3
    folder_path = 'data/csv'
    # ------------------------------------------
    _start_time = time.time()
    
    if not os.path.isdir(folder_path):
        print(f'Directory path {folder_path} does not exist')
        exit()
    
    # Map CSV files to their corresponding DATA_IDs
    csv_file_mapping = {
        'Iris.csv': 53,
        'Wine.csv': 109,
        'DryBean.csv': 602
    }
    
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    for csvfile in csv_files:
        if csvfile not in csv_file_mapping:
            print(f'Unknown CSV file: {csvfile}')
            continue
            
        ID = csv_file_mapping[csvfile]
        csv_path = os.path.join(folder_path, csvfile)
        
        # Load CSV data directly
        import pandas as pd
        df = pd.read_csv(csv_path)
        _dt = {
            'X': df.iloc[:, :-1].values,  # All columns except last (features)
            'Y': df.iloc[:, -1:].values   # Last column (labels/target)
        }
        
        C = TEST_CASES[ID]['n_cluster']
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        # --------------------------------
        _start_time = time.time()
        # Nối tiếp 
        dfcm = Dfcm(C, m, epsilon, maxiter)
        U, V, step = dfcm.fit(_dt['X'], seed)
        labels = extract_labels(U)
        clusters = extract_clusters(_dt['X'], labels, C)
        metric_nt = {
        'FCM': f"{ID} nt ",
        'Size': f"{_dt['X'].shape[0]}x{_dt['X'].shape[1]}",
        'C': C,
        'Time': round_float(time.time() - _start_time),
        'DB': round_float(davies_bouldin(_dt['X'], labels)) ,
        'PC': round_float(partition_coefficient(U)) ,
        'CE': round_float(classification_entropy(U)) ,
        'S': round_float(separation(_dt['X'], U, V, m)) ,
        'CH': round_float(calinski_harabasz(_dt['X'], labels)) ,
        'SI': round_float(silhouette(_dt['X'], labels)) ,
        'FHV': round_float(hypervolume(U, m)) ,
        'CS': round_float(cs(_dt['X'], U, V, m)) 
        }
        metrics.append(metric_nt)
        
        
    export_to_latex_data(metrics, 'outputs/logs/fcm_data.txt')
    print("Metrics exported to fcm_data.txt")