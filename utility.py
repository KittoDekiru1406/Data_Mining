import os
import re
import json
import numpy as np
import pandas as pd
from urllib import request, parse, error
import certifi
import ssl

TEST_CASES = {
    14: {
        'name': 'BreastCancer',
        'n_cluster': 2,
        'test_points': ['30-39', 'premeno', '30-34', '0-2', 'no', 3, 'left', 'left_low', 'no']
    },
    53: {
        'name': 'Iris',
        'n_cluster': 3,
        'test_points': [5.1, 3.5, 1.4, 0.2]
    },
    80: {
        'name': 'Digits',
        'n_cluster': 10,
        'test_points': [0, 1, 6, 15, 12, 1, 0, 0, 0, 7, 16, 6, 6, 10, 0, 0, 0, 8, 16, 2, 0, 11, 2, 0, 0, 5, 16, 3, 0, 5, 7, 0, 0, 7, 13, 3, 0, 8, 7, 0, 0, 4, 12, 0, 1, 13, 5, 0, 0, 0, 14, 9, 15, 9, 0, 0, 0, 0, 6, 14, 7, 1, 0, 0]
    },
    109: {
        'name': 'Wine',
        'n_cluster': 3,
        'test_points': [14.23, 1.71, 2.43, 15.6,
                        127, 2.80, 3.06, 0.28,
                        2.29, 5.64, 1.04, 3.92,
                        1065]
    },
    236: {
        'name': 'Seeds',
        'n_cluster': 3,
        'test_points': [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22]
    },
    602: {
        'name': 'DryBean',
        'n_cluster': 7,
        'test_points': [
            28395, 610.291, 208.178117, 173.888747,
            1.197191, 0.549812, 28715, 190.141097,
            0.763923, 0.988856, 0.958027, 0.913358,
            0.007332, 0.003147, 0.834222, 0.998724]
    }
}


# Mã hóa nhãn
class LabelEncoder:
    def __init__(self):
        self.index_to_label = {}
        self.unique_labels = None

    @property
    def classes_(self) -> np.ndarray:
        return self.unique_labels

    def fit_transform(self, labels) -> np.ndarray:
        self.unique_labels = np.unique(labels)
        label_to_index = {label: index for index, label in enumerate(self.unique_labels)}
        self.index_to_label = {index: label for label, index in label_to_index.items()}
        return np.array([label_to_index[label] for label in labels])

    def inverse_transform(self, indices) -> np.ndarray:
        return np.array([self.index_to_label[index] for index in indices])


class Read_NC():
    def __init__(self, file_path: str, factor: int = 4, n_days: int = 1):       
        self.file_path = file_path  
        self.factor = factor
        self.data = None
        self.n_days = n_days  
    
    def get_time(self) -> np.ndarray:   
        import xarray as xr
        ds = xr.open_dataset(self.file_path)    
        return ds['time'].values
    
    def average_downsampling(self) -> np.ndarray:
        n_valid_time, n_lat, n_lon = self.data.shape

        # Tính kích thước mới
        new_lat = n_lat // self.factor
        new_lon = n_lon // self.factor
        
        # Tạo mảng kết quả
        downsampled_data = np.empty((n_valid_time, new_lat, new_lon))
        
        # Lặp qua các lưới con
        for i in range(new_lat):
            for j in range(new_lon):
                # Tính các chỉ số cho điểm dữ liệu hiện tại
                start_lat = i * self.factor
                end_lat = min(start_lat + self.factor, n_lat)
                start_lon = j * self.factor
                end_lon = min(start_lon + self.factor, n_lon)
                
                # Lấy các điểm dữ liệu nằm trong lưới con
                values_to_average = self.data[:, start_lat:end_lat, start_lon:end_lon]
                
                # Tính giá trị trung bình
                downsampled_data[:, i, j] = np.mean(values_to_average, axis=(1, 2))
        
        return downsampled_data

    def nc_to_numpy(self) -> np.ndarray:
        # Đọc file .nc
        import xarray as xr
        ds = xr.open_dataset(self.file_path)    

        # Trích xuất dữ liệu từ biến t2m
        self.data = ds['t2m'].values
        
        self.data= self.average_downsampling()

        n_valid_time, n_lat, n_lon = self.data.shape
        # self.data = self.data.reshape(n_valid_time * n_lat * n_lon, -1)
        self.data = self.data[:int(self.n_days*24), :, :].reshape(int(self.n_days*24) * n_lat * n_lon, -1)
        return self.data


def random_negative_assignment(labels: np.ndarray, ratio: float = 0.3, val: float = -1) -> np.ndarray:
    _length = len(labels)
    neg_count = int(ratio * _length)
    random_indices = np.random.choice(_length, neg_count, replace=False)
    result = np.copy(labels)
    result[random_indices] = val
    return result


def split_data_for_semi_supervised_learning(data: np.ndarray, labels: np.ndarray, n_sites: int = 3, ratio: float = 0.3, val: float = -1) -> list:
    result = []
    datas = np.array_split(data, n_sites)
    labeled = np.array_split(labels, n_sites)
    for i, data in enumerate(datas):
        y_true = labeled[i]
        y_lble = random_negative_assignment(labels=y_true, ratio=ratio, val=val)
        result.append({'X': data, 'Y': y_lble, 'T': y_true})
    return result


def name_slug(text: str, delim: str = '-') -> str:
    __punct_re = re.compile(r'[\t !’"“”#@$%&~\'()*\+:;\-/<=>?\[\\\]^_`{|},.]+')
    if text:
        from unidecode import unidecode
        result = [unidecode(word) for word in __punct_re.split(text.lower()) if word]
        result = [rs if rs != delim and rs.isalnum() else '' for rs in result]
        return re.sub(r'\s+', delim, delim.join(result).strip())


# Làm tròn số
def round_float(number: float, n: int = 3) -> float:
    if n == 0:
        return int(number)
    return round(number, n)


# Ma trận độ thuộc ra nhãn (giải mờ)
def extract_labels(membership: np.ndarray) -> np.ndarray:
    return np.argmax(membership, axis=1).astype(np.int8)


# Chia các điểm vào các cụm
def extract_clusters(data: np.ndarray, labels: np.ndarray, n_cluster: int = 0) -> list:
    if n_cluster == 0:
        n_cluster = np.unique(labels)
    return [data[labels == i] for i in range(n_cluster)]


# Chuẩn Euclidean của một vector đo lường độ dài của vector
# là căn bậc hai của tổng bình phương các phần tử của vector đó.
# d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
def norm_distances(A: np.ndarray, B: np.ndarray, axis: int = None) -> float:
    # np.sqrt(np.sum((np.asarray(A) - np.asarray(B)) ** 2))
    # np.sum(np.abs(np.array(A) - np.array(B)))
    return np.linalg.norm(A - B, axis=axis)


# Tổng bình phương của hiệu khoảng cách giữa 2 ma trận trên tất cả các chiều
def euclidean_distance(XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    # Hiệu giữa các điểm trong XA và XB
    differences = XA[:, np.newaxis, :] - XB[np.newaxis, :, :]
    return np.sum(differences ** 2, axis=2)


# Ma trận khoảng cách Euclide giữa các điểm trong 2 tập hợp dữ liệu
def euclidean_cdist(XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    # _df = euclidean_distance(XA,XB)
    # return np.sqrt(_df)
    from scipy.spatial.distance import cdist
    return cdist(XA, XB)


# lấy giá trị lớn nhất để tránh lỗi chia cho 0
def not_division_by_zero(data: np.ndarray):
    return np.fmax(data, np.finfo(np.float64).eps)


# Chuẩn hóa mỗi hàng của ma trận sao cho tổng của mỗi hàng bằng 1.
# \mathbf{x}_{norm} = \frac{\mathbf{x}}{\sum_{i=1}^m \mathbf{x}_{i,:}}
def standardize_rows(data: np.ndarray) -> np.ndarray:
    # Ma trận tổng của mỗi cột (cùng số chiều)
    _sum = np.sum(data, axis=0, keepdims=1)
    # Chia từng phần tử của ma trận cho tổng tương ứng của cột đó.
    return data / _sum


# Đếm số lần xuất hiện của từng phần tử trong 1 mảng
def count_data_array(data: np.ndarray) -> dict:
    unique, counts = np.unique(data, return_counts=True)
    return dict(zip(unique, counts))


# Giảm chiều PCA
def pca(data: np.ndarray, k: int, sklearn: bool = True) -> np.ndarray:
    if sklearn:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=k)
        return pca.fit_transform(data)

    data_mean = np.mean(data, axis=0)
    data_bar = data - data_mean
    covariance_matrix = np.cov(data_bar, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    principal_components = eigenvectors[:, :k]
    return data_bar @ principal_components


def load_dataset(data: dict, file_csv: str = '', header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    print('uci_id=', data['data']['uci_id'])  # Mã bộ dữ liệu
    print('data name=', data['data']['name'])  # Tên bộ dữ liệu
    print('data abstract=', data['data']['abstract'])  # Tên bộ dữ liệu
    print('feature types=', data['data']['feature_types'])  # Kiểu nhãn
    print('num instances=', data['data']['num_instances'])  # Số lượng điểm dữ liệu
    print('num features=', data['data']['num_features'])  # Số lượng đặc trưng
    metadata = data['data']
    df = pd.read_csv(file_csv if file_csv != '' else metadata['data_url'], header=header, index_col=index_col, usecols=usecols, nrows=nrows)
    return {'data': data['data'], 'ALL': df.iloc[:, :].values, 'X': df.iloc[:, :-1].values, 'Y': df.iloc[:, -1:].values}


# Lấy dữ liệu từ ổ cứng
def fetch_data_from_local(name_or_id=53, folder: str = 'data/csv', header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    if isinstance(name_or_id, str):
        name = name_or_id
    else:
        name = TEST_CASES[name_or_id]['name']
    _folder = os.path.join(folder, name)
    fileio = os.path.join(_folder, 'api.json')
    if not os.path.isfile(fileio):
        print(f'File {fileio} not found!')
    with open(fileio, 'r') as cr:
        response = cr.read()
    return load_dataset(json.loads(response),
                        file_csv=os.path.join(_folder, 'data.csv'),
                        header=header, index_col=index_col, usecols=usecols, nrows=nrows)


# Lấy dữ liệu từ ISC UCI (53: Iris, 602: DryBean, 109: Wine)
def fetch_data_from_uci(name_or_id=53, header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    api_url = 'https://archive.ics.uci.edu/api/dataset'
    if isinstance(name_or_id, str):
        api_url += '?name=' + parse.quote(name_or_id)
    else:
        api_url += '?id=' + str(name_or_id)
    try:
        _rs = request.urlopen(api_url, context=ssl.create_default_context(cafile=certifi.where()))
        response = _rs.read()
        _rs.close()
        return load_dataset(json.loads(response),
                            header=header, index_col=index_col, usecols=usecols, nrows=nrows)
    except (error.URLError, error.HTTPError):
        raise ConnectionError('Error connecting to server')

def export_to_latex_data(metrics, filename):
    latex_code = r"""
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{FCM} & \textbf{Size} & \textbf{C} & \textbf{Time} & \textbf{DB$\triangledown$} & \textbf{PC$\Delta$} & \textbf{CE$\triangledown$} & \textbf{S$\triangledown$} & \textbf{CH$\Delta$} & \textbf{SI$\Delta$} & \textbf{FHV$\Delta$} & \textbf{CS$\triangledown$} \\ \hline
"""
    for metric in metrics:
        latex_code += f"{metric['FCM']} & {metric['Size']} & {metric['C']} & {metric['Time']} & {metric['DB']} & {metric['PC']} & {metric['CE']} & {metric['S']} & {metric['CH']} & {metric['SI']} & {metric['FHV']} & {metric['CS']} \\\\ \\hline\n"

    latex_code += r"""
\end{tabular}
"""
    with open(filename, "w") as file:
        file.write(latex_code)
        
def export_to_latex_image(metrics, filename):
    latex_code = r"""
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{FCM} & \textbf{Time} & \textbf{DB$\triangledown$} & \textbf{PC$\Delta$} & \textbf{CE$\triangledown$} & \textbf{S$\triangledown$} & \textbf{CH$\Delta$} & \textbf{FHV$\Delta$} & \textbf{CS$\triangledown$} \\ \hline
"""
    for metric in metrics:
        latex_code += f"{metric['FCM']} & {metric['Time']}  & {metric['DB']} & {metric['PC']} & {metric['CE']} & {metric['S']} & {metric['CH']} & {metric['FHV']} & {metric['CS']} \\\\ \\hline\n"

    latex_code += r"""
\end{tabular}
"""
    with open(filename, "w") as file:
        file.write(latex_code)