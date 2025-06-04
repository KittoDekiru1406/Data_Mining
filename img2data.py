import os
import numpy as np
from rasterio import open as rasterio_open


class Dimg2data:

    @staticmethod
    def __standard_scaler(bands: list, normalize: bool = False) -> np.ndarray:
        result = np.stack(bands, axis=-1).reshape(-1, len(bands))
        return Dimg2data.map_to_0_1(result) if normalize else result

    def read_multi_band_image(self, imgpath: str, normalize: bool = True) -> tuple:
        with rasterio_open(imgpath) as src:
            bands = src.read()
            height, width = src.height, src.width
            imd = self.__standard_scaler(list(bands), normalize=normalize)
        return imd, height, width

    def read_multi_band_directory(self, directory: str, normalize: bool = True) -> tuple:
        # Lấy danh sách tất cả các file ảnh trong thư mục
        image_files = [f for f in os.listdir(directory) if f.endswith(('.tif', '.jpg', '.png'))]
        image_files.sort()  # Sắp xếp để đảm bảo thứ tự nhất quán

        # Đọc từng ảnh và đưa vào ma trận dữ liệu
        _bands, height, width = [], 0, 0
        for image_file in image_files:
            imgpath = os.path.join(directory, image_file)
            with rasterio_open(imgpath) as src:
                _bands.append(src.read())
                if width == 0:
                    height, width = src.height, src.width
        # bands = np.concatenate(_bands, axis=0)
        imd = self.__standard_scaler(_bands, normalize=normalize)
        return imd, height, width

    def read_single_band_images(self, imgpaths: list, normalize: bool = False) -> dict:
        result = {'height': 0, 'width': 0, 'bands': []}
        for imgpath in imgpaths:
            with rasterio_open(imgpath) as src:
                _bands = src.read()
                if result['width'] == 0:
                    result.update({'height': src.height, 'width': src.width})
                result['bands'].append(self.__standard_scaler([_bands], normalize=normalize))
        return result

    # Hàm ánh xạ 1 điểm từ khoảng [-1,1] sang [0,255]
    @staticmethod
    def map_to_0_255(data: np.ndarray) -> np.ndarray:
        return ((data + 1) * 255 / 2).astype(np.uint8)

    # Hàm ánh xạ 1 điểm từ khoảng [-1,1] sang [0,255]
    @staticmethod
    def map_to_0_1(data: np.ndarray) -> np.ndarray:
        return data.astype(np.float64) / np.max(data)
