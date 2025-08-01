import cv2
import numpy as np
# plsa_color utils
class PLSAColorClassifier:
    def __init__(self, w2cM_path='w2cM.xml'):
        self.plsa_color_names = ['Black', 'Blue', 'Brown', 'Gray', 
                                 'Green', 'Orange', 'Pink', 'Purple', 
                                 'Red', 'White', 'Yellow']
        self.category_map = {
            'Red': 0,
            'Green': 1,
            'Blue': 2,
            'Black': 3,
            'White': 4,
            'Gray': 5,
            'Pink': 6,
            'Orange': 7,
            'Purple': 8,
            'Cyan': 9,
            'Yellow': 10,
            'Brown': 11
        }
        self.w2cM = self.load_w2cM(w2cM_path)

    def floor_mat(self, double_mat):
        return np.floor_divide(double_mat.astype(np.float64), 8).astype(np.int32)

    def load_w2cM(self, w2cM_path):
        """
        加载 w2cM 映射矩阵。

        Args:
            w2cM_path (str): w2cM 矩阵文件的路径。

        Returns:
            np.ndarray: 加载的 w2cM 矩阵。
        """
        fs = cv2.FileStorage(w2cM_path, cv2.FILE_STORAGE_READ)
        w2cM = fs.getNode("w2cM").mat()
        fs.release()
        if w2cM is None or w2cM.shape[0] != 32768:
            raise ValueError("Invalid w2cM matrix")
        return w2cM.astype(np.int32)

    def classify_color(self, rgb):
        '''
        Input: 0-255 RGB
        Output: 0-10 color category
        Using PLSA color name
        '''
        # 转换输入为图像格式
        image = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)

        # 分离通道并转换为 float64
        r, g, b = cv2.split(image.astype(np.float64))

        # 量化处理 (floor(channel / 8.0))
        fb = self.floor_mat(b)
        fg = self.floor_mat(g)
        fr = self.floor_mat(r)

        # 计算颜色索引 (范围最多 0~32767)
        index = fr + 32 * fg + 32 * 32 * fb
        index = int(index.flatten()[0])
        # 将颜色索引转换为11类颜色索引（0~10）
        color_category = self.w2cM[index, 0]
        color_name = self.plsa_color_names[color_category]
        return color_name, self.category_map[color_name]

import pandas as pd
class ChipColorClassifier:
    def __init__(self, excel_path='name_table.xlsx'):
        self.df = pd.read_excel(excel_path, index_col='Colorname')
        self.color_name = []
        self.color_value = []
        self.category_map = {
            'Red': 0,
            'Green': 1,
            'Blue': 2,
            'Black': 3,
            'White': 4,
            'Gray': 5,
            'Pink': 6,
            'Orange': 7,
            'Purple': 8,
            'Cyan': 9,
            'Yellow': 10,
            'Brown': 11
        }
        self.category_names = list(self.category_map.keys())
        self._process_data()

    def _process_data(self):
        # 遍历DataFrame中的每一行
        for index, row in self.df.iterrows():
            # 获取颜色分类
            self.color_name.append(row['Classification'])
            # 将RGB字符串转换为数组
            rgb_array = [int(x) for x in row['RGB'].split(',')]
            self.color_value.append(rgb_array)

        self.color_value_array = np.array(self.color_value)
        self.color_value_array_lab = self.sRGB_to_Lab(self.color_value_array / 255.)

    def sRGB_to_Lab(self, rgb1):
        rgb_batch = np.float32(rgb1)
        # 重新调整输入数组的形状，使其成为 (n, 1, 3)，符合OpenCV的要求
        ori_shape = rgb_batch.shape
        rgb_batch = rgb_batch.reshape(-1, 1, 3)
        # 使用OpenCV的cvtColor函数转换RGB到Lab
        lab_batch = cv2.cvtColor(rgb_batch, cv2.COLOR_RGB2Lab)
        return lab_batch.reshape(ori_shape)  # 还原形状

    def classify_color(self, rgb):
        # calculate norm as distance between input color and template colors
        ## use distance in RGB #
        # distances = np.linalg.norm(self.color_value_array - rgb, axis=1)
        ## or use distance in Lab #
        input_lab = self.sRGB_to_Lab(rgb / 255.)
        distances = np.linalg.norm(self.color_value_array_lab - input_lab, axis=1)

        # # or use distance in HSV
        # color_value_array_hsv = colour.RGB_to_HSV(self.color_value_array/255.)
        # input_hsv = colour.RGB_to_HSV(rgb/255.)
        # distances = np.linalg.norm(color_value_array_hsv - input_hsv, axis=1)    
        # check if it is gray
        index = np.argmin(distances)
        if(input_lab[0]>10 and input_lab[0]<90 and abs(input_lab[1])<5 and abs(input_lab[2])<5):
            return 'Gray',5
        # return color_name[index],None
        return self.color_name[index],self.category_map[self.color_names[index]]