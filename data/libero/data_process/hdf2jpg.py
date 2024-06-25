import h5py
import numpy as np
from PIL import Image
import os

def save_as_npy_or_jpg(obj, parent_path):
    if isinstance(obj, h5py.Dataset):
        # 如果是数据集
        data = obj[:]
        if 'rgb' in obj.name:
            # 如果名称中包含'rgb'，保存为jpg文件
            try:
                for i in range(data.shape[0]):  # 处理每一帧图像
                    image = Image.fromarray(data[i])
                    # 获取相对路径并创建保存路径
                    rel_path = obj.name.lstrip('/').replace('/', '_')
                    save_path = os.path.join(parent_path, rel_path)
                    os.makedirs(save_path, exist_ok=True)
                    filename = os.path.join(save_path, f"{os.path.basename(obj.name)}_{i}.jpg")
                    image.save(filename)
                    print(f"Converted {obj.name} to JPG successfully.")
            except Exception as e:
                print(f"Error converting {obj.name} to JPG: {e}")
        else:
            # 否则保存为npy文件
            filename = os.path.join(parent_path, obj.name.lstrip('/').replace('/', '_') + '.npy')
            os.makedirs(os.path.dirname(filename), exist_ok=True)  # 确保目录存在
            np.save(filename, data)
    elif isinstance(obj, h5py.Group):
        # 如果是组，递归处理组内的每个对象
        for key in obj.keys():
            save_as_npy_or_jpg(obj[key], os.path.join(parent_path, key))

def extract_hdf_data_to_npy_and_jpg(file_path, save_prefix):
    with h5py.File(file_path, 'r') as f:
        # 获取相对路径
        rel_path = os.path.relpath(file_path, start=prefix)
        # 构建保存路径
        save_path = os.path.join(save_prefix, rel_path.split('.')[0].rsplit('_',1)[0])
        os.makedirs(save_path, exist_ok=True)
        
        # 递归处理HDF文件中的每个对象并保存为npy或jpg
        save_as_npy_or_jpg(f, save_path)

# 调用函数来提取HDF文件中的数据

prefix = 'data/libero/dataset/origin/'
task_name = 'libero_goal/'
save_prefix = 'data/libero/dataset/data_jpg/'

files = os.listdir(os.path.join(prefix, task_name))
for f in files:
    if f.endswith('.hdf5'):
        file_path = os.path.join(prefix, task_name, f)
        print(f"HDF5 file name: {f}")
        extract_hdf_data_to_npy_and_jpg(file_path, save_prefix)
