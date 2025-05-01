import os
import pickle

import pandas as pd

# ระบุโฟลเดอร์ที่มีไฟล์
folder_path = "/Users/bill/origin-source-code-bill/dynamic/output/seatunnel"

file_significant = "../../output/output/seatunnel_all_status_significant.pkl"
file_each_smell = "../../output/output/seatunnel_rdf_quantile_each_smell.pkl"
file_main_group = "../../output/output/seatunnel_correlation_main_group_4.pkl"

data_each_smell = pd.read_pickle(file_each_smell)
data_mian_group = pd.read_pickle(file_main_group)
data_qr1 = pd.read_pickle(file_significant)

# สร้าง Dictionary เพื่อเก็บข้อมูลแต่ละไฟล์
data_dict = {}

# วนลูปอ่านไฟล์ทั้งหมดในโฟลเดอร์
for filename in os.listdir(folder_path):
    # ตรวจสอบว่าเป็นไฟล์ .pkl
    if filename.endswith(".pkl"):
        file_path = os.path.join(folder_path, filename)

        # โหลดข้อมูลจากไฟล์
        with open(file_path, "rb") as file:
            # เก็บข้อมูลลง dictionary
            data_dict[filename] = pickle.load(file)
        print(f"Loaded: {filename}")


print("\nSummary of Loaded Data:")
for key, value in data_dict.items():
    print(f"{key}: {type(value)}")

