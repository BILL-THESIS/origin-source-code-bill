import pandas as pd
import ast

# สร้าง DataFrame จากข้อมูลตัวอย่าง
data = {
    "#": [1, 2, 3, 4, 5, 6],
    "ฟีเจอร์ของลำดับ": [
        "[['java:S1134_created', 'java:S5164_created']]",
        "[['java:S2629_created', 'java:S1134_created']]",
        "[['java:S2222_created', 'java:S5164_created'], ['java:S5164_created', 'java:S899_created']]",
        "[['java:S1134_created', 'java:S2142_created']]",
        "[['java:S2629_created', 'java:S899_created'], ['java:S2629_created', 'java:S2222_created']]",
        "[['java:S2142_created', 'java:S2222_created'], ['java:S2142_created', 'java:S899_created']]"
    ]
}

df = pd.DataFrame(data)

# แปลงสตริงให้เป็นลิสต์ และหาข้อมูลไม่ซ้ำกัน
def extract_unique_features(feature_str):
    feature_list = ast.literal_eval(feature_str)
    flat_list = [item for sublist in feature_list for item in sublist]
    return list(set(flat_list))

# สร้างคอลัมน์ใหม่ 'new'
df['new'] = df['ฟีเจอร์ของลำดับ'].apply(extract_unique_features)

print(df[['#', 'new']])
