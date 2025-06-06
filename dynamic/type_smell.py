import pandas as pd
import ast

s_rf = {

}

df_s_rf = pd.DataFrame(s_rf)
df_s_lgbm = pd.DataFrame(s_lgbm)
df_p_rf = pd.DataFrame(p_rf)
df_p_lgbm = pd.DataFrame(p_lgbm)
df_o_rf = pd.DataFrame(o_rf)
df_o_lgbm = pd.DataFrame(o_lgbm)


# แปลงสตริงให้เป็นลิสต์ และหาข้อมูลไม่ซ้ำกัน
def extract_unique_features(feature_str):
    feature_list = ast.literal_eval(feature_str)
    flat_list = [item for sublist in feature_list for item in sublist]
    return list(set(flat_list))

# สร้างคอลัมน์ใหม่ 'new'
df_s_rf['new'] = df_s_rf['ฟีเจอร์ของลำดับ'].apply(extract_unique_features)


