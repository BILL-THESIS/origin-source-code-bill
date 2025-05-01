import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# โหลดไฟล์
file_path = '../../output/seatunnel_all_status_significant.pkl'
df = pd.read_pickle(file_path)

# สร้าง scatter plot
plt.figure(figsize=(12, 10))
scatter = sns.scatterplot(
    data=df,
    x='smell_sum_q1',
    y='smell_sum_q3',
    hue='eff_size',        # สีตาม effect size
    size='d_value',         # ขนาดตาม d_value
    sizes=(50, 300),        # กำหนดช่วงขนาดจุด
    palette='viridis',
    alpha=0.8,
    edgecolor='black'
)

# วาดเส้น y = x (ไม่เปลี่ยนแปลง)
max_val = max(df['smell_sum_q1'].max(), df['smell_sum_q3'].max())
plt.plot([0, max_val], [0, max_val], ls="--", c="red", label="y=x (no change)")

# # ใส่ label ชื่อ metric เฉพาะที่ eff_size เป็น large
# for _, row in df[df['eff_size'] == 'large'].iterrows():
#     plt.text(
#         row['smell_sum_q1'] + 2,   # ขยับ x ไปขวานิดนึง
#         row['smell_sum_q3'] + 2,   # ขยับ y ขึ้นนิดนึง
#         row['metric'],
#         fontsize=8,
#         color='black'
#     )

# ตั้งชื่อแกน และหัวข้อ
plt.xlabel('จำนวนโคดสเมล Q1')
plt.ylabel('จำนวนโคดสเมลQ3')
plt.title('เปรียบเทียบความแตกต่างโคดสเมล (Q1 vs Q3) กับขนาดผลกระทบของ คลิฟฟ์ส เดลต้า')

# จัด legend ไปไว้ข้างนอกเพื่อไม่บังข้อมูล
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Effect Size')

plt.grid(True)
plt.tight_layout()
plt.show()



# สร้าง FacetGrid โดยแยกแต่ละกราฟตาม effect size
g = sns.relplot(
    data=df,
    x='smell_sum_q1',
    y='smell_sum_q3',
    col='eff_size',          # แยกกราฟตามกลุ่ม effect size
    hue='eff_size',          # ยังคงสีตาม effect size (optional)
    size='d_value',
    sizes=(50, 300),
    kind='scatter',
    palette='viridis',
    height=5,
    aspect=1
)

# ใส่เส้น y = x ในแต่ละกราฟ
for ax in g.axes.flat:
    limits = [0, max(df['smell_sum_q1'].max(), df['smell_sum_q3'].max())]
    ax.plot(limits, limits, ls='--', c='red')

g.set_titles(col_template="{col_name} effect")
g.set_axis_labels("Smell Sum Q1", "Smell Sum Q3")
plt.tight_layout()
plt.show()

