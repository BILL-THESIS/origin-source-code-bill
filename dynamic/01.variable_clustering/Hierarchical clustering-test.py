import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np

# รายชื่อตัวแปร
labels = ['S2209', 'S1181', 'S1659']

# สร้างตารางระยะทางแบบ condensed (เฉพาะค่าบนขวา)
# ลำดับ: S2209-S1181, S2209-S1659, S1181-S1659
distance_matrix = [0.2546, 0.0871, 0.40]

# ทำ clustering
linkage_matrix = sch.linkage(distance_matrix, method='average')

# วาด dendrogram
plt.figure(figsize=(6, 4))
plt.xticks(rotation=90)
sch.dendrogram(linkage_matrix, labels=labels)
plt.title(f'Hierarchical Clustering Dendrogram of \n'
          f'Spearman Correlation by Type of Code Smell')
plt.ylabel('Distance (1 - Spearman)')
plt.tight_layout()
plt.savefig("test_dendrogram.png")
plt.show()



