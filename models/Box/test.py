import pandas as pd

import pyarrow.parquet as pq

part = "/Users/bill/origin-source-code-bill/models/KMeans/combia2/['end_OOA', 'OOA_change', 'D_percent', 'B_percent', 'C_percent', 'OOA_percent'].parquet"

# Read a Parquet file
table = pq.read_table(part)
df = table.to_pandas()

print(df.to_markdown)