import pandas as pd
import os
from itertools import chain, combinations, permutations
from itertools import product

df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
d1 = df_original.iloc[:, [5,9,10,11,12,13]]
labels = pd.read_pickle('lable/labels_final.pkl')
group_lables = pd.read_pickle('lable/labels_group.pkl')
directory_path = 'D:\origin-source-code-bill\models\KMeans\combia'

df1_col = d1.columns
df1_col = df1_col.tolist()
group_lables_col = group_lables.columns
group_lables_col = group_lables_col.tolist()

# initialize list and tuple
test_list = df1_col
test_tup = group_lables_col

# printing original list and tuple
print("The original list : " + str(test_list))
print("The original tuple : " + str(test_tup))

# Construct Cartesian Product Tuple list
# using itertools.product()
res = list(product( test_list, test_tup))

# printing result
print("The Cartesian Product is : " + str(res))

filter_res = []
for x in res:
    # print(x[0], ' vs ', x[1])
    if x[0] in x[1]:
        filter_res.append(x)
print('filter res is : ' + str(filter_res))

# result_list = []
# for x in substrings:
#     # append True/False for substring x
#     result_list.append(x.lower() in string.lower())
#
# # call any() with boolean results list
# print(any(result_list))


# abc = '["changed_files", "begin_Dispensables", "begin_Bloaters"]_2'
# abc_dat = re.split("_\d", abc)[0]
# df_original[eval(abc_dat)]