import pandas as pd


def select_cols(data):
    prefix = "java:"
    suffix = "_created"
    selected_cols = [col for col in data.columns if col.startswith(prefix) and col.endswith(suffix)]
    selected_cols = data[selected_cols].fillna(0)
    return selected_cols


ozone = pd.read_pickle("output/ozone_cut_time.pkl")
se = pd.read_pickle("output/seatunnal_cut_time.pkl")
pulsar = pd.read_pickle("output/pulsar_cut_time.pkl")

ozone_col = select_cols(ozone)
pulsar_col = select_cols(pulsar)
se_col = select_cols(se)

data_output = se[['url','base.sha','created_at','merged_at','total_time']]
new_df = pd.concat([se_col.iloc[:,:5], data_output], axis=1).head(5)

o = ozone_col.loc[:, (ozone_col != 0).any(axis=0)]
p = ozone_col.loc[:, (pulsar_col != 0).any(axis=0)]
s = ozone_col.loc[:, (se_col != 0).any(axis=0)]

o_list = list(o.columns)
p_list = list(p.columns)
s_list = list(s.columns)

# Count the amount code smell of each project

ozone_count = o.sum()
o_de = ozone_count.describe()
o_de_df = pd.DataFrame(o_de).T

plsar_count = p.sum()
p_de = plsar_count.describe()
p_de_df = pd.DataFrame(p_de).T

se_count = s.sum()
se_de = se_count.describe()
se_de_df = pd.DataFrame(se_de).T

df_smell = pd.concat([p_de_df, se_de_df, o_de_df], axis=0).round(4)



# Rechack Time description
ozone_time = ozone['total_time'].describe()
ozone_time = pd.DataFrame(ozone_time).T
se_time = se['total_time'].describe()
se_time = pd.DataFrame(se_time).T
pulsar_time = pulsar['total_time'].describe()
pulsar_time = pd.DataFrame(pulsar_time).T

df_time = pd.concat([ozone_time, se_time, pulsar_time], axis=1).round(4)