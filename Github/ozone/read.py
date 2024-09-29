import pandas as pd

df_ozone =pd.read_pickle('../output/ozone_pulls.pkl')
df_ozone_filter = pd.read_pickle('../output/ozone_filtered_issues_requests_comments.pkl')
df_ozone_filter2 = pd.read_pickle('../output/ozone_filtered_issues_requests_comments_pulls.pkl')
df_ozone_filter3 = pd.read_pickle('../output/ozone_filtered_final_api.pkl')

pulsar = pd.read_pickle('../output/pulsar_pulls.pkl')
pulsar_filter = pd.read_pickle('../output/pulsar_filtered_issues_requests_comments.pkl')
pulsar_filter2 = pd.read_pickle('../output/pulsar_filtered_issues_requests_comments_pulls.pkl')
pulsar_filter3 = pd.read_pickle('../output/pulsar_filtered_final_api.pkl')

seatunnal = pd.read_pickle('../output/seatunnal_pulls.pkl')
seatunnal_filter = pd.read_pickle('../output/seatunnel_filtered_issues_requests_comments.pkl')
seatunnal_filter2 = pd.read_pickle('../output/seatunnel_filtered_issue_requests_comments_pulls.pkl')
seatunnal_filter3 = pd.read_pickle('../output/seatunnel_filtered_final_api.pkl')