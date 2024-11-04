import pandas as pd

data_category_smell = pd.read_pickle('../Sonar/output/05.sonar_group_rules_category_smells_pull_bug.pkl')
seatunnal_bug = pd.read_pickle('seatunnal_bug_comapare_time.pkl')
seatunnal_bug['total_time'] = pd.to_datetime(seatunnal_bug['merged_at']) - pd.to_datetime(seatunnal_bug['created_at'])
# seatunnal_bug['bug_diff'] = seatunnal_bug['value_ended'] - seatunnal_bug['value_created']

smell_created = pd.merge(seatunnal_bug, data_category_smell, left_on=['key_created', 'revision_created'], right_on=['key','revision'], how='inner')
smell_ended = pd.merge(seatunnal_bug, data_category_smell, left_on=['key_ended', 'revision_ended'], right_on=['key','revision'], how='inner')