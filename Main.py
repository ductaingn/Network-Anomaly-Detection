import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('./Data/kddcup99/corrected/corrected')

df_x = df.iloc[:,:41]
df_y = df.iloc[:,41]
df_x.columns=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]

prt_map ={}
srv_map = {}
flag_map = {}
value = 0
for i in df_x['protocol_type'].unique():
    prt_map.update({i:value})
    value+=1
value = 0
for i in df_x['service'].unique():
    srv_map.update({i:value})
    value+=1

for i in df_x['flag'].unique():
    flag_map.update({i:value})
    value+=1

print(prt_map,srv_map,flag_map)

df_x['protocol_type'] = df_x['protocol_type'].map(prt_map)
df_x['service'] = df_x['service'].map(srv_map)
df_x['flag'] = df_x['flag'].map(flag_map)

df_y = df_y.str.replace('.','')

map_y_label={}
value=0
for i in df_y.unique():
    map_y_label.update({i:value})
    value+=1
df_y = df_y.map(map_y_label)

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)

y_predict = rf.predict(x_test)
score = rf.score(x_test,y_test)
print(score)
