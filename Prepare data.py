import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import IO
import seaborn as sns

df = pd.read_csv('./Data/kddcup99/kddcup.data_10_percent/kddcup.data_10_percent')

X = df.iloc[:,:41]
Y = df.iloc[:,41]
X.columns=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]

prt_map ={}
srv_map = {}
flag_map = {}
value = 0
for i in X['protocol_type'].unique():
    prt_map.update({i:value})
    value+=1

value = 0
for i in X['service'].unique():
    srv_map.update({i:value})
    value+=1
    
value = 0
for i in X['flag'].unique():
    flag_map.update({i:value})
    value+=1

X['protocol_type'] = X['protocol_type'].map(prt_map)
X['service'] = X['service'].map(srv_map)
X['flag'] = X['flag'].map(flag_map)

Y = Y.str.replace('.','')
attacks_types = {
'normal': 'normal',
'back': 'dos',
'buffer_overflow': 'u2r',
'ftp_write': 'r2l',
'guess_passwd': 'r2l',
'imap': 'r2l',
'ipsweep': 'probe',
'land': 'dos',
'loadmodule': 'u2r',
'multihop': 'r2l',
'neptune': 'dos',
'nmap': 'probe',
'perl': 'u2r',
'phf': 'r2l',
'pod': 'dos',
'portsweep': 'probe',
'rootkit': 'u2r',
'satan': 'probe',
'smurf': 'dos',
'spy': 'r2l',
'teardrop': 'dos',
'warezclient': 'r2l',
'warezmaster': 'r2l',
}
Y = Y.map(attacks_types)

map_y_label={}
value=0
for i in Y.unique():
    map_y_label.update({i:value})
    value+=1
Y = Y.map(map_y_label)

X = X[[col for col in X if X[col].nunique() > 1]]# keep columns where there are more than 1 unique values

corr = X.corr()

plt.figure(figsize=(15,12))
sns.heatmap(corr)
plt.show()

corr_dict = (corr.rename_axis('row_id')
         .reset_index()
         .melt(id_vars='row_id', var_name='column_id')
         .to_dict('records'))
highly_corr = []
for d in corr_dict:
    if d['value']>0.97 and d['row_id']!=d['column_id']:
        highly_corr.append([d['column_id'],d['row_id']])

#This variable is highly correlated with num_compromised and should be ignored for analysis.
#(Correlation = 0.9938277978738366)
X.drop('num_root',axis = 1,inplace = True)

#This variable is highly correlated with serror_rate and should be ignored for analysis.
#(Correlation = 0.9983615072725952)
X.drop('srv_serror_rate',axis = 1,inplace = True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9947309539817937)
X.drop('srv_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_serror_rate and should be ignored for analysis.
#(Correlation = 0.9993041091850098)
X.drop('dst_host_srv_serror_rate',axis = 1, inplace=True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9869947924956001)
X.drop('dst_host_serror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
#(Correlation = 0.9821663427308375)
X.drop('dst_host_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9851995540751249)
X.drop('dst_host_srv_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with dst_host_srv_count and should be ignored for analysis.
#(Correlation = 0.9736854572953938)
X.drop('dst_host_same_srv_rate',axis = 1, inplace=True)

X = X.values
Y = Y.values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=41)

IO.save(X_train,'X_train')
IO.save(X_test,'X_test')
IO.save(Y_train,'Y_train')
IO.save(Y_test,'Y_test')