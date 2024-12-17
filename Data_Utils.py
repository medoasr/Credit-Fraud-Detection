import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler

def load_data(config):
    train_data=pd.read_csv(config['dataset']['train']['path'])
    val_data=pd.read_csv(config['dataset']['val']['path'])
    x_train=train_data.drop(config['dataset']['target'],axis=1)
    y_train=train_data[config['dataset']['target']]
    x_val=val_data.drop(config['dataset']['target'],axis=1)
    y_val=val_data[config['dataset']['target']]
    return x_train,y_train,x_val,y_val

def sampling_data(x_train,y_tarin,balance_type='SMOTE',sampling_strategy='auto',random_state=None):
    print("Dataset Before Sampling ")
    print(f"number of Frauds: {len(y_tarin[y_tarin==1])}")
    print(f"number of NON Frauds: {len(y_tarin[y_tarin==0])}")

    if balance_type=='SMOTE':
        sampler=SMOTE(random_state=random_state,sampling_strategy=sampling_strategy)
        x_sampled,y_sampled=sampler.fit_resample(x_train,y_tarin)
        print("Dataset After SMOTE ")
        print(f"number of Frauds: {len(y_sampled[y_sampled==1])}")
        print(f"number of NON Frauds: {len(y_sampled[y_sampled==0])}")
    if balance_type=='under':
        sampler=RandomUnderSampler(random_state=random_state,sampling_strategy=sampling_strategy)
        x_sampled,y_sampled=sampler.fit_resample(x_sampled,y_sampled)
        print("Dataset After UnderSampling ")
        print(f"number of Frauds: {len(y_sampled[y_sampled==1])}")
        print(f"number of NON Frauds: {len(y_sampled[y_sampled==0])}")
    return x_sampled,y_sampled

def scaling(x_train,x_val,type_of_scale='robust'):
    if type_of_scale=='robust':
        scaler=RobustScaler()
    if type_of_scale=='standard':
        scaler=StandardScaler()
    if type_of_scale=='minmax':
        scaler=MinMaxScaler()
    x_train_scaled=scaler.fit_transform(x_train)
    if(x_val is not None):
        x_val_scaled=scaler.transform(x_val)
        return x_train_scaled,x_val_scaled

    return x_train_scaled

