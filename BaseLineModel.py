import pandas as pd
import numpy as np
import argparse
import  joblib


from seaborn.algorithms import bootstrap
from sklearn.metrics import f1_score ,classification_report,confusion_matrix,make_scorer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neural_network import  MLPClassifier

import datetime
import os 
import json
from helper_utils import *
from Data_Utils import *
from evaluation_Utils import *
import datetime
from evaluation_Utils import  *
def logistic_Reg(x_train,y_train,x_val,y_val,config,RANDOM_SEED,model_comparison):
    best_param={}
    if config['train']['logsitcregression']['use_grid_search']==True:
        param_grid={'C':[.0001,.001,.01,.1,1.0,10],
                    'class_weight':['balanced',None,{0:.25,1:.75},{0:.15,1:.85},{0:.3,1:.7}],
                    'penalty':['l2'],
                    'solver':['sag','lbfgs','newton-cg','saga'],
                    'max_iter':[100,200,300,400,500]
                    }
        lr=LogisticRegression()
        scorer=make_scorer(f1_score,pos_label=1)
        str_fold=StratifiedKFold(5,shuffle=True,random_state=RANDOM_SEED)
        grid_s=GridSearchCV(lr,param_grid,cv=str_fold,scoring=scorer,n_jobs=-1)
        grid_s.fit(x_train,y_train)
        best_param=grid_s.best_params_
        print(best_param)
    else:
        best_param=train_config['train']['logsitcregression']['param']


    lr=LogisticRegression(**best_param,random_state=RANDOM_SEED)
    
    lr.fit(x_train,y_train)
    model_comparison,threshold=eval_model(lr,model_comparison,'','logistic_regression',x_train,y_train,x_val,y_val,train_config['eval'])
    return {'model':lr,'parameters':best_param,'threshold':threshold}


def random_forest(x_train, y_train, x_val, y_val, config, RANDOM_SEED, model_comparison):
    best_param = {}
    if config['train']['random_forest']['randomized_search'] == True:
        param_grid = {
            'n_estimators': [200, 400, 600 ,800],
            'min_samples_leaf': [2, 5, 10, 15],
            'min_samples_split': [5, 10, 20],
            'class_weight': [{0: 0.20, 1: 0.80}, 'balanced_subsample', {0: 0.15, 1: 0.85}],
        }

        rf = RandomForestClassifier()
        scorer = make_scorer(f1_score, pos_label=1)
        str_fold = StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)
        random_s = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=-1,bootstrap=True,random_state=RANDOM_SEED),
                                      param_distributions=param_grid,
                                      scoring=scorer,
                                      cv=str_fold,
                                      n_iter=20,
                                      n_jobs=-1,
                                      verbose=2,random_state=RANDOM_SEED)
        random_s.fit(x_train, y_train)
        best_param = random_s.best_params_
        print(best_param)
    else:
        best_param = train_config['train']['random_forest']['param']

    rf = RandomForestClassifier(**best_param, random_state=RANDOM_SEED)

    rf.fit(x_train, y_train)
    model_comparison, threshold = eval_model(rf, model_comparison, '', 'Random_Forest', x_train, y_train, x_val,
                                             y_val, train_config['eval'])
    return {'model': rf, 'parameters': best_param, 'threshold': threshold}


def neural_network(x_train, y_train, x_val, y_val, config, RANDOM_SEED, model_comparison):
    best_param = {}
    if config['train']['neural_network']['randomized_search'] == True:
        param_grid = {
            'activation': ['relu', 'tanh'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [.0001, .001, .01],
            'learning_rate': ['adaptive', 'constant'],
            'hidden_layer_sizes':[(64,32),(128,32),(64,32,16),(30,20,10),(128,64,32),(265 ,128,32)]

        }

        Mlp = MLPClassifier()
        scorer = make_scorer(f1_score, pos_label=1)
        str_fold = StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED)
        random_s = RandomizedSearchCV(estimator=Mlp,
                                      param_distributions=param_grid,
                                      scoring=scorer,
                                      cv=str_fold,
                                      n_iter=20,
                                      n_jobs=-1,
                                      verbose=2,random_state=RANDOM_SEED)
        random_s.fit(x_train, y_train)
        best_param = random_s.best_params_
        print(best_param)
    else:
        best_param = train_config['train']['neural_network']['param']

    Mlp = MLPClassifier(**best_param,random_state=RANDOM_SEED)

    Mlp.fit(x_train, y_train)
    model_comparison, threshold = eval_model(Mlp, model_comparison, '', 'neural_network', x_train, y_train, x_val,
                                             y_val, train_config['eval'])
    return {'model': Mlp, 'parameters': best_param, 'threshold': threshold}
def voting_classfier(x_train, y_train, x_val, y_val, config, RANDOM_SEED, model_comparison,models):
    lr_model=models['logsitcregression']['model']
    rf=models['random_forest']['model']
    ml=models['neural_network']['model']

    vc=VotingClassifier(estimators=[('lr',lr_model),("rf",rf),('ml',ml)],voting='soft')
    best_param = config['train']['voting_classifier']['param']



    vc.fit(x_train, y_train)
    model_comparison, threshold = eval_model(vc, model_comparison, '', 'voting_classifier', x_train, y_train, x_val,
                                             y_val, config['eval'])
    return {'model': vc, 'param':best_param, 'threshold': threshold}

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config",  help="path to the dataset and preprocessing config file", default="config.yml")
    parser.add_argument("--train_config",  help="path to the train param config file", default="config.yml")

    args = parser.parse_args()
    data_config=load_Config('data_config.yml')
    train_config=load_Config('train_config.yml')
    RANDOM_SEED=data_config['random_seed']
    np.random.seed(RANDOM_SEED)
    x_train,y_train,x_val,y_val=load_data(data_config)
    if data_config['sampling']['sample']:
           x_train,y_train=sampling_data(x_train,y_train,data_config['sampling']['method'],data_config['sampling']['sample_strategy'],random_state=RANDOM_SEED)
    x_train,x_val=scaling(x_train,x_val,data_config['preprocessing']['scaler_type'])

    models={}
    model_comparison={}

    if train_config['train']['logsitcregression']['train']:
         models['logsitcregression']=logistic_Reg(x_train,y_train,x_val,y_val,train_config,RANDOM_SEED,model_comparison)

    if train_config['train']['random_forest']['train']:
        models['random_forest'] = random_forest(x_train, y_train, x_val, y_val, train_config, RANDOM_SEED,
                                                   model_comparison)
    if train_config['train']['neural_network']['train']:
        models['neural_network'] = neural_network(x_train, y_train, x_val, y_val, train_config, RANDOM_SEED,
                                                   model_comparison)

    if train_config['train']['voting_classifier']['train']:
            models['voting_classifier'] = voting_classfier(x_train, y_train, x_val, y_val, train_config, RANDOM_SEED,
                                                      model_comparison,models)
    if model_comparison:
        plot_models_comparsion(model_comparison)
    print(pd.DataFrame(model_comparison))
    print(pd.DataFrame(model_comparison).T)
    print(pd.DataFrame(model_comparison).T.to_markdown())
### save models
    joblib.dump(models,'Trained_Models.pkl')
