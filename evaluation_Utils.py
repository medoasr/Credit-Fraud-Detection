import seaborn as sns
from numpy import argmax
from sklearn.metrics import classification_report,precision_recall_curve,confusion_matrix,auc
import matplotlib.pyplot as plt

plt.rcParams['figure.max_open_warning'] = 100


def eval_preict_threshold(model,x,threshold=0):
    ypred_prob=model.predict_proba(x)
    y_pred=(ypred_prob[:,1]>=threshold).astype('int')
    return y_pred,ypred_prob[:,1]
def eval_cm(ypred,ytrue,title='',save_png=True):
    lables=['True Negative','False Positive','False Negative','True Positive']
    cm=confusion_matrix(ytrue,ypred,)
    cm_flat=cm.flatten()
    plt.figure(figsize=(8,8))
    sns.heatmap(cm,annot=False,fmt='d',cmap='coolwarm')
    for i,txt in enumerate(cm_flat):
        plt.text(i%2+.5,i//2+.5,f"{lables[i]}\n{txt}",ha='center',va='center',color='black')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_png:
        plt.savefig(f'{title}.png')
    else  :
        plt.show()
    plt.close()
def eval_auc_pr(ypred_prob,ytrue):
    precison,recall,_=precision_recall_curve(y_score=ypred_prob,y_true=ytrue)
    return auc(x=recall,y=precison)

def eval_pr_thresholds(ypred_proba,ytrue,title='',savepng=True):
    precision,recall,thresholds=precision_recall_curve(y_score=ypred_proba,y_true=ytrue)
    plt.figure(figsize=(8,8))
    plt.plot(thresholds,precision[:-1],label='Precision',marker='.')
    plt.plot(thresholds,recall[:-1],label='Recall',marker='.')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("P&R for Different Thresholds")
    plt.legend()
    if savepng:
        plt.savefig(f'{title}.jpg')
    else :
        plt.show()
    plt.close()
def eval_classfication_report(ypred,ytrue,digits=4):
    print("Classfication Report")
    print(classification_report(y_true=ytrue,y_pred=ypred,digits=digits))
    report_stats=classification_report(y_true=ytrue,y_pred=ypred,digits=digits,output_dict=True)
    return  report_stats
def eval_pr_curve(ypred_prob,ytrue,savepng=True,title=''):
    precision,recall,thresholds=precision_recall_curve(y_score=ypred_prob,y_true=ytrue)
    plt.plot(recall,precision,label='Precision_Recall Curve',lw=3)
    plt.xlabel('Recall')
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    if savepng:
        plt.savefig(f'{title}.jpg')
    else :
        plt.show()
    plt.close()


def eval_best_threshold(ypred,ytrue,accordingto='f1_score'):
    precision,recall,thresholds=precision_recall_curve(y_score=ypred,y_true=ytrue)
    f1_scores=((2*precision*recall)/(precision+recall))
    if accordingto=='f1_score':
        optimal_thresh=argmax(f1_scores)
    elif accordingto=='precision':
        optimal_thresh=argmax(precision)
    elif accordingto=='recall':
        optimal_thresh=argmax(recall)
    op_thresh=thresholds[optimal_thresh]
    print("optimal_thresholds:",op_thresh,"f1_Score",f1_scores[optimal_thresh])
    return op_thresh,f1_scores
def update_model_stats(model_comparison, model_name, report_val, metric_config=None):
    if metric_config is None:
        model_comparison[model_name]={
            "F1 Score Positive class": report_val['1']['f1-score'],
            "F1 Score Negative class": report_val['0']['f1-score'],
            "Precision Positive class": report_val['1']['precision'],
            "Recall Positive class": report_val['1']['recall'],
            "F1 Score Average": report_val['macro avg']['f1-score'],

        }
        return  model_comparison




def eval_model(model,model_comparison,path,title,x_train,y_train,x_val,y_val,evaluation_config):
    optimal_threshold=.5
    print(f"eval_fn For: {title}")
    y_train_pred=model.predict(x_train)
    y_train_pred_prob=model.predict_proba(x_train)[:,1]
    eval_classfication_report(y_train_pred,y_train)
    eval_pr_thresholds(y_train_pred_prob,y_train,f'{title} train_pr_thresholds')
    eval_pr_curve(y_train_pred_prob,y_train,title=f'{title} train_pr_curve')
    eval_cm(y_train_pred,y_train,f'{title} tarin_CM')

    print('*'*50)
    y_val_pred=model.predict(x_val)
    y_val_pred_prob=model.predict_proba(x_val)[:,1]
    report_val=eval_classfication_report(y_val_pred,y_val)
    eval_pr_curve(y_val_pred_prob,y_val,title=f'{title} val_pr_curve')
    eval_cm(y_val_pred,y_val,f'{title} val_CM')
    model_comparison = update_model_stats(model_comparison, title, report_val)
    model_comparison[title]['PR AUC'] = eval_auc_pr(y_val_pred_prob, y_val)
    if(evaluation_config['use_optimal_threshold']==True):
        optimal_threshold,f1_scores=eval_best_threshold(y_train_pred_prob,y_train)
        y_val_pred,y_val_pred_prob=eval_preict_threshold(model,x_val,optimal_threshold)
        report_val=eval_classfication_report(y_val_pred,y_val)
        model_comparison=update_model_stats(model_comparison,title+'OptimalThreshold',report_val)
        model_comparison[title+'OptimalThreshold']['PR AUC']=eval_auc_pr(y_val_pred_prob,y_val)
    return  model_comparison,optimal_threshold


