# coding: utf-8

import numpy as np 
import pandas as pd
import pandas as pd 
import numpy as np
# import gc
import json
import os
import time
# import pickle
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Classifier Libraries

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import plot_importance
from imblearn.over_sampling import RandomOverSampler
# from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import ADASYN


# 训练类别编码（one_hot/label_encoder)
def train_label_encoder(df_tag):
    df_tag['edu_deg_cd'] = df_tag['edu_deg_cd'].astype(str)
    df_tag['deg_cd'] = df_tag['deg_cd'].astype(str)
    df_tag['atdd_type'] = df_tag['atdd_type'].astype(str)
    df_tag['acdm_deg_cd'] = df_tag['acdm_deg_cd'].astype(str)

    le_gdr_cd = LabelEncoder()
    le_mrg_situ_cd = LabelEncoder()
    le_edu_deg_cd = LabelEncoder()
    le_acdm_deg_cd = LabelEncoder()
    le_deg_cd = LabelEncoder()
    le_atdd_type = LabelEncoder()

    df_tag['gdr_encoded'] = le_gdr_cd.fit_transform(df_tag.gdr_cd)
    df_tag['mrg_encoded'] = le_mrg_situ_cd.fit_transform(df_tag.mrg_situ_cd)
    df_tag['edu_encoded'] = le_edu_deg_cd.fit_transform(df_tag.edu_deg_cd)
    df_tag['acdm_encoded'] = le_acdm_deg_cd.fit_transform(df_tag.acdm_deg_cd)
    df_tag['deg_encoded'] = le_deg_cd.fit_transform(df_tag.deg_cd)
    df_tag['atdd_encoded'] = le_atdd_type.fit_transform(df_tag.atdd_type)
    # print(le_gdr_cd.classes_,le_mrg_situ_cd.classes_,le_edu_deg_cd.classes_,le_acdm_deg_cd.classes_,le_deg_cd.classes_,le_atdd_type.classes_)

    joblib.dump(le_gdr_cd, os.path.join('model/label/', "le_gdr_encoder.pkl"))
    joblib.dump(le_mrg_situ_cd, os.path.join('model/label/', "le_mrg_encoder.pkl"))
    joblib.dump(le_edu_deg_cd, os.path.join('model/label/', "le_edu_encoder.pkl"))
    joblib.dump(le_acdm_deg_cd, os.path.join('model/label/', "le_acdm_encoder.pkl"))
    joblib.dump(le_deg_cd, os.path.join('model/label/', "le_deg_encoder.pkl"))
    joblib.dump(le_atdd_type, os.path.join('model/label/', "le_atdd_encoder.pkl"))

    gdr_ohe = OneHotEncoder()
    mrg_ohe = OneHotEncoder()
    edu_ohe = OneHotEncoder()
    acdm_ohe = OneHotEncoder()
    deg_ohe = OneHotEncoder()
    atdd_ohe = OneHotEncoder()

    X = gdr_ohe.fit_transform(df_tag.gdr_encoded.values.reshape(-1,1)).toarray()
    Xm = mrg_ohe.fit_transform(df_tag.mrg_encoded.values.reshape(-1,1)).toarray()
    Xe = edu_ohe.fit_transform(df_tag.edu_encoded.values.reshape(-1,1)).toarray()
    Xac = acdm_ohe.fit_transform(df_tag.acdm_encoded.values.reshape(-1,1)).toarray()
    Xd = deg_ohe.fit_transform(df_tag.deg_encoded.values.reshape(-1,1)).toarray()
    Xat = atdd_ohe.fit_transform(df_tag.atdd_encoded.values.reshape(-1,1)).toarray()

    joblib.dump(gdr_ohe, os.path.join('model/label/', "ohe_gdr_encoder.pkl"))
    joblib.dump(mrg_ohe, os.path.join('model/label/', "ohe_mrg_encoder.pkl"))
    joblib.dump(edu_ohe, os.path.join('model/label/', "ohe_edu_encoder.pkl"))
    joblib.dump(acdm_ohe, os.path.join('model/label/', "ohe_acdm_encoder.pkl"))
    joblib.dump(deg_ohe, os.path.join('model/label/', "ohe_deg_encoder.pkl"))
    joblib.dump(atdd_ohe, os.path.join('model/label/', "ohe_atdd_encoder.pkl"))

    df_tag.drop(['gdr_encoded','mrg_encoded','edu_encoded','acdm_encoded','deg_encoded'
        ,'atdd_encoded'],axis=1,inplace=True)

    return

#处理类别标签
def process_category(df_tag):
    df_tag['edu_deg_cd'] = df_tag['edu_deg_cd'].astype(str)
    df_tag['deg_cd'] = df_tag['deg_cd'].astype(str)
    df_tag['atdd_type'] = df_tag['atdd_type'].astype(str)
    df_tag['acdm_deg_cd'] = df_tag['acdm_deg_cd'].astype(str)

    le_gdr_cd = joblib.load('model/label/le_gdr_encoder.pkl')
    le_mrg_situ_cd = joblib.load('model/label/le_mrg_encoder.pkl')
    le_edu_deg_cd = joblib.load('model/label/le_edu_encoder.pkl')
    le_acdm_deg_cd = joblib.load('model/label/le_acdm_encoder.pkl')
    le_deg_cd = joblib.load('model/label/le_deg_encoder.pkl')
    le_atdd_type = joblib.load('model/label/le_atdd_encoder.pkl')

    df_tag['gdr_encoded'] = le_gdr_cd.transform(df_tag.gdr_cd)
    df_tag['mrg_encoded'] = le_mrg_situ_cd.transform(df_tag.mrg_situ_cd)
    df_tag['edu_encoded'] = le_edu_deg_cd.transform(df_tag.edu_deg_cd)
    df_tag['acdm_encoded'] = le_acdm_deg_cd.transform(df_tag.acdm_deg_cd)
    df_tag['deg_encoded'] = le_deg_cd.transform(df_tag.deg_cd)
    df_tag['atdd_encoded'] = le_atdd_type.transform(df_tag.atdd_type)

    gdr_ohe = joblib.load('model/label/ohe_gdr_encoder.pkl')
    mrg_ohe = joblib.load('model/label/ohe_mrg_encoder.pkl')
    edu_ohe = joblib.load('model/label/ohe_edu_encoder.pkl')
    acdm_ohe = joblib.load('model/label/ohe_acdm_encoder.pkl')
    deg_ohe = joblib.load('model/label/ohe_deg_encoder.pkl')
    atdd_ohe = joblib.load('model/label/ohe_atdd_encoder.pkl')

    X = gdr_ohe.transform(df_tag.gdr_encoded.values.reshape(-1,1)).toarray()
    Xm = mrg_ohe.transform(df_tag.mrg_encoded.values.reshape(-1,1)).toarray()
    Xe = edu_ohe.transform(df_tag.edu_encoded.values.reshape(-1,1)).toarray()
    Xac = acdm_ohe.transform(df_tag.acdm_encoded.values.reshape(-1,1)).toarray()
    Xd = deg_ohe.transform(df_tag.deg_encoded.values.reshape(-1,1)).toarray()
    Xat = atdd_ohe.transform(df_tag.atdd_encoded.values.reshape(-1,1)).toarray()

    dfOhe = pd.DataFrame(X,columns=['gdr_'+str(int(i)) for i in range(X.shape[1])])
    df_tag = pd.concat([df_tag,dfOhe],axis=1)

    dfOhe = pd.DataFrame(Xm,columns=['mrg_'+str(int(i)) for i in range(Xm.shape[1])])
    df_tag = pd.concat([df_tag,dfOhe],axis=1)

    dfOhe = pd.DataFrame(Xe,columns=['edu_'+str(int(i)) for i in range(Xe.shape[1])])
    df_tag = pd.concat([df_tag,dfOhe],axis=1)

    dfOhe = pd.DataFrame(Xac,columns=['acdm_'+str(int(i)) for i in range(Xac.shape[1])])
    df_tag = pd.concat([df_tag,dfOhe],axis=1)

    dfOhe = pd.DataFrame(Xd,columns=['deg_'+str(int(i)) for i in range(Xd.shape[1])])
    df_tag = pd.concat([df_tag,dfOhe],axis=1)

    dfOhe = pd.DataFrame(Xat,columns=['atdd_'+str(int(i)) for i in range(Xat.shape[1])])
    df_tag = pd.concat([df_tag,dfOhe],axis=1)

    '''
    #构造二阶特征
    col_names = ['gdr_'+str(int(i)) for i in range(X.shape[1])] 
    col_names += ['mrg_'+str(int(i)) for i in range(Xm.shape[1])]
    col_names += ['edu_'+str(int(i)) for i in range(Xe.shape[1])]
    col_names += ['atdd_'+str(int(i)) for i in range(Xat.shape[1])]
    col_names += ['ic_ind','fr_or_sh_ind','dnl_mbl_bnk_ind','dnl_bind_cmb_lif_ind','hav_car_grp_ind',
    'hav_hou_grp_ind','l6mon_agn_ind','vld_rsk_ases_ind','loan_act_ind']

    non_binary_cols = ['confirm_rsk_ases_lvl_typ_cd','pl_crd_lmt_cd','bk1_cur_year_mon_avg_agn_amt_cd',
    'tot_ast_lvl_cd','pot_ast_lvl_cd','pl_crd_lmt_cd','perm_crd_lmt_cd','l1y_crd_card_csm_amt_dlm_cd',
    'l6mon_daim_aum_cd','ovd_30d_loan_tot_cnt']

    # print('二阶特征个数：',len(col_names)**2)

    count=0
    for n in range(len(col_names)):
        for m in range(n+1,len(col_names)):
            # df_tag[col_names[n]+'_'+col_names[m]] = df_tag.apply(lambda x: x[col_names[n]]*x[col_names[m]], axis=1)
            count+=1

    li=[]
    for n in col_names:
        for m in non_binary_cols:
            # df_tag[n+'_'+m] = df_tag.apply(lambda x: x[n]*x[m], axis=1)
            li.append(n+'_'+m)
            count+=1
    print('二阶特征个数：',count)
    print(li)

    X = df_tag.drop(['id','gdr_cd', 'mrg_situ_cd', 'edu_deg_cd', 'acdm_deg_cd', 'deg_cd', 'atdd_type',
        'gdr_encoded','mrg_encoded','edu_encoded','acdm_encoded','deg_encoded'
        ,'atdd_encoded','app_fraud','trade_fraud'],axis=1)

    # X.to_csv('df_tag_add_feature.csv',header=True,index=False)
    '''
    # print(X_nan.columns)
    # X = SimpleImputer(missing_values=np.nan , strategy='mean').fit_transform(X_nan)
    X = df_tag.drop(['id','gdr_cd', 'mrg_situ_cd', 'edu_deg_cd', 'acdm_deg_cd', 'deg_cd', 'atdd_type',
        'gdr_encoded','mrg_encoded','edu_encoded','acdm_encoded','deg_encoded'
        ,'atdd_encoded','trade_fraud','app_fraud'],axis=1)
    # y = df_tag['flag']
    # X = pd.DataFrame(X,columns=X_nan.columns)

    return X

# 划分训练(含验证)和测试数据集
def split_train_valid(Xall):
    Xall = Xall.drop(['flag'],axis=1)
    y = ['flag']
    X = SimpleImputer(missing_values=np.nan , strategy='most_frequent').fit_transform(Xall)
    X = pd.DataFrame(X,columns=Xall.columns)
    
    # X_sec = pd.read_csv('df_tag_sub_feature.csv')
    # tmp_numpy,n = reduce_dim(X_sec)
    # X_sec = pd.DataFrame(tmp_numpy,columns=['f'+str(i) for i in range(n)]) 
    # X = pd.concat([X,X_sec],axis=1)

    print(X.shape,y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=TEST_SIZE, 
                                        random_state=2018, stratify=y)

    # X_train = reduce_dim(X_train)
    # X_test = reduce_dim(X_test)
    print('X_train, X_test, Y_train, Y_test:',X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # X_train = X_train.values
    # X_test = X_test.values
    # Y_train = Y_train.values
    # Y_test = Y_test.values

    # # See if both the train and test label distribution are similarly distributed
    # train_unique_label, train_counts_label = np.unique(Y_train, return_counts=True)
    # test_unique_label, test_counts_label = np.unique(Y_test, return_counts=True)
    # print('-' * 100)

    # print('Label Distributions: \n')
    # print(train_counts_label/ len(Y_train))
    # print(test_counts_label/ len(Y_test))
    # print(X_train.shape,X_test.shape)
    return X_train, X_test, Y_train, Y_test

def reduce_dim(X):
    n_components = 2 #7
    
    X = SimpleImputer(missing_values=np.nan , strategy='most_frequent').fit_transform(X)
    # X_embedded = TSNE(n_components=n_components).fit_transform(X)
    pca = PCA(n_components=n_components)
    X_embedded = pca.fit_transform(X)
    print('降维以后：',X_embedded.shape)
    # print(pca.explained_variance_ratio_)
    return (X_embedded,n_components)


# 训练集和测试集的划分
def train(clf,X_train,Y_train):
    params = {
            # "penalty": ['l1', 'l2'], 
            'C': [0.01, 0.1, 1]
    }

    kFold = StratifiedKFold(n_splits=NUMBER_KFOLDS, random_state=0, shuffle=False)

    grid_xgb = GridSearchCV(clf, param_grid=params,scoring='roc_auc',cv=kFold,n_jobs=4)

    # ros = RandomOverSampler(random_state=0,sampling_strategy=0.5)
    # print(len(X_train))
    # X_train, Y_train = ros.fit_resample(X_train, Y_train)

    grid_result=grid_xgb.fit(X_train, Y_train,)

    print("Best: %f using %s" % (grid_result.best_score_, grid_xgb.best_params_))

    #查看所有参数组合
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean,param in zip(means,params):
        print("%f  with:   %r" % (mean,param))  

    #保存参数
    # with open('model/best_params_tag.json','a+') as f:
    #     f.write(json.dumps((grid_result.best_score_, grid_xgb.best_params_))+'\n')


# 使用训练好的数据在测试集上尝试效果+输出重要特征
def test(model, X_train, Y_train, X_test,Y_test):
    # XGBoost训练过程，下面的参数就是刚才调试出来的最佳参数组合
    ros = RandomOverSampler(random_state=0,sampling_strategy=0.25)
    print('过采样前：',len(X_train))
    X_train, Y_train = ros.fit_resample(X_train, Y_train)
    

    # # X_train,Y_train = SMOTE(random_state=42,sampling_strategy=0.5).fit_resample(X_train, Y_train)
    # # X_train, Y_train = ADASYN(random_state=0,sampling_strategy=0.5).fit_resample(X_train, Y_train)

    print('过采样后：',len(X_train))
    
    model.fit(X_train, Y_train)
    # model = joblib.load('model/xgb_classifier.pkl')

    # Predict training set
    train_predictions = model.predict(X_train)
    train_predprob = model.predict_proba(X_train)[:,1]
 
    # Print model report:
    # print "\nModel Report"
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(Y_train, train_predictions)
    print("fraud 正类Precision (Train): ", precision[1], "负类Precision (Train): ", precision[0])
    print("AUC Score (Train): %f" % metrics.roc_auc_score(Y_train, train_predprob))
    print("Recall: %f" % metrics.recall_score(Y_train, train_predictions))
    print("Precision: %f" % metrics.precision_score(Y_train, train_predictions))

    # 保存模型
    model_fname = "%s.pkl" % ("xgb_classifier")
    joblib.dump(model, os.path.join('model/', model_fname))

    # 对测试集进行预测
    test_predictions = model.predict(X_test)
    test_predprob = model.predict_proba(X_test)[:,1]

    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(Y_test, test_predictions)
    print("fraud 正类Precision (Test): ", precision[1], "负类Precision (Test): ", precision[0])
    print("AUC Score (Test): %f" % metrics.roc_auc_score(Y_test, test_predprob))
    print("Recall: %f" % metrics.recall_score(Y_test, test_predictions))
    print("Precision: %f" % metrics.precision_score(Y_test, test_predictions))


    # 显示重要特征
    # plot_importance(model)
    # plt.show()
    # print(X_test.columns,model.feature_importances_)

def predict():
    X_test = pd.read_csv('test/df_tag_trd_app_time_a.csv')

    # 处理标签数据(无flag变量)
    X=process_category(X_test)
    # X_sec = pd.read_csv('test/df_tag_sub_feature_test.csv')
    # tmp_numpy,n = reduce_dim(X_sec)
    # X_sec = pd.DataFrame(tmp_numpy,columns=['f'+str(i) for i in range(n)]) 
    # X = pd.concat([X,X_sec],axis=1)

    # 加载模型预测
    # model = joblib.load('model/xgb_classifier.pkl')
    # test_predprob = model.predict_proba(X)[:,1]

    print(X.shape)

    with open('result3.txt','w',encoding="utf-8") as f:
        for i in range(len(X_test)):
            f.write(X_test.loc[i,'id'].strip()+'\t'+'{:.14f}'.format(test_predprob[i])+'\n') 
    print(i)
    return

if __name__ == '__main__':
    TEST_SIZE = 0.20 # test size using_train_test_split
    NUMBER_KFOLDS = 5 #number of KFolds for cross-validation
    trainMode = True

    if trainMode:
        df_tag = pd.read_csv('train/df_tag_trd_app_time.csv')
        # df_tag = pd.read_csv('train/df_tag_trd_app.csv')

        # train_label_encoder(df_tag)
        # print('finished....')

        # 划分训练(含验证)和测试数据集
        Xall = process_category(df_tag)
        
        X_train, X_test, Y_train, Y_test = split_train_valid(Xall)
        


        # 模型构建
        clf = LogisticRegression(
                                class_weight='balanced',
                                penalty='l2',
                                C=1,
                                random_state=0
                                )
        # clf = "KNearest": KNeighborsClassifier(),
        # clf = "Support Vector Classifier": SVC(),
        # clf = "DecisionTreeClassifier": DecisionTreeClassifier()

        s=time.time()
        train(clf,X_train,Y_train)
        print('Traning time consumed:',time.time()-s)
        # test(clf,X_train, Y_train, X_test, Y_test)
    else:
        predict()
        