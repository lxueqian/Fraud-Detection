    # coding: utf-8

import numpy as np 
import pandas as pd
import pandas as pd 
import numpy as np
import gc
import json
import os
import time
# import pickle
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from xgboost import plot_importance

def train_label_encoder(df_beh):

    df_beh_a = pd.read_csv('test/beh_a.csv')
    df = df_beh[['page_no']].append(df_beh_a[['page_no']],ignore_index=True)

    df['page_no'] = df['page_no'].astype(str)
    le_page_no = LabelEncoder()

    df['page_no_encoded'] = le_page_no.fit_transform(df.page_no)

    joblib.dump(le_page_no, 'model/label/le_page_no_encoder.pkl')

    page_no_ohe = OneHotEncoder()

    X = page_no_ohe.fit_transform(df.page_no_encoded.values.reshape(-1,1)).toarray()

    joblib.dump(page_no_ohe, 'model/label/ohe_page_no_encoder.pkl')

    # df.drop(['page_no_encoded'],axis=1,inplace=True)

    return


#处理类别标签
def process_category(df_beh):

    df_beh['page_no'] = df_beh['page_no'].astype(str)
    df_beh['id'] = df_beh['id'].astype(str)

    le_page_no = joblib.load('model/label/le_page_no_encoder.pkl')
    le_id = LabelEncoder()

    df_beh['page_no_encoded'] = le_page_no.transform(df_beh.page_no)
    df_beh['id_encoded'] = le_id.fit_transform(df_beh.id)

    page_no_ohe=joblib.load('model/label/ohe_page_no_encoder.pkl')

    X = page_no_ohe.transform(df_beh.page_no_encoded.values.reshape(-1,1)).toarray()

    dfOhe = pd.DataFrame(X,columns=['page_no_'+str(int(i)) for i in range(X.shape[1])])
    df_beh = pd.concat([df_beh,dfOhe],axis=1)
  
    X = df_beh.drop(['id','page_no','page_no_encoded','page_tm'],axis=1)
    # y = df_beh['flag']

    return X

def reduce_dim(X):
    n_components = 0.99
    
    X_embedded = TSNE(n_components=n_components).fit_transform(X)
    X_embedded = PCA(n_components=n_components).fit_transform(X)
    print(X_embedded.shape)
    print(X_embedded)

    return X_embedded


# 训练集和测试集的划分
def train(clf,X_train,Y_train):
    params = {
            # 'learning_rate':[0.01, 0.02, 0.05, 0.1, 0.15],
            # 'n_estimators':[50, 100, 200, 300, 500],
            'subsample':[0.6, 0.7, 0.8, 0.9], #对于每棵树，随机采样的比例  0.9
            # 'colsample_bytree':[0.5, 0.6, 0.7, 0.8, 0.9], #随机采的列数的占比
            # 'scale_pos_weight':[0.3,0.4], #负/正 0.2
            # 'reg_lambda':[0, 0.1,0.25, 0.5, 0.75, 1], #L2正则
            # 'reg_alpha':[0.1,0.2, 0.4, 0.6, 0.8, 1], #L1正则
            # 'max_depth':[6,7,8]
            'min_child_weight':[0.8,0.9,1]
    }

    kFold = StratifiedKFold(n_splits=NUMBER_KFOLDS, random_state=0, shuffle=False)

    grid_xgb = GridSearchCV(clf, param_grid=params,scoring='roc_auc',cv=kFold,n_jobs=4)
    grid_result=grid_xgb.fit(X_train, Y_train,)

    print("Best: %f using %s" % (grid_result.best_score_, grid_xgb.best_params_))

    #查看所有参数组合
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean,param in zip(means,params):
        print("%f  with:   %r" % (mean,param))  

    #保存参数
    with open('model/best_params_beh.json','a+') as f:
        f.write(json.dumps(grid_xgb.best_params_)+'\n')

# 
# 使用训练好的数据在测试集上尝试效果+输出重要特征
def test(model, X_train, Y_train, X_test,Y_test):
    # XGBoost训练过程，下面的参数就是刚才调试出来的最佳参数组合
    # model.fit(X_train, Y_train)
    model = joblib.load('model/xgb_classifier_beh.pkl')

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

    #保存模型
    # model_fname = "%s.pkl" % ("xgb_classifier_beh")
    # joblib.dump(model, os.path.join('model/', model_fname))

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
    print(X_test.columns,model.feature_importances_)

def predict():
    X_test = pd.read_csv('test/beh_a.csv')

    # 处理标签数据(无flag变量)
    X=process_category(X_test)

    # 加载模型预测
    model = joblib.load('model/xgb_classifier_beh.pkl')
    test_predprob = model.predict_proba(X)[:,1]

    print(X.shape)

    X_test['flag'] = test_predprob

    X_test.to_csv('test/beh_a_.csv', header=True, index=False)

    print(X_test['flag'])

    return

if __name__ == '__main__':
    TEST_SIZE = 0.20 # test size using_train_test_split
    NUMBER_KFOLDS = 5 #number of KFolds for cross-validation
    trainMode = False

    if trainMode:
        df_beh = pd.read_csv('train/beh.csv')

        train_label_encoder(df_beh)

        #划分数据集 训练：测试
        X=process_category(df_beh)
        y=X['flag']


        X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=TEST_SIZE, 
                                            random_state=2018, stratify=y)

        X = X_train[X_train['flag']==0][:34000]
        X_train = X.append(X_train[X_train['flag']==1][:6000],ignore_index=True)
        Y_train = X_train['flag']

        X = X_test[X_test['flag']==0][:8400]
        X_test = X.append(X_test[X_test['flag']==1][:1600],ignore_index=True)
        Y_test = X_test['flag']

        X_train = X_train.drop(['flag'],axis=1)
        X_test = X_test.drop(['flag'],axis=1)

        print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

        #模型构建
        clf=XGBClassifier(        
                learning_rate=0.1,
                n_estimators=500,
                max_depth=7, #树的最大深度 3-10
                min_child_weight=1, #决定最小叶子节点样本权重和
                gamma=0, #默认0
                subsample=0.9, #对于每棵树，随机采样的比例 
                colsample_bytree=0.3, #随机采的列数的占比
                objective='binary:logistic',
                scale_pos_weight=0.7, #负/正
                seed=10, 
                reg_lambda=0.005, #L2正则
                reg_alpha=0.005, #L1正则
                base_score=0.3,  #正/所有
                max_delta_step=0
        )

        # s=time.time()
        # train(clf,X_train,Y_train)
        # print(time.time()-s)
        test(clf,X_train, Y_train, X_test, Y_test)
    else:
        predict()