#this code is adapted from: https://github.com/vmasrani/dementia_classifier/tree/master/dementia_classifier

import numpy as np
import pandas as pd
from adapt.feature_based import CORAL
from scipy import stats
# from dementia_classifier.analysis import util
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold,LeaveOneOut,train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, mean_squared_error, precision_score, \
    recall_score, log_loss
from scipy.cluster.vq import whiten
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression,RidgeClassifier,Ridge

from sklearn.svm import SVR,SVC
from sklearn.pipeline import make_pipeline
from data_handler import concat_and_shuffle
from sklearn.utils import check_array
from _dann import DANN
from _fmmd import fMMD
from _tradaboost import TrAdaBoost,TrAdaBoostR2
import tensorflow as tf
# from adapt.instance_based import TrAdaBoostR2

#import  tensorflow.keras.callbacks as Callback
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Input, Dense, Reshape
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import log_loss
# from keras.utils import np_utils
from keras.utils import to_categorical





ALZHEIMERS     = ['PossibleAD', 'ProbableAD']
CONTROL        = ['Control']
NON_ALZHEIMERS = ['MCI', 'Memory', 'Other', 'Vascular']
MCI = ['MCI']


# class SavePrediction(Callback):
#     """
#     Callbacks which stores predicted
#     labels in history at each epoch.
#     """
#     def __init__(self):
#         self.X = np.linspace(-0.7, 0.6, 100).reshape(-1, 1)
#         self.custom_history_ = []
#         super().__init__()
#
#     def on_epoch_end(self, batch, logs={}):
#         """Applied at the end of each epoch"""
#         predictions = self.model.predict_on_batch(self.X).ravel()
#         self.custom_history_.append(predictions)
class DementiaCV(object):
    """
    DementiaCV performs 10-fold group cross validation, where data points with a given label
    are confined in a single fold. This object is written with the intention of being a 
    superclass to the DomainAdaptation and BlogAdaptation subclasses, where the subclasses 
    only need to override the 'get_data_folds' method
    """

    def __init__(self, model=None, X=None, y=None,l=None, labels=None, silent=False,source='ccc',target='pitt',adresstest=False,nfold=10):
        super(DementiaCV, self).__init__()
        self.model = model
        self.path='../result/'
        # self.path = '/content/drive/My Drive/Colab Notebooks/code_da/result/dann/'

        self.X = X
        self.y = y
        self.l = l
        self.label_predictor=None
        self.domain_predictor=None
        self.feature_extractor=None

        self.labels = labels
        # self.columns = X.columns
        self.methods = ['multiaugment']
        # self.methods = ['default']
        self.nfolds = nfold
        # self.source=source
        # self.target=target
        # self.adresstest=adresstest

        # Results
        self.silent = silent
        self.results    = {}
        self.best_score = {}
        self.results_mmse = {}
        self.best_score_mmse = {}
        self.best_k     = {}

        self.myprint("Model %s" % model)
        self.myprint("===========================")

    def get_data_folds(self, fold_type='default'):
        X, y, labels = self.X, self.y, self.labels
        if X is None or y is None:
            raise ValueError("X or y is None")
        data = []
        if self.nfolds==1:
            fold = {}
            fold["X_train"] = X.values
            fold["y_train"] = y.values
            fold["X_test"] = self.Xt_test.values
            fold["y_test"] = self.yt_test.values
            data.append(fold)
            return data


        group_kfold = GroupKFold(n_splits=self.nfolds).split(X, y, groups=labels)
        for train_index, test_index in group_kfold:
            fold = {}
            fold["X_train"] = X.values[train_index]
            fold["y_train"] = y.values[train_index]
            fold["X_test"]  = X.values[test_index]
            fold["y_test"]  = y.values[test_index]
            fold["train_labels"]  = np.array(labels)[train_index]
            data.append(fold)
        return data

    def myprint(self, msg):
        if not self.silent:
            print(msg)
    def get_pivot_feature(self,feat,method,model_name,selective_features,model,feature_weights,krange):
        if  model_name == 'lr':
            model_name = 'logisticregression'
            # model = make_pipeline(StandardScaler(), LogisticRegression(C=c, penalty='l2',
            #                                                            max_iter=3000))  # c=bias-variance tradeoff hyperpaprameter
        elif  model_name == 'svm':
            model_name = 'svc'
            #
            # model = make_pipeline(StandardScaler(),
            #                       SVC(C=c, gamma='auto', kernel='poly', class_weight='balanced', probability=True))
            #
        if feat and 'multiaugment' not in method :
            # pivot

            weights = model.named_steps[model_name].coef_.ravel()
            joint_weights = weights[:np.size(self.Xt, 1)]
            feature_weights['joint_features'].append([self.features[i] for i in range(len(joint_weights))])
            feature_weights['joint_features_weight'].append([joint_weights[i] for i in range(len(joint_weights))])

            sort_idx_joint = np.argsort(weights[:np.size(self.Xt, 1)])
            sort_idx_joint = sort_idx_joint.reshape((-1, 1))
            selective_features['joint_features'].append([self.features[i[0]] for i in sort_idx_joint[-krange:]])
            selective_features['joint_features'].append([self.features[i[0]] for i in sort_idx_joint[:krange]])

            # source
            source_weights = weights[np.size(self.Xt, 1):np.size(self.Xt, 1) * 2]
            feature_weights['source_features'].append([self.features[i] for i in range(len(source_weights))])
            feature_weights['source_features_weight'].append([source_weights[i] for i in range(len(source_weights))])

            sort_idx_s = np.argsort(weights[np.size(self.Xt, 1):np.size(self.Xt, 1) * 2])
            sort_idx_s = sort_idx_s.reshape((-1, 1))

            selective_features['s_features'].append([self.features[i[0]] for i in sort_idx_s[-krange:]])
            selective_features['s_features'].append([self.features[i[0]] for i in sort_idx_s[:krange]])

            # target
            target_weights = weights[np.size(self.Xt, 1) * 2:np.size(self.Xt, 1) * 3]
            feature_weights['target_features'].append([self.features[i] for i in range(len(target_weights))])
            feature_weights['target_features_weight'].append([target_weights[i] for i in range(len(target_weights))])

            sort_idx_t = np.argsort(weights[np.size(self.Xt, 1) * 2:np.size(self.Xt, 1) * 3])
            sort_idx_t = sort_idx_t.reshape((-1, 1))
            selective_features['t_features'].append([self.features[i[0]] for i in sort_idx_t[-krange:]])
            selective_features['t_features'].append([self.features[i[0]] for i in sort_idx_t[:krange]])
        elif feat and 'multiaugment' in method:
            # pivot
            # last portion of columns source,target, general
            weights = model.named_steps[model_name].coef_.ravel()
            start = np.size(self.Xt, 1) * (self.n_domains_ + 1)
            stop = np.size(self.Xt, 1) * (self.n_domains_ + 2)
            joint_weights = weights[start:stop]
            feature_weights['joint_features_multidom'].append([self.features[i] for i in range(len(joint_weights))])
            feature_weights['joint_features_weight_multidom'].append([joint_weights[i] for i in range(len(joint_weights))])

            sort_idx_joint = np.argsort(weights[start:stop])
            sort_idx_joint = sort_idx_joint.reshape((-1, 1))
            selective_features['joint_features_multidom'].append([self.features[i[0]] for i in sort_idx_joint[-krange:]])
            selective_features['joint_features_multidom'].append([self.features[i[0]] for i in sort_idx_joint[:krange]])

            # source
            start = 0

            stop = np.size(self.Xt, 1) * self.n_domains_
            #source dom 1
            source_weights = weights[start:np.size(self.Xt, 1) * 1]
            feature_weights['source_features_multidom_1'].append([self.features[i] for i in range(len(source_weights))])
            feature_weights['source_features_weight_multidom_1'].append(
                [source_weights[i] for i in range(len(source_weights))])
            #source dom 2
            source_weights = weights[np.size(self.Xt, 1) * 1:np.size(self.Xt, 1) * self.n_domains_]
            feature_weights['source_features_multidom_2'].append([self.features[i] for i in range(len(source_weights))])
            feature_weights['source_features_weight_multidom_2'].append(
                [source_weights[i] for i in range(len(source_weights))])

            sort_idx_s = np.argsort(weights[start:np.size(self.Xt, 1) * 1])
            sort_idx_s = sort_idx_s.reshape((-1, 1))

            selective_features['s_features_multidom_1'].append([self.features[i[0]] for i in sort_idx_s[-krange:]])
            selective_features['s_features_multidom_1'].append([self.features[i[0]] for i in sort_idx_s[:krange]])

            sort_idx_s = np.argsort(weights[np.size(self.Xt, 1) * 1:np.size(self.Xt, 1) * self.n_domains_])
            sort_idx_s = sort_idx_s.reshape((-1, 1))

            selective_features['s_features_multidom_2'].append([self.features[i[0]] for i in sort_idx_s[-krange:]])
            selective_features['s_features_multidom_2'].append([self.features[i[0]] for i in sort_idx_s[:krange]])

            # target

            stop = np.size(self.Xt, 1) * (self.n_domains_ + 1)

            start = np.size(self.Xt, 1) * self.n_domains_
            target_weights = weights[start:stop]
            feature_weights['joint_features_multidom'].append([self.features[i] for i in range(len(target_weights))])
            feature_weights['joint_features_weight_multidom'].append(
                [target_weights[i] for i in range(len(target_weights))])

            sort_idx_t = np.argsort(weights[start:stop])
            sort_idx_t = sort_idx_t.reshape((-1, 1))
            selective_features['t_features_multidom'].append([self.features[i[0]] for i in sort_idx_t[-krange:]])
            selective_features['t_features_multidom'].append([self.features[i[0]] for i in sort_idx_t[:krange]])

    def nn_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, activation='relu', input_dim=104))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(0.02), loss='mean_squared_error')
        return model

    def get_model(self,model_name,method,c=None,dign='ad',n_estimator=None,lr=None):

        if model_name=='lr':

            model = make_pipeline(StandardScaler(),LogisticRegression(C=c,penalty='l2', max_iter=3000)) #c=bias-variance tradeoff hyperpaprameter

        elif  model_name == 'svm':
            if dign == 'mmse':
                model=make_pipeline(StandardScaler(), SVR(C=c,gamma='auto',kernel='poly'))

            else:
                model = make_pipeline(StandardScaler(),SVC(C=c,gamma='auto',kernel='poly',class_weight='balanced',probability=True))#for consistent bias
                # model = make_pipeline(StandardScaler(),SVC(C=c,gamma='auto',kernel='poly',probability=True)) # augment (l+s) , ccc, db
                # model = make_pipeline(StandardScaler(),SVC(C=c,kernel='poly',probability=True)) # augment (l+s) , ccc, db

                # model = make_pipeline(StandardScaler(),SVC(C=c,gamma='auto',kernel='rbf',probability=True)) #db , ccc, augment l,s,a,
                # model = make_pipeline(StandardScaler(),SVC(C=c,kernel='rbf',probability=True))
                # model = make_pipeline(StandardScaler(),SVC(C=c,gamma='auto',kernel='linear',probability=True))

        elif model_name == 'dann':
            model = DANN(loss=tf.keras.losses.CategoricalCrossentropy(), lambda_=c, metrics=["acc"], random_state=0)
            return model
        if 'dann' not in model_name:
            if 'fmmd' in method:
                estimator = model
                # estimator=Ridge()
                model = fMMD(estimator,  kernel="rbf", random_state=0, verbose=0)
                return model
            if 'tradaboost' in method:
                # print('learning rate %f'%(lr))
                if n_estimator is None:
                    n_estimator=20

                estimator = model
                # estimator = self.nn_model()
                if dign=='mmse':
                    model = CORAL(self.nn_model(), lambda_=1e-3, random_state=0)

                    # model = TrAdaBoostR2(estimator=self.nn_model(), n_estimators=n_estimator, random_state=0)
                else:
                    model=TrAdaBoost(estimator, n_estimators=n_estimator, lr=1.0, random_state=0)
                return model

        return model
    def print_feat_weights(self,weights):
        print(weights)
        for elem in weights:
            print(elem)
            print(len(weights[elem]))
    def train_model(self,model_name,c, method='default',feat_list=None,dign='ad',n_estimator=None, k_range=None, model=None):


        acc = []
        fms = []
        rmse=[]
        r_2=[]
        # roc = []
        if 'dann' in model_name :
            result = {'true': [], 'pred': []}
        else:
            # if self.nfolds==1:
            #     result = {'pred': []}
            # else:

                result={'true':[],'pred':[],'prob':[]}
                result = {'true': [], 'pred': []}

        if 'multiaugment' in self.methods:
            selective_features = {'joint_features_multidom': [], 's_features_multidom_1': [], 's_features_multidom_2': [], 't_features_multidom': []}
            feature_weights = {'joint_features_multidom': [], 'joint_features_weight_multidom': [],
                               'source_features_multidom_1': [],'source_features_multidom_2': [],
                               'target_features_multidom': [], 'source_features_weight_multidom_1': [],
                               'source_features_weight_multidom_2': [],
                               'target_features_weight_multidom': []}

        elif 'augment' in  self.methods:
            selective_features={'joint_features':[],'s_features':[],'t_features':[]}
            feature_weights={'joint_features':[],'joint_features_weight':[],'source_features':[],'target_features':[],'source_features_weight':[],'target_features_weight':[]}
        feat = False #calculate and save feature weights

        #adress_test data as target
        model=self.get_model(model_name,method,c,dign=dign,n_estimator=n_estimator)
        if self.nfolds==1:
            for idx, fold in enumerate(self.get_data_folds(method)):

                if 'dann' not in model_name:
                    X_train, y_train = fold["X_train"], fold["y_train"].ravel()  # Ravel flattens a (n,1) array into (n, )
                    if fold["y_test"] is not None:
                        X_test, y_test   = fold["X_test"], fold["y_test"].ravel()
                    else:
                        X_test, y_test   = fold["X_test"], None


                else:
                    X_train, y_train = fold["X_train"], fold["y_train"] # Ravel flattens a (n,1) array into (n, )
                    X_test, y_test = fold["X_test"], fold["y_test"]


                if y_test is not None and  y_test.all():
                    print(y_test)
                    print("All values in y_test are the same in fold 1, ROC not defined.")
                roc_scores = []


                if 'dann' not in model_name:
                    if 'fmmd' in method or 'tradaboost' in method:
                        # save_preds = SavePrediction()

                        model = model.fit(X=X_train, y=y_train,Xt=fold["Xt_train"], yt=fold["yt_train"].ravel(), epochs=100, verbose=0,domains=[0,1])
                    else:
                        model = model.fit(X_train, y_train)
                else:
                    model=model.fit(X_train, y_train,Xt=fold["Xt_train"],yt=fold["yt_train"], epochs=100, verbose=0,domains=[0,1])


                # feat=False #whether to record feature selection
                # #feature importance analysis
                # if feat and 'augment' in method:
                #     self.get_pivot_feature(feat, method, model_name, selective_features, model,feature_weights=feature_weights)

                # Predict
                if 'dann' not in model_name:

                    yhat = model.predict(X_test)
                # print('Prediction.................')
                # print(yhat)
                # ----- save fold -----
                acc.append(accuracy_score(y_test, yhat))
                fms.append(f1_score(y_test, yhat))
                # if 'dann' not in model_name and method  not in ['fmmd','tradaboost'] :
                #     yhat_probs = model.predict_proba(X_test)
                #
                #     if y_test.all():
                #         roc_scores.append(np.nan)
                #     else:
                #         roc_scores.append(roc_auc_score(y_test, yhat_probs[:, 1]))
                #
                #
                #     roc.append(roc_scores)
                #     result['prob'].extend(yhat_probs)

                #----------save_raw_result------------
                result['true'].extend(y_test)
                result['pred'].extend(yhat)
                # result['loss'].extend(loss)




        else:
            import collections
            for idx, fold in enumerate(self.get_data_folds(method)):
                self.myprint("Processing fold: %i" % idx)
                if 'dann' in model_name :
                    X_train, y_train = fold["X_train"], fold["y_train"]  # Ravel flattens a (n,1) array into (n, )
                    X_test, y_test   = fold["X_test"], fold["y_test"]
                else:
                    X_train, y_train = fold["X_train"], fold["y_train"].ravel()  # Ravel flattens a (n,1) array into (n, )
                    X_test, y_test = fold["X_test"], fold["y_test"].ravel()
                # counter = collections.Counter(y_train)

                # print(np.array(X_train).shape)
                # print(np.array(X_test).shape)
                # print(np.array(y_train).shape)
                # print(np.array(y_test).shape)


                # counter = collections.Counter(y_test)
                #
                # print(counter)

                # acc_scores = []
                # fms_scores = []
                # print('data train len')
                #
                # print(len(y_train))
                # unique, counts = np.unique(y_train, return_counts=True)
                # print(np.asarray((unique, counts)).T)
                #
                # print('data test len')
                #
                # print(len(y_test))
                # unique, counts = np.unique(y_test, return_counts=True)
                # print(np.asarray((unique, counts)).T)

                if y_test.all():
                    print("All values in y_test are the same in fold %i, ROC not defined." % idx)
                roc_scores = []

                # nfeats = X_train.shape[1]
                # feats = util.get_top_pearson_features(X_train, y_train, nfeats)
                # if k_range is None:
                #     k_range = range(1, nfeats)
                # if k_range[0] == 0:
                #     raise ValueError("k_range cannot start with 0")
                # for k in k_range:
                #     indices = feats[:k]
                    # Select k features
                    # X_train_fs = X_train[:, indices]


                if 'dann' in model_name:
                    # model=model.fit(X_train, y_train,domains=[0,1], epochs=20, verbose=0)
                    model=model.fit(X_train, y_train,Xt=fold["Xt_train"],yt=fold["yt_train"], epochs=100, verbose=0,domains=[0,1])

                else:
                    if 'fmmd' in method or 'tradaboost' in method :
                        # model=model.fit(X=X_train,y=y_train,Xt=fold["Xt_train"],yt=fold["yt_train"].ravel(),epochs=100, batch_size=32)
                        # print(np.array(fold["Xt_train"]).shape)
                        # print(np.array(fold["Xt_train"]).reshape(-1,1).shape)
                        #
                        # print(np.array(fold["yt_train"]).shape)

                        model=model.fit(X=X_train,y=np.array(y_train).reshape(-1,1),Xt=fold["Xt_train"],yt=np.array(fold["yt_train"]).reshape(-1,1))

                    else:

                        model = model.fit(X_train, y_train)
                # feature importance analysis
                if feat and 'augment' in method:
                    self.get_pivot_feature(feat, method, model_name, selective_features, model,feature_weights=feature_weights,krange=20)


                # Predict
                if dign=='ad':
                    if 'dann' not in model_name:
                        yhat = model.predict(X_test)
                        # if 'tradaboost' in method:
                        #     print('source sample weights')
                        #     src_wght=model. predict_weights(domain='src')
                        #     print(src_wght)
                        #     print('target sample weights')
                        #     tgt_wght = model.predict_weights(domain='tgt')
                        #     print(tgt_wght)


                        # y_test=np.argmax(y_test, axis=-1)
                    else:
                        y_test=np.argmax(y_test, axis=-1)

                        # print('real')
                        # print(y_test)
                        yhat=np.argmax(model.predict_task(X_test), axis=-1)
                else:
                    yhat=model.predict(X_test)
                    r2=model.score(X_test,y_test)
                # loss = log_loss(X_test, yhat_probs, eps=1e-15)

                # Save
                # acc_scores.append(accuracy_score(y_test, yhat))
                # fms_scores.append(f1_score(y_test, yhat))

                # ----- save fold -----
                # print(y_test)
                # print('hat')
                # print(yhat)
                if dign=='ad' and self.nfolds>1:
                    acc.append(accuracy_score(y_test, yhat))
                    fms.append(f1_score(y_test, yhat))

                    if 'dann' not in model_name and  method not in ['fmmd','tradaboost']:
                        yhat_probs = model.predict_proba(X_test)

                    # if y_test.all():
                    #     roc_scores.append(np.nan)
                    # else:
                    #     roc_scores.append(roc_auc_score(y_test, yhat_probs[:, 1]))
                    #
                    # roc.append(roc_scores)
                    # result['prob'].extend(yhat_probs)
                    #
                    # roc.append(roc_scores)

                    #----------save_raw_result------------

                    # result['loss'].extend(loss)
                elif dign=='mmse' and self.nfolds>1:
                    rmse.append(mean_squared_error(y_test,yhat))
                    r_2.append(r2)
                if self.nfolds>1:
                    result['true'].extend(y_test)
                result['pred'].extend(yhat)

        #for cross-validation exp.
        if dign=='ad' and self.nfolds>=1:
            self.results[method] = {"acc": np.asarray(acc),
                                    "fms": np.asarray(fms)
                                    # "roc": np.asarray(roc)
                                    }

            self.best_score[method] = {"acc": np.nanmean(acc, axis=0),
                                       "fms": np.nanmean(fms, axis=0),
                                       "acc_stdev":np.nanstd(acc,axis=0),
                                       "fms_stdev":np.nanstd(fms,axis=0)

                                       # "roc": np.nanmean(roc, axis=0)

                                       }
        elif dign=='mmse' and self.nfolds>1:
            self.results_mmse[method] = {"rmse": np.asarray(rmse),
                                    "r2": np.asarray(r_2)
                                    # "roc": np.asarray(roc)
                                    }

            self.best_score_mmse[method] = {"rmse": np.nanmean(rmse, axis=0),
                                       "r2": np.nanmean(r_2, axis=0)
                                       # "roc": np.nanmean(roc, axis=0)

                                       }

        ################result#############################################
        if n_estimator is None:
            n_estimator=0
        print('result for method %s param %f nest %d source %s target %s'%(method,c,n_estimator,self.source,self.target))
        if self.nfolds>=1:
            if dign=='ad':
                # print(self.best_score[method])
                print(self.results[method])
                print(self.best_score[method])
                print(result['true'])
                print(result['pred'])
            elif dign=='mmse':
                print(self.best_score_mmse[method])
                print(self.results_mmse[method])
                print(result['true'])




        print(result['pred'])
        # print( result['pred'].count(1))


        # self.save_file(pd.DataFrame(self.best_score),'metric_'+method+'_'+'feat_'+str(feat_list)+'_'+model_name+'_'+str(c)+'_'+'nest_'+str(n_estimator)+str(self.source)+'_'+str(self.target)+'_'+str(self.nfolds)+'_k'+str(k_range),method)
        if self.nfolds==1:
            print(result)
            self.save_file(pd.DataFrame(result),'raw_results_madress_'+method+'feat_'+str(feat_list)+'_'+'_'+model_name+'_'+str(c)+'_'+'nest_'+str(n_estimator)+str(self.source)+'_'+str(self.target)+'_'+str(self.nfolds),method)

        if feat and ('augment' in self.methods) or ('multiaugment' in self.methods):
            # self.save_file(pd.DataFrame(selective_features),'selective_features_'+method+'_'+'feat_'+str(feat_list)+'_'+model_name+'_'+str(c)+'nest_'+str(n_estimator)+'_'+str(self.source)+'_'+str(self.target)+'_'+str(self.nfolds)+'_k'+str(k_range),method)
            # self.print_feat_weights(feature_weights)
            self.save_file(pd.DataFrame(feature_weights),'feature_weights'+method+'_'+'feat_'+str(feat_list)+'_'+model_name+'_'+str(c)+'nest_'+str(n_estimator)+'_'+str(self.source)+'_'+str(self.target)+'_'+str(self.nfolds)+'_k'+str(k_range),method)

        if dign=='ad' and self.nfolds>=1:
            print(classification_report(result['true'], result['pred'], target_names=['non-AD','AD']))

        # self.save_file(pd.DataFrame(self.results[method]),'results_'+method+'_'+model_name+'_'+str(c))



        # if dign=='ad' :
        #     return self.results[method]
        # elif dign=='mmse':
        #     return self.results_mmse[method]
        return result,self.best_score[method]



    import pandas as pd
    def save_file(self,df, name,dir):
        # df.to_pickle(self.path+dir+'/' + name + '.pickle')
        df.to_csv(self.path+dir+'/' + name + '.csv')
        print('done')

    # def feature_rank(self, method='default', thresh=50):
    #     nfeats = self.columns.shape[0]
    #     nfolds = self.nfolds
    #     feat_scores = pd.DataFrame(np.zeros([nfeats, nfolds]), columns=range(nfolds), index=self.columns)
    #
    #     for fold_idx, fold in enumerate(self.get_data_folds(method)):
    #         X_train, y_train = fold["X_train"], fold["y_train"].ravel()  # Ravel flattens a (n,1) array into (n, )
    #         ranked_features = util.get_top_pearson_features(X_train, y_train, nfeats)
    #         for rank, feat_idx in enumerate(ranked_features[:thresh]):
    #             feature = self.columns[feat_idx]
    #             weight = (thresh - rank) / float(thresh)
    #             feat_scores[fold_idx].ix[feature] = weight
    #
    #     # Drop rows with all zeros
    #     df = feat_scores[(feat_scores.T != 0).any()]
    #     df = df.stack().reset_index()
    #     df.columns = ['feature', 'fold', 'weight']
    #
    #     return df


class DomainAdaptationCV(DementiaCV):
    """Subclass of DementiaCV where the six domain adaptation methods 
    are computed.
    """

    def __init__(self, methods, Xt, yt,lt, Xs, ys,ls,Xt_con=None, yt_con=None, lt_con=None, silent=False,model=None,source='ccc',target='pitt',features=None,domains=2,Xt_test=None,yt_test=None,lt_test=None, random_state=1,nfold=8):
        super(DomainAdaptationCV, self).__init__(model, X=Xt, y=yt,l=lt, silent=silent)
        self.silent = silent
        self.Xt, self.yt,self.lt = Xt, yt,lt  # Save target data + labels
        self.Xs, self.ys,self.ls = Xs, ys,ls  # Save source data + labels
        self.Xt_con, self.yt_con, self.lt_con = Xt_con, yt_con, lt_con
        self.Xt_test, self.yt_test,self.lt_test = Xt_test, yt_test,lt_test  # Save target data + labels
        # self.target_split=0.3
        # self.methods = ['baseline','augment','coral']
        # self.methods =['augment']
        self.methods = methods

        # encode class values as integers
        # encoder = LabelEncoder()
        # convert integers to dummy variables (i.e. one hot encoded)
        # dummy_y = to_categorical(encoded_Y)
        if  type(self.Xs) is list and len(self.Xs)>1: #multi source domain case
            for i in range(len(Xs)):
                self.Xs[i] = self.Xs[i].to_numpy()
            for i in range(len(ys)):
                if 'dann' in self.methods[0]:
                    label_encoder = LabelEncoder()

                    ytrain = label_encoder.fit_transform(ys[i].to_numpy())

                    onehot_encoder = OneHotEncoder(sparse=False)
                    self.ys[i] = onehot_encoder.fit_transform(ytrain.reshape(-1, 1))
                else:
                    self.ys[i] = self.ys[i].to_numpy()
            # encoder.fit(ys[i])
                # self.ys[i] = to_categorical(encoder.transform(ys[i]))
            # self.Xt=self.Xt.to_numpy()
            # self.yt=self.yt.to_numpy()
            # encoder = LabelEncoder()

            # encoder.fit(yt)
            # self.yt = to_categorical(encoder.transform(yt))
            self.yt=self.yt.to_numpy()
            self.n_domains_ = len(self.Xs)
        else:
            if 'dann' in self.methods[0]:
                label_encoder = LabelEncoder()

                ytrain = label_encoder.fit_transform(ys.to_numpy())

                onehot_encoder = OneHotEncoder(sparse=False)
                self.ys = onehot_encoder.fit_transform(ytrain.reshape(-1, 1))
            else:

                self.ys = ys.to_numpy()
            self.yt = yt.to_numpy()
            # encoder = LabelEncoder()
            # encoder.fit(self.ys)
            # self.ys = to_categorical(encoder.transform(self.ys))
            # encoder = LabelEncoder()
            # encoder.fit(self.yt)



            # self.yt = to_categorical(encoder.transform(self.yt))




        # print('shapeeeeeeeeeeeee.....................')
        # print(self.ys.shape)
        self.nfolds=nfold
        self.source=source
        self.target=target
        self.features=features



    def whiten_matrix(self,X):
        '''Computes the square root matrix of symmetric square matrix X.'''
        (L, V) = np.linalg.eigh(X)
        return V.dot(np.diag(np.power(L, -0.5))).dot(V.T)

    def color_matrix(self,X):
        '''Computes the square root matrix of symmetric square matrix X.'''
        (L, V) = np.linalg.eigh(X)
        return V.dot(np.diag(np.power(L, 0.5))).dot(V.T)
    #get data folds as per cjhangye paper with ccc paper

    def get_data_folds(self, fold_type):
        if fold_type not in self.methods:
            raise KeyError('fold_type not one of: %s' % self.methods)

        data = []
        source_target_percnt={'src':[],'y_tgt_1':[],'y_src_1':[]}
        dign_mci=False
        if self.nfolds>1:
            lt = self.Xt.index
            ls = self.ls
        Xs, ys = self.Xs, self.ys
        Xt, yt = self.Xt, self.yt
        if self.Xt_con is not None:
            Xt_con, yt_con,lt_con = self.Xt_con.values, self.yt_con.values,self.Xt_con.index.values
            dign_mci=True

        # folds={}
        # Split target samples into target/source set (making sure one patient doesn't appear in both t and s)
        if self.nfolds==1:
            #for adres test set
            # print(len(self.Xt_test))
            # print(len(self.yt_test))

            # joint baseline
            if fold_type == 'baseline':
                # merge'em
                X_merged_relab = np.concatenate([Xs, Xt])
                y_merged_relab = np.concatenate([ys, yt])
                # train_labels   = np.concatenate([np.array(ls), np.array(lt)])
                # shuffle
                X_train_relab, y_train_relab = shuffle(
                    X_merged_relab, y_merged_relab, random_state=1)
                X_train = X_train_relab
                y_train = y_train_relab
                X_test = self.Xt_test.values
                if self.yt_test is not None:

                    y_test = self.yt_test.values
                else:
                    y_test = self.yt_test

            elif fold_type == 'source_only':
                X_train = Xs
                y_train = ys
                X_test = self.Xt_test.values
                if self.yt_test is not None:
                    y_test = self.yt_test.values
                else:
                    y_test = self.yt_test
            elif fold_type == 'target_only':
                X_train = Xt
                y_train = yt
                print(len(X_train))
                X_test = self.Xt_test.values
                if self.yt_test is not None:
                    y_test = self.yt_test.values
                else:
                    y_test = self.yt_test




            elif fold_type == 'augment':
                source_target_percnt['src'].append(
                    (len(Xs) ) / ((len(Xs) +  + len(Xt))))
                # source_target_percnt['tgt'].append(
                #     (len(Xt)) / ((len(Xs)  + len(Xt))))
                source_target_percnt['y_tgt_1'].append(
                    (np.count_nonzero(yt==1)) / ((len(yt) )))

                source_target_percnt['y_src_1'].append(
                    (np.count_nonzero(ys == 1)) / ((len(ys) )))


                # Extend feature space (train)
                X_merged_aug = self.merge_and_extend_feature_space(Xt, Xs)
                y_merged_aug = np.concatenate([yt, ys])
                # train_labels = np.concatenate([np.array(lt), np.array(ls)])
                # Extend feature space (test)
                X_test_aug = self.merge_and_extend_feature_space(self.Xt_test.values)
                X_train = X_merged_aug
                y_train = y_merged_aug

                X_test = X_test_aug
                if self.yt_test is not None:

                    y_test = self.yt_test.values
                else:
                    y_test = self.yt_test

            elif fold_type == 'multiaugment':

                source_target_percnt['src'].append(
                    (len(Xs[0]) + len(Xs[1])) / ((len(Xs[0]) + len(Xs[1]) + len(Xt))))
                source_target_percnt['y_tgt_1'].append(
                    (np.count_nonzero(yt == 1)) / ((len(yt))))

                source_target_percnt['y_src_1'].append(
                    ((np.count_nonzero(ys[0] == 1)) + (np.count_nonzero(ys[1] == 1))) / ((len(ys[0]) + len(ys[1]))))
                # Extend feature space (train) for multidomain augment

                # Extend feature space (train) for multidomain augment
                # print(train_index)

                # print(temp)
                X_merged_aug, y_merged_aug = self.merge_and_extend_feature_space_multidomain(Xt=Xt, Xs=Xs,
                                                                                             ys=ys, yt=yt)
                # print('######################################')
                # print(yt[train_index].shape)
                # print(type(yt[train_index]))
                # print(ys.shape)
                # print(type(ys))
                # y_merged_aug = np.concatenate([yt[train_index], ys])
                # train_labels = np.concatenate([np.array(lt)[train_index], np.array(ls)])
                # Extend feature space (test)
                X_test_aug, ytest = self.merge_and_extend_feature_space_multidomain(Xt=self.Xt_test.to_numpy())
                X_train = X_merged_aug
                y_train = y_merged_aug
                X_test = X_test_aug
                y_test = self.yt_test.to_numpy()
            elif fold_type in ['fmmd','tradaboost']:
                # ---------coral------------
                X_train = Xs
                y_train = ys
                Xt_train=Xt
                yt_train=yt

                X_test  = self.Xt_test.values
                if self.yt_test is not None:

                    y_test  = self.yt_test.values
                else:
                    y_test = self.yt_test


            elif fold_type == 'coral':
                # ---------coral------------
                X_train = self.CORAL(Xs, Xt)
                y_train = ys
                X_test  = self.Xt_test.values
                if self.yt_test is not None:

                    y_test  = self.yt_test.values
                else:
                    y_test = self.yt_test

                # train_labels = ls
            else:
                raise KeyError('fold_type not one of: %s' % str(self.methods))
            fold = {}
            fold["X_train"] = X_train
            fold["y_train"] = y_train
            fold["X_test"]  = X_test
            fold["y_test"]  = y_test
            if fold_type in ['dann','fmmd','tradaboost']:
                fold["Xt_train"]=Xt_train
                fold["yt_train"] = yt_train
            # fold["train_labels"] = train_labels
            data.append(fold)
        else:
            #####################  for dign mci ############################################
            if dign_mci==True:
                # print('before concat')
                # print('mci full len %d %d'%(len(Xt),len(yt)))
                # print('con full len %d %d'%(len(Xt_con),len(Xt_con)))


                group_kfold_mci = GroupKFold(n_splits=10).split(Xt, yt, groups=lt)
                group_kfold_con = GroupKFold(n_splits=10).split(Xt_con, Xt_con, groups=lt_con)
                # print(group_kfold_mci)
                # print(group_kfold_con)
                for mci,con in zip(group_kfold_mci,group_kfold_con):
                    # folds[count]={}
                    # folds['train']=train_index
                    # folds['test']=test_index
                    # count+=1
                    train_index_mci, test_index_mci=mci
                    train_index_con, test_index_con=con
                    # print('after split')
                    # print('train mci %d  test mci %d'%(len(train_index_mci),len(test_index_mci)))
                    # print(test_index_mci)
                    # print(type(yt[train_index_mci]))

                    Xt_train,yt_train= concat_and_shuffle(Xt[train_index_mci], yt[train_index_mci], lt[train_index_mci], Xt_con[train_index_con], yt_con[train_index_con], np.array(lt_con)[train_index_con], random_state=1)
                    Xt_test,yt_test= concat_and_shuffle(Xt[test_index_mci], yt[test_index_mci], lt[test_index_mci], Xt_con[test_index_con], yt_con[test_index_con], np.array(lt_con)[test_index_con], random_state=1)
                    yt_train = np.asarray([val[0] for val in yt_train])
                    yt_test = np.asarray([val[0] for val in yt_test])


                    # print(type(yt_train))
                    # print(yt_train)


                    # if fold_type == 'baseline':
                    #     # merge'em
                    #     X_merged_relab = np.concatenate([Xs, Xt[train_index]])
                    #     y_merged_relab = np.concatenate([ys, yt[train_index]])
                    #     train_labels = np.concatenate([np.array(ls), np.array(lt)[train_index]])
                    #     # shuffle
                    #     X_train_relab, y_train_relab, train_labels = shuffle(
                    #         X_merged_relab, y_merged_relab, train_labels, random_state=1)
                    #     X_train = X_train_relab
                    #     y_train = y_train_relab
                    #     X_test = Xt[test_index]
                    #     y_test = yt[test_index]
                    if fold_type == 'augment':
                        # Extend feature space (train)
                        X_merged_aug = self.merge_and_extend_feature_space(Xt_train, Xs)
                        # print('######################################')
                        #
                        # print(yt_train.shape)
                        # print(type(yt_train))
                        # print(ys.shape)
                        # print(type(ys))


                        # print(yt_train)
                        y_merged_aug = np.concatenate([yt_train, ys])

                        # y_merged_aug = np.concatenate([yt_train, ys])
                        # train_labels = np.concatenate([np.array(lt)[train_index], np.array(ls)])
                        # Extend feature space (test)
                        X_test_aug = self.merge_and_extend_feature_space(Xt_test)
                        X_train = X_merged_aug
                        y_train = y_merged_aug
                        X_test = X_test_aug
                        y_test = yt_test

                    elif fold_type == 'coral':
                        # ---------coral------------
                        X_train = self.CORAL(Xs, Xt_train)
                        y_train = ys
                        X_test = Xt_test
                        y_test = yt_test
                        train_labels = ls
                    else:
                        raise KeyError('fold_type not one of: %s' % self.models)
                    fold = {}
                    fold["X_train"] = X_train
                    fold["y_train"] = y_train
                    fold["X_test"] = X_test
                    fold["y_test"] = y_test
                    # fold["train_labels"] = train_labels
                    data.append(fold)
                print('percent source data in training %.3f' % (np.mean(np.array(source_target_percnt['src']), axis=0)))
                print('percent target data in training %.3f' % (np.mean(np.array(source_target_percnt['tgt']), axis=0)))

                return data

            ###################################################dign ad###########################################
            #############for creating separate test set from target data########################################
            # self.Xt_train, self.Xt_test, self.yt_train, self.yt_test = train_test_split(
            #     self.Xt, self.yt, test_size=self.target_split, random_state=0,statify=self.yt)
            # #for cross validation
            if self.nfolds==8:
                group_kfold = LeaveOneOut().split(Xt, yt,groups=lt)
            else:
                group_kfold = StratifiedGroupKFold(n_splits=self.nfolds).split(Xt, yt, groups=lt)
            # print(Xt)
            Xt = Xt.to_numpy()
            # yt = yt.to_numpy()
            # for i in range(len(Xs)):
            #     Xs[i] = Xs[i].to_numpy()
            # for i in range(len(ys)):
            #     ys[i] = ys[i].to_numpy()
            for train_index, test_index in group_kfold:
                # folds[count]={}
                # folds['train']=train_index
                # folds['test']=test_index
                # count+=1





                if fold_type == "target_only":
                    X_train = Xt[train_index]
                    y_train = yt[train_index]
                    X_test  = Xt[test_index]
                    y_test  = yt[test_index]
                    train_labels = np.array(lt)[train_index]
                elif fold_type == 'source_only':
                    X_train = Xs
                    y_train = ys
                    X_test  = Xt[test_index]
                    y_test  = yt[test_index]
                    # train_labels = ls
                elif fold_type in ['fmmd','tradaboost']:
                    # ---------coral------------
                    X_train = Xs
                    y_train = ys
                    Xt_train = Xt[train_index]
                    yt_train = yt[train_index]

                    X_test = Xt[test_index]
                    y_test = yt[test_index]
                elif fold_type=='dann':
                    if type(self.Xs) is list and len(self.Xs) > 1:  # multi source domain case
                        source_target_percnt['src'].append(
                            (len(Xs[0])+len(Xs[1])) / ((len(Xs[0])+len(Xs[1]) + len(Xt[train_index]))))
                        source_target_percnt['tgt'].append(
                            (len(Xt[train_index])) / ((len(Xs[0])+len(Xs[1]) + len(Xt[train_index]))))
                    else:


                        source_target_percnt['src'].append(
                            (len(Xs)) / ((len(Xs)  + len(Xt[train_index]))))
                        source_target_percnt['tgt'].append(
                            (len(Xt[train_index])) / ((len(Xs)  + len(Xt[train_index]))))


                    X_train = Xs
                    y_train = ys
                    # print('cross_vaidator')
                    # print(X_train.shape)

                    # print(y_train.shape)




                    Xt_train=Xt[train_index]

                    yt_train=yt[train_index]

                    label_encoder = LabelEncoder()
                    yt_train = label_encoder.fit_transform(yt_train)
                    # print(yt_train.shape)

                    onehot_encoder = OneHotEncoder(sparse=False)
                    yt_train = onehot_encoder.fit_transform(yt_train.reshape(-1, 1))
                    # yt_train = onehot_encoder.fit_transform(yt_train)




                    X_test  = Xt[test_index]
                    y_test  = yt[test_index]

                    label_encoder = LabelEncoder()
                    y_test = label_encoder.fit_transform(y_test)

                    onehot_encoder = OneHotEncoder(sparse=False)
                    y_test = onehot_encoder.fit_transform(y_test.reshape(-1, 1))
                    # print(y_test.shape)

                    # y_test = onehot_encoder.fit_transform(y_test)






                elif fold_type == 'baseline':
                    source_target_percnt['src'].append(
                        (len(Xs)) / ((len(Xs) + len(Xt[train_index]))))
                    # source_target_percnt['tgt'].append(
                    #     (len(Xt)) / ((len(Xs)  + len(Xt))))
                    source_target_percnt['y_tgt_1'].append(
                        (np.count_nonzero(yt[train_index] == 1)) / ((len(yt[train_index]))))

                    source_target_percnt['y_src_1'].append(
                        (np.count_nonzero(ys == 1)) / ((len(ys))))
                    # merge'em
                    X_merged_relab = np.concatenate([Xs, Xt[train_index]])
                    y_merged_relab = np.concatenate([ys, yt[train_index]])
                    train_labels   = np.concatenate([np.array(ls), np.array(lt)[train_index]])
                    # shuffle
                    X_train_relab, y_train_relab, train_labels = shuffle(
                        X_merged_relab, y_merged_relab, train_labels, random_state=1)
                    X_train = X_train_relab
                    y_train = y_train_relab
                    X_test  = Xt[test_index]
                    y_test  = yt[test_index]
                elif fold_type == 'augment':
                    source_target_percnt['src'].append(
                        (len(Xs)) / ((len(Xs)  + len(Xt[train_index]))))
                    # source_target_percnt['tgt'].append(
                    #     (len(Xt)) / ((len(Xs)  + len(Xt))))
                    source_target_percnt['y_tgt_1'].append(
                        (np.count_nonzero(yt[train_index] == 1)) / ((len(yt[train_index]))))

                    source_target_percnt['y_src_1'].append(
                        (np.count_nonzero(ys == 1)) / ((len(ys))))
                    # Extend feature space (train)
                    X_merged_aug = self.merge_and_extend_feature_space(Xt[train_index], Xs)
                    # print('######################################')
                    # print(yt[train_index].shape)
                    # print(type(yt[train_index]))
                    # print(ys.shape)
                    # print(type(ys))
                    y_merged_aug = np.concatenate([yt[train_index], ys])
                    train_labels = np.concatenate([np.array(lt)[train_index], np.array(ls)])
                    # Extend feature space (test)
                    X_test_aug = self.merge_and_extend_feature_space(Xt[test_index])
                    X_train = X_merged_aug
                    y_train = y_merged_aug
                    X_test = X_test_aug
                    y_test = yt[test_index]
                elif fold_type == 'multiaugment':
                    # source_target_percnt['src'].append((len(Xs[0])+len(Xs[1]))/((len(Xs[0])+len(Xs[1])+len(Xt[train_index]))))
                    # source_target_percnt['y_tgt_1'].append(
                    #     (np.count_nonzero(yt[train_index] == 1)) / ((len(yt[train_index]))))
                    #
                    # source_target_percnt['y_src_1'].append(
                    #     ((np.count_nonzero(ys[0] == 1))+ (np.count_nonzero(ys[1] == 1)) )/ ((len(ys[0])+len(ys[1]))))
                    # # Extend feature space (train) for multidomain augment
                    # # print(train_index)



                    # print(temp)
                    X_merged_aug,y_merged_aug = self.merge_and_extend_feature_space_multidomain(Xt=Xt[train_index], Xs=Xs,ys=ys,yt=yt[train_index])
                    # print('######################################')
                    # print(yt[train_index].shape)
                    # print(type(yt[train_index]))
                    # print(ys.shape)
                    # print(type(ys))
                    # y_merged_aug = np.concatenate([yt[train_index], ys])
                    # train_labels = np.concatenate([np.array(lt)[train_index], np.array(ls)])
                    # Extend feature space (test)
                    X_test_aug,ytest = self.merge_and_extend_feature_space_multidomain(Xt=Xt[test_index])
                    X_train = X_merged_aug
                    y_train = y_merged_aug
                    X_test = X_test_aug
                    y_test = yt[test_index]
                elif fold_type == 'coral':
                    # ---------coral------------
                    X_train = self.CORAL(Xs, Xt[train_index])
                    y_train = ys
                    X_test  = Xt[test_index]
                    y_test  = yt[test_index]
                    train_labels = ls
                else:
                    raise KeyError('fold_type not one of: %s' % str(self.methods))
                fold = {}
                fold["X_train"] = X_train
                fold["y_train"] = y_train
                fold["X_test"]  = X_test
                fold["y_test"]  = y_test
                if fold_type in ['dann','fmmd','tradaboost']:
                    fold["Xt_train"] = Xt_train
                    fold["yt_train"] = yt_train

                # fold["train_labels"] = train_labels
                data.append(fold)
        # self.save_file(pd.DataFrame(folds),'folds')
        # print(folds)
        print('percent source data in training %.3f'%(np.mean(np.array(source_target_percnt['src']), axis=0)))
        print('percent y=1 in target data in training %.3f'%(np.mean(np.array(source_target_percnt['y_tgt_1']), axis=0)))
        print('percent y=1 in source data in training %.3f'%(np.mean(np.array(source_target_percnt['y_src_1']), axis=0)))


        return data

    def train_all(self,model_name,feat_list,c=None,n_estimator=None, k_range=None):
        for method in self.methods:
            self.myprint("\nTraining: %s" % method)
            self.myprint("---------------------------")
            best_score=self.train_model(model_name,c,method,feat_list=feat_list,n_estimator=n_estimator, k_range=k_range)
            return best_score


    def train_majority_class(self):
        self.myprint("\nTraining Majority Class")
        self.myprint("===========================")
        lt     = self.Xt.index.values
        Xt, yt = self.Xt.values, self.yt.values
        group_kfold = GroupKFold(n_splits=10).split(Xt, yt, groups=lt)

        acc_scores = []
        fms_scores = []

        for train_index, test_index in group_kfold:
            # Data is same as target_only data
            y_train, y_test = yt[train_index], yt[test_index]
            labels          = np.array(lt)[train_index]
            patient_ids     = np.unique(labels)
            maj = []

            # Need to predict most common patient type, not most common interview type
            for patient in patient_ids:
                ind = np.where(labels == patient)[0]
                patient_type = y_train[ind].flatten()[0]
                maj.append(patient_type)

            maj = stats.mode(maj)[0]
            yhat = np.full(y_test.shape, maj, dtype=bool)

            acc_scores.append(accuracy_score(y_test, yhat))     # Save
            fms_scores.append(f1_score(y_test, yhat,average='macro'))           # Save

        # ----- save row -----
        self.results['majority_class'] = {"acc": np.asarray(acc_scores), "fms": np.asarray(fms_scores)}

    def check_arrays(self,X, y, **kwargs):
        """
        Check arrays and reshape 1D array in 2D array
        of shape (-1, 1). Check if the length of X
        match the length of y.
        Parameters
        ----------
        X : numpy array
            Input data.
        y : numpy array
            Output data.

        Returns
        -------
        X, y
        """
        X = check_array(X, ensure_2d=True, allow_nd=True, **kwargs)
        y = check_array(y, ensure_2d=False, allow_nd=True, dtype=None, **kwargs)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Length of X and y mismatch: %i != %i" %
                             (X.shape[0], y.shape[0]))
        return X, y

    # map row to new feature space
    # in accordance with 'frustratingly simple' paper
    def merge_and_extend_feature_space(self, X_target, X_source=None):
        X_target_extended = np.concatenate([X_target, np.zeros(X_target.shape), X_target], axis=1)
        if X_source is None:
            return X_target_extended
        else:
            X_source_extended = np.concatenate([X_source, X_source, np.zeros(X_source.shape)], axis=1)
            return np.concatenate([X_target_extended, X_source_extended])
    #multiple source domain
    def _check_domains(self, domains):
        domains = check_array(domains, ensure_2d=False)
        if len(domains.shape) > 1:
            raise ValueError("`domains` should be 1D array")
        self._domains_dict = {}
        new_domains = np.zeros(len(domains))
        unique = np.unique(domains)
        for dom, i in zip(unique, range(len(unique))):
            new_domains[domains == dom] = i
            self._domains_dict[i] = dom
        return new_domains

    def merge_and_extend_feature_space_multidomain(self, Xt, Xs=None,ys=None,yt=None,domains=None):
        dim = Xt.shape[-1]

        Xt_emb = np.concatenate((np.zeros((len(Xt), dim * self.n_domains_)),
                                 Xt, Xt), axis=-1)
        if Xs is None and ys is None and yt is None: #test data
            return Xt_emb, None

        Xs_emb1 = np.concatenate([Xs[0],Xs[0], np.zeros(Xs[0].shape),Xs[0]], axis=1)#source domain 1
        Xs_emb2 = np.concatenate([Xs[1], Xs[1], np.zeros(Xs[1].shape),Xs[1]], axis=1)#source domain 2
        Xs_emb = np.concatenate((Xs_emb1, Xs_emb2))
        ys_emb = np.concatenate((ys[0], ys[1]))
        X = np.concatenate((Xs_emb, Xt_emb))
        y = np.concatenate((ys_emb, yt))
        return X,y


        if domains is None:
            domains = np.zeros(len(Xs))


        domains = self._check_domains(domains).astype(int)
        domains[1]=1


        self.n_domains_ = int(max(domains) + 1)
        # dim = Xs[0].shape[-1]

        Xt, yt = self.check_arrays(Xt, yt)

        # Xs, ys = self.check_arrays(Xs, ys)
        # print(domains)




        dim = Xs[0].shape[-1]

        for i in range(1,self.n_domains_):
            x=dim * i
            y= (np.sum(domains == i))

            a=np.zeros((np.sum(domains == i), dim * i))
            # b=np.array(Xs)[domains == i]
            b=Xs[i]

            z=dim * (self.n_domains_ - i)
            c= np.zeros((np.sum(domains == i), dim * (self.n_domains_ - i)))
            d= np.array(Xs)[domains == i]
            # Xs_emb_i = np.concatenate(
            #     (np.zeros((np.sum(domains == i), dim * i)),
            #      np.array(Xs)[domains == i],
            #      np.zeros((np.sum(domains == i), dim * (self.n_domains_ - i))),
            #      np.array(Xs)[domains == i]),
            #     axis=-1)

            # Xs_emb_i1 = np.concatenate(
            #     [np.zeros((np.sum(domains == i), dim * i)),
            #      Xs[ i]],axis=-1)
            Xs_emb_i2 = np.concatenate((
                 np.zeros((np.sum(domains == i), dim * (self.n_domains_ - i))),
                 Xs[ i]),
                axis=1)
            if i == 0:
                Xs_emb = Xs_emb_i
                ys_emb = ys[domains == i]
            else:
                Xs_emb = np.concatenate((Xs_emb, Xs_emb_i))
                ys_emb = np.concatenate((ys_emb, ys[domains == i]))



        X = np.concatenate((Xs_emb, Xt_emb))
        y = np.concatenate((ys_emb, yt))

        if self.verbose:
            print("New shape: %s" % str(X.shape))
        return X, y

        # Following CORAL paper -http://arxiv.org/abs/1511.05547

    # Algorithm 1
    def CORAL(self, Ds, Dt):
        EPS = 1 #lambda
        N, D = Ds.shape
        # # Normalize (Here 'whiten' divides by std)
        Ds = whiten(Ds - Ds.mean(axis=0))
        Dt = whiten(Dt - Dt.mean(axis=0))

        Cs = np.cov(Ds, rowvar=False) + EPS * np.eye(D)
        Ct = np.cov(Dt, rowvar=False) + EPS * np.eye(D)

        Ws = self.whiten_matrix(Cs)
        Wcolor = self.color_matrix(Ct)

        Ds = np.dot(Ds, Ws)      # Whiten
        Ds = np.dot(Ds, Wcolor)  # Recolor

        assert not np.isnan(Ds).any()
        return Ds


class BlogCV(DementiaCV):
    """BlogCV is a subclass of DementiaCV which performs a 9-fold cross validation 
    where the test fold has contains posts from blogs not in the training fold.
    """

    def __init__(self, model, X, y, labels, silent=False, random_state=1):
        super(BlogCV, self).__init__(model, X=X, y=y, labels=labels, silent=silent)
        self.methods = ['model', 'majority_class']
        
    def get_data_folds(self, fold_type='default'):

        X, y, labels = self.X, self.y, self.labels

        testset1 = ["creatingmemories", "journeywithdementia"]
        testset2 = ["creatingmemories", "earlyonset"]
        testset3 = ["creatingmemories", "helpparentsagewell"]

        testset4 = ["living-with-alzhiemers", "journeywithdementia"]
        testset5 = ["living-with-alzhiemers", "earlyonset"]
        testset6 = ["living-with-alzhiemers", "helpparentsagewell"]

        testset7 = ["parkblog-silverfox", "journeywithdementia"]
        testset8 = ["parkblog-silverfox", "earlyonset"]
        testset9 = ["parkblog-silverfox", "helpparentsagewell"]

        folds = [testset1, testset2, testset3, testset4, testset5, testset6, testset7, testset8, testset9]
        data = []
        for fold in folds:
            train_index = ~X.index.isin(fold)
            test_index = X.index.isin(fold)
            fold = {}
            fold["X_train"] = X.values[train_index]
            fold["y_train"] = y.values[train_index]
            fold["X_test"]  = X.values[test_index]
            fold["y_test"]  = y.values[test_index]
            fold["train_labels"]  = np.array(labels)[train_index]
            data.append(fold)

        return data
