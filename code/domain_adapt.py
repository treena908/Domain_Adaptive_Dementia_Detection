import pandas as pd
# from sqlalchemy import types
from cross_validators import DomainAdaptationCV
from data_handler import get_target_source_data,get_adress_test_data
import numpy as np
# from dementia_classifier.settings import DOMAIN_ADAPTATION_RESULTS_PREFIX
# import util
# from util import bar_plot

# --------MySql---------
# from dementia_classifier import db
# cnx = db.get_connection()
# ----------------------

# =================================================================
# ----------------------Save results to sql------------------------
# =================================================================

def save_domain_adapt_results_to_sql():
    ######params
    # lam=[0.01,0.05,0.1,0.5,0.75,1.0]
    lam=[0.05]
    lr=[0.01,0.05,0.1,0.5]
    # krange=[30,25,20,15,10,40,50]
    krange=None #feat selection

    params={'c':[1.0],'lam':lam,'lr':lr}
    # params=[0.3]

    models=['lr']
    bias='c'#c=consistent, e=equal, or None
    ling_sem_audio_features=[1,1,0] # which features to use ling, sem, audio
    # models=['dann']

    adresstest=False #whether to run test on adress test set when source is adress train set
    #if false and target=adress, run cv on adress train set
    src=['ccc']
    tgt=['pitt'] #to run 5 fold  cv on adress train, src=ccc, tgt=adress, adresstest=False,
    # to train on src ccc, tgt adress train, test on adress test, src=ccc, tgt=adress, adresstest=True
    dign='ad' #whether dign is mci or not
    if 'madress' in tgt[0]:
        if not adresstest:
            nfold=8
        else:
            nfold=1


    elif 'adress' in tgt[0] and not adresstest:
        nfold=10

    elif 'adress' in tgt[0] and adresstest:
        nfold=1

    else:
        nfold=5
    #data for training model for DA, source and tgt
    # Xt, yt,lt, Xs, ys,ls=get_target_source_data(source=['pittad'], target=['pittct'], features=ling_sem_audio_features) #for getting feaure valu groupwise
    # # Xt.to_pickle(Xt,'../result/augment/pitt_ct.pickle')

    #
    # Xs.to_csv('../result/augment/'+'pitt_ad'+'.csv')
    # Xt.to_csv('../result/augment/pitt_ct.csv')


    acc=[]
    f1=[]

    combination_src=[(72,28),(57,43),(42,58),(28,72)] #for cons. bias
    combination_tgt=[(90,35),(75,50),(60,65),(35,90)] #for cons. bias
    combination_src_eq=[(48,48)]#for equal bias

    # combination=[(72,28),(57,43),(42,58),(28,72)]

    # combination=[(None,None)]


    # for source in zip(combination):
    for i,source in enumerate(combination_src):


        if dign=='mci':
            Xt_con, yt_con, lt_con, Xt_mci, yt_mci, lt_mci, Xs, ys, ls=get_target_source_data(source=src,target=tgt,dign=dign)
        else:
            Xt, yt,lt, Xs, ys,ls = get_target_source_data(source=src,target=tgt,features=ling_sem_audio_features,dign=dign,k_range=krange,src_pos=source,tgt_pos=combination_tgt[i],bias=bias)
            # dataset_ys = tf.data.Dataset.zip(tuple(

                    # X[domains == dom]).repeat(repeats[dom])
                    # for dom in range(self.n_sources_))



        # print(len(Xs.columns.tolist()))
        # print(len(Xt.columns.tolist()))
        # print(set(Xs.columns.tolist()) - set(Xt.columns.tolist()))
        # print(set(Xt.columns.tolist()) - set(Xs.columns.tolist()))
        # print(Xs.columns.tolist())
        if type(Xs) is list:
            print(Xs[0].shape)
            print(Xs[1].shape)
        else:
            print(Xs.shape)


        print(Xt.columns.tolist())
        print(Xt.shape)





        if not adresstest:
            if dign=='mci':
                da = DomainAdaptationCV(Xt=Xt_mci, yt=yt_mci, lt=lt_mci, Xs=Xs, ys=ys, ls=ls,Xt_con=Xt_con, yt_con=yt_con, lt_con=lt_con, source=src, target=tgt,nfold=nfold)
            else:
                da = DomainAdaptationCV( Xt, yt, lt, Xs, ys, ls,features=Xt.columns.tolist(),source=src,target=tgt,nfold=nfold)
        else:
            # Xt_test, yt_test, lt_test=get_adress_test_data(target=['madress_test'],dign=dign,features=ling_sem_audio_features,k_range=krange)

            Xt_test, yt_test, lt_test=get_adress_test_data(target=['adresstest'],features=ling_sem_audio_features,k_range=krange)
            # print(Xs.columns.tolist())
            # print(Xt_test.shape)


            da = DomainAdaptationCV( Xt, yt, lt, Xs, ys, ls,features=Xt.columns.tolist(),source=src,target=tgt,Xt_test=Xt_test, yt_test=yt_test, lt_test=lt_test,nfold=nfold)


        for model in models:
            print('model %s'%model)
            if 'dann' in model:
                parameters=params['lam']

            else:
                parameters=params['c']



            for param in parameters:
                for n_esmator in [45]:
                    for model in models:
                        print('Running %s for param C %f %d' % (model,param,n_esmator))
                    # print('Running %s for param C %f %s' % (model,param, str()))


            # da = DomainAdaptationCV(models, Xt, yt,lt, Xs, ys,ls)
            #         da.train_all(model,c=param,n_estimator=n_esmator,feat_list=ling_sem_audio_features)
                        _,best_score=da.train_all(model,c=param,n_estimator=n_esmator,feat_list=ling_sem_audio_features,k_range=krange)
                        #print(best_score)
                        acc.append(best_score['acc'])
                        f1.append(best_score['fms'])
    # print(acc)
    # print(f1)

    print('mean acc %f'%(np.mean(np.array(acc)))) #for bias exp.
    print('mean f1 %f'%(np.mean(np.array(f1))))#for bias exp

    print("acc_stdev %f"% np.nanstd(acc, axis=0))
    print("fms_stdev %f"% np.nanstd(f1, axis=0))
    print('acc')
    print(acc)
    print('f1')
    print(f1)





    # save_domain_adapt_to_sql_helper(da, model)

    # da.train_majority_class()
    # save_majority_class(da)


# def save_majority_class(da):
#     for metric in ['fms', 'acc']:
#         results = da.results['majority_class'][metric]
#         df = pd.DataFrame(results)
#         name = "cv_majority_class_%s" % metric
#         df.to_sql(name, cnx, if_exists='replace')
#
#
# def save_domain_adapt_to_sql_helper(da, model_name, if_exists='replace'):
#     dfs = []
#     name = "%s_%s" % (DOMAIN_ADAPTATION_RESULTS_PREFIX, model_name)
#     for method in da.methods:
#         k_range = da.best_k[method]['k_range']
#         # for metric in ['roc', 'fms', 'acc']:
#         for metric in ['fms', 'acc']:
#             if metric in da.results[method].keys():
#                 results = da.results[method][metric]
#                 df = pd.DataFrame(results, columns=k_range)
#                 df['metric'] = metric.decode('utf-8', 'ignore')
#                 df['method'] = method.decode('utf-8', 'ignore')
#                 dfs += [df]
#
#     df = pd.concat(dfs, axis=0, ignore_index=True)
#     typedict = {col_name: types.Float(precision=5, asdecimal=True) for col_name in df}
#     typedict['metric'] = types.NVARCHAR(length=255)
#     typedict['method'] = types.NVARCHAR(length=255)
#     df.to_sql(name, cnx, if_exists=if_exists, dtype=typedict)
#
#
# # =================================================================
# # ----------------------Get results from sql-----------------------
# # =================================================================
#
# # Returns 10 folds for best k (k= number of features)
# def get_da_results(classifier_name, domain_adapt_method, metric):
#     name = "results_domain_adaptation_%s" % classifier_name
#     table = pd.read_sql_table(name, cnx, index_col='index')
#     table = table[(table.metric == metric) & (table.method == domain_adapt_method)].dropna(axis=1)
#     df = util.get_max_fold(table)
#     df['model'] = classifier_name
#     df['method'] = domain_adapt_method
#     return df
#
#
# # =================================================================
# # ------------------------- Make plots ----------------------------
# # =================================================================
#
# def domain_adaptation_plot_helper(classifiers, metric='acc'):
#     print("Plotting domain_adaptation, classifiers: %s" % classifiers)
#     METHODS = ['baseline','augment', 'coral','dann']
#     dfs = []
#     for method in METHODS:
#         for classifier in classifiers:
#             df = get_da_results(classifier, method, metric)
#             util.print_ci_from_df(df['folds'], method, classifier)
#             dfs.append(df)
#
#     dfs = pd.concat(dfs)
#
#     if metric == 'acc':
#         y_label = "Accuracy"
#     elif metric == 'fms':
#         y_label = "F-Measure"
#     else:
#         y_label = "AUC"
#
#     plot_specs = {
#         'x_col': 'method',
#         'y_col': 'folds',
#         'hue_col': 'model',
#         'x_label': 'Model',
#         'figsize': (10, 8),
#         'font_scale': 1.2,
#         'fontsize': 20,
#         'y_label': y_label,
#         'y_lim': (0, 1)
#     }
#
#     figname = 'domain_adapt_plot_%s_%s.pdf' % (metric, classifiers[1])
#     bar_plot(dfs, figname, **plot_specs)
#
#
# def good_classifiers_plot(metric='acc'):
#     domain_adaptation_plot_helper(models.CLASSIFIER_SET_1, metric)
#
#
# def bad_classifiers_plot(metric='acc'):
#     domain_adaptation_plot_helper(models.CLASSIFIER_SET_2, metric)

save_domain_adapt_results_to_sql()
