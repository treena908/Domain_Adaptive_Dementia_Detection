#this code is adapted from: https://github.com/vmasrani/dementia_classifier/tree/master/dementia_classifier
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
# from dementia_classifier.feature_extraction.feature_sets import feature_set_list
# from dementia_classifier.settings import SQL_DBANK_TEXT_FEATURES, SQL_DBANK_DIAGNOSIS, SQL_DBANK_DEMOGRAPHIC, SQL_DBANK_ACOUSTIC_FEATURES, SQL_DBANK_DISCOURSE_FEATURES
# --------MySql---------
# from dementia_classifier import db
# cnx = db.get_connection()
# ----------------------
# PATH='/content/drive/My Drive/Colab Notebooks/Data/'
PATH='../data/'
PATH_TO_LINGUISTIC_FEATURE='linguistic_'
PATH_TO_AUDIO_FEATURE='_audio'
PATH_TO_SEMANTIC_FEATURE='_distance'

PATH_TO_DEMO_FEATURE='demo_'
ALZHEIMERS     = ["PossibleAD", "ProbableAD"]
CONTROL        = ["Control"]
MCI            = ["MCI"]
NON_ALZHEIMERS = ["MCI", "Memory", "Other", "Vascular"]
CONTROL_BLOGS  = ["earlyonset", "helpparentsagewell", "journeywithdementia"]
DEMENTIA_BLOGS = ["creatingmemories", "living-with-alzhiemers", "parkblog-silverfox"]

# ------------------
# Diagnosis keys
# - Control
# - MCI
# - Memory
# - Other
# - PossibleAD
# - ProbableAD
# - Vascular
# ------------------

# ===================================================================================
# ----------------------------------DementiaBank,CCC,ADRC-------------------------------------
# ===================================================================================
def get_labels(db,fv):
    # print(db)
    if 'adrc' in db or 'madress' in db:
        labels=[label  for label in fv['filename'] ]
        return labels

    if 'pitt' in db or 'adress' in db:

        # print(' get data labels')
        labels = [label[:3]  for label in fv['filename'] ]

        # print(labels)
        return labels
    if 'ccc' in db:
        labels=[label  for label in fv['id'] ]
        return labels

def get_y(db,demo,dign):
    print(db)
    if dign=='mmse' and 'madress' in db:
        df=pd.DataFrame({})
        mmse=[n/30.0 for n in demo['mmse']]
        df['mmse']=mmse
        return df
        return demo['mmse']
    if 'madress' in db:
        return demo['label']


    if 'pitt' in db or 'adress' in db:
        return demo['dem vs. ct']
    if 'ccc' in db:
        # label=[1 if lbl=='AD' else 0 for lbl in demo['label']]
        # demo['dignosis']=label
        return demo['dignosis']
        # label=[1 if lbl=='AD' else 0 for lbl in demo['label']]
        # demo['dignosis']=label
    if 'adrc' in db:
        # label=[1 if lbl=='AD' else 0 for lbl in demo['label']]
        # demo['dignosis']=label
        labels=[]
        # print(demo['ncc'].unique())

        return demo['ncc']

def drop_rows(db,df):
    if 'ccc' in db:
        exclude=[120,365,384,383,157,158,167,378,640,364,107]
    elif 'pitt' in db:
        # exclude=[]
        exclude=['190-2', '062-0', '062-3', '190-1','144-0', '213-2', '134-2', '134-1', '134-3', '213-1']
    # name=df.loc[0,'filename']
    # print(df.shape)
    for item in exclude:
        df.drop(df[df.filename == item].index, inplace=True)
    return df
    # df1 = df.drop_duplicates(subset='filename', keep="first")


    # return df1



def get_features(features,db,feat_name=None,k_range=None):
    if features[0] == 1 and features[1] == 1 and features[2] == 1:

        ling = pd.read_csv(PATH + PATH_TO_LINGUISTIC_FEATURE + db + '.csv')#linguistic
        # ling=ling.iloc[:,get_ling_selected_feat(ling)]
        sem = pd.read_csv(PATH +db+ PATH_TO_SEMANTIC_FEATURE  + '.csv')#semantic
        audio = pd.read_csv(PATH +db+ PATH_TO_AUDIO_FEATURE  + '.csv')#audio
        audio=audio.iloc[:,79:105]
        feat = pd.merge(ling, sem, on=['filename'])

    elif features[0] == 1 and features[1] == 1 and features[2] == 0:

        ling = pd.read_csv(PATH + PATH_TO_LINGUISTIC_FEATURE + db + '.csv')
        # ling=ling.iloc[:,get_ling_selected_feat(ling)]

        sem = pd.read_csv(PATH + db+PATH_TO_SEMANTIC_FEATURE  + '.csv')
        # audio = pd.read_csv(PATH+PATH_TO_AUDIO_FEATURE+db+'.csv')
        feat = pd.merge(ling, sem, on=['filename'])
        # feat=pd.merge(feat, audio,on=['filename'])
    elif features[0] == 1 and features[1] == 0 and features[2] == 1:

        ling = pd.read_csv(PATH + PATH_TO_LINGUISTIC_FEATURE + db + '.csv')
        # ling=ling.iloc[:,get_ling_selected_feat(ling)]

        # sem = pd.read_csv(PATH + PATH_TO_SEMANTIC_FEATURE + db + '.csv')
        audio = pd.read_csv(PATH +db+ PATH_TO_AUDIO_FEATURE  + '.csv')
        audio=audio.iloc[:,79:105]

        feat = pd.merge(ling, audio, on=['filename'])
        # feat=pd.merge(feat, audio,on=['filename'])
    elif features[0] == 0 and features[1] == 1 and features[2] == 1:

        # ling = pd.read_csv(PATH + PATH_TO_LINGUISTIC_FEATURE + db + '.csv')
        sem = pd.read_csv(PATH +db+ PATH_TO_SEMANTIC_FEATURE  + '.csv')
        audio = pd.read_csv(PATH +db+ PATH_TO_AUDIO_FEATURE  + '.csv')
        audio=audio.iloc[:,79:105]

        feat = pd.merge(sem, audio, on=['filename'])
        # feat=pd.merge(feat, audio,on=['filename'])
    elif features[0] == 1 and features[1] == 0 and features[2] == 0:

        feat = pd.read_csv(PATH + PATH_TO_LINGUISTIC_FEATURE + db + '.csv')
        # feat=feat.iloc[:,get_ling_selected_feat(feat)]

        # sem = pd.read_csv(PATH + PATH_TO_SEMANTIC_FEATURE + db + '.csv')
        # audio = pd.read_csv(PATH+PATH_TO_AUDIO_FEATURE+db+'.csv')
        # feat = pd.merge(ling, sem, on=['filename'])
        # feat=pd.merge(feat, audio,on=['filename'])
    elif features[0] == 0 and features[1] == 1 and features[2] == 0:

        # ling = pd.read_csv(PATH + PATH_TO_LINGUISTIC_FEATURE + db + '.csv')
        feat = pd.read_csv(PATH + db+PATH_TO_SEMANTIC_FEATURE  + '.csv')
        # audio = pd.read_csv(PATH+PATH_TO_AUDIO_FEATURE+db+'.csv')
        # feat = pd.merge(ling, sem, on=['filename'])
        # feat=pd.merge(feat, audio,on=['filename'])
    elif features[0] == 0 and features[1] == 0 and features[2] == 1:

        # ling = pd.read_csv(PATH + PATH_TO_LINGUISTIC_FEATURE + db + '.csv')
        # sem = pd.read_csv(PATH + PATH_TO_SEMANTIC_FEATURE + db + '.csv')
        feat = pd.read_csv(PATH + db+PATH_TO_AUDIO_FEATURE  + '.csv')
        # feat=feat.iloc[:,31:105]

        # feat = pd.merge(ling, sem, on=['filename'])
        # feat=pd.merge(feat, audio,on=['filename'])
    if 'ccc' in db or 'pitt' in db:
        feat=drop_rows(db,feat)
    if k_range is not None:
        feat=feat.iloc[:,get_selected_feat('joint',feat,k_range,'ccc')]

    return feat
def select_unique_rows(ad,pos):
    select=[]

    unique= ad.drop_duplicates(subset='unique_label', keep="first")
    columns=unique.columns.tolist()
    cnt=len(unique)
    filenames=unique['filename'].values.tolist()
    if cnt>pos:
        unique=unique[:pos]
    else:
        for idx,row in ad.iterrows():
            dict={}
            if cnt==pos:
                break
            if row['filename'] not in filenames:
                select.append(row['filename'])
                c=np.asarray(ad.loc[idx]).reshape(1,unique.shape[1])
                for i,col in enumerate(columns):
                    dict[col]=c[0][i]

                temp=pd.DataFrame(dict,index=[0])

                unique=pd.concat([unique,temp],axis=0,ignore_index=True)
                cnt+=1
    unique = unique.drop(['unique_label'], axis=1, errors='ignore')

    if unique.shape[0]==pos:
        return unique
    else:
        print('len mis %d'%(len(unique)))




def  create_bias_data(db,fv,pos=48,neg=48): #src
    if 'ccc' in db:
        label='dignosis'
    elif 'pitt' in db:
        label='dem vs. ct'
    ad=fv[fv[label]==1]
    ct=fv[fv[label]==0]
    lbl=get_labels(db,ad)
    ad['unique_label']=lbl
    ad=select_unique_rows(ad,pos)
    lbl = get_labels(db, ct)
    ct['unique_label'] = lbl
    ct=select_unique_rows(ct,neg)
    fv=pd.concat([ad,ct],axis=0)
    return fv



def get_data(dbs, drop_features=True, polynomial_terms=None,dign='ad',demographic=False,opt=[1,0,0],feat_name=None,k_range=None,pos_neg=None,bias=None):

    # Read from sql
    xs=[]
    ys=[]
    ls=[]
    for db  in dbs:
        print(db)
        # print(dign)
        actual_db=db

        if 'madress' in db:
            db=db
        elif 'adress' in db or 'db' in db:
            db='pitt'
        elif 'pittad' in db or 'pittct' in db:
            db='pitt'
        elif 'cccad' in db or 'cccct' in db:
            db='ccc'
        # ling = pd.read_pickle(PATH+PATH_TO_LINGUISTIC_FEATURE+db+'.pickle')
        ling=get_features(opt,db,feat_name,k_range) #get features based on 'features' variable




        db=actual_db

        if 'adresstest' in db:
            db='adress'
        if not 'madress' in db:

            # demo = pd.read_pickle(PATH+PATH_TO_DEMO_FEATURE+db+'.pickle')
            demo = pd.read_csv(PATH+PATH_TO_DEMO_FEATURE+db+'.csv')
            fv = pd.merge(ling, demo, on=['filename'])


        else:



            if 'madress_test' not in db:
                demo = pd.read_csv(PATH+PATH_TO_DEMO_FEATURE+db+'.csv')
                fv = pd.merge(ling, demo, on=['filename'])

            else:
                if  demographic:
                    demo = pd.read_csv(PATH + PATH_TO_DEMO_FEATURE + db + '.csv')
                    fv = pd.merge(ling, demo, on=['filename'])
                else:

                    fv = ling




        # sntx = pd.read_pickle(path+PATH_TO_SYNTX_FEATURE)
        #
        # liwc= pd.read_pickle(path+PATH_TO_LIWC_FEATURE)
        # disf= pd.read_pickle(path+PATH_TO_DISF_FEATURE)
        #
        # pplx= pd.read_pickle(path+PATH_TO_PRPLX_FEATURE)




        # merge linguistic wid demo

        if 'pitt' in actual_db and dign == 'mci':
            return make_com_mci_partition(db,fv)
        if 'pittad' in actual_db:
            fv = fv[fv['dem vs. ct'] == 1]
        elif 'pittct' in actual_db:
            fv = fv[fv['dem vs. ct'] == 0]
        elif 'cccad' in actual_db:
            fv = fv[fv['dignosis'] == 1]
        elif 'cccct' in actual_db:
            fv = fv[fv['dignosis'] == 0]








        #when running test on adresstest data
        if 'adresstest' in actual_db:
            fv = fv[fv['adress_test'] == 1]
            print('ekhane')



        #while CV with adress train set
        elif 'madress' not in actual_db and 'adress' in actual_db:
            fv = fv[fv['adress_train'] == 1]
        if 'adrc' in actual_db:
            fv = fv[fv['ncc'] != 2 ]
        #create equal/consistent bias
        if bias is not None and pos_neg is not None:
            #src=ccc, trg=pitt
            if 'ccc' in db and bias=='e':
                fv=create_bias_data(db,fv,pos=48,neg=48) #src
                print('equal bias')
            elif 'pitt' in db and bias=='e':
                fv=create_bias_data(db,fv,pos=60,neg=60) #target
            elif 'ccc' in db and bias=='c':
                print('consistent bias')
                fv=create_bias_data(db,fv,pos=pos_neg[0],neg=pos_neg[1])#src
            elif 'pitt' in db and bias == 'c':
                fv=create_bias_data(db,fv,pos=pos_neg[0], neg=pos_neg[1])#trgt



        # # add syntax
        # fv = pd.merge(fv, sntx, on=['filename'])
        # # Add liwc
        # fv = pd.merge(fv, liwc, on=['filename'])
        # # Add disf
        # fv = pd.merge(fv, disf, on=['filename'])
        # # Add disf
        # fv = pd.merge(fv, pplx, on=['filename'])

        # Randomize
        if 'madress_train' in actual_db:
            fv['domain']=[0]*len(fv)
        elif 'madress_sample' in actual_db or 'madress_test' in actual_db:
            fv['domain']=[1]*len(fv)


        fv = fv.sample(frac=1, random_state=20)
        # print(fv.head(1))


        # Collect Labels
        labels=get_labels(db,fv)

        # print(labels)
        if 'madress_test' not in db:

            y=get_y(db,fv,dign)
        else:
            y=None
        # print(type(y))
        # print(y)
        # print(y.shape)
        # Clean
        # drop = ['level_0', 'interview', 'diagnosis', 'gender', 'index', 'gender_int']
        if drop_features :

            X = fv.drop(get_drop_features(db,fv,opt,demographic), axis=1, errors='ignore')
        print(X.columns.tolist())

        # print(X.columns.tolist())
        # print(X.columns.tolist())
        # print(len(X.columns.tolist()))
        features=X.columns.tolist()

        X = X.apply(pd.to_numeric, errors='ignore')
        X = X.fillna(0)

        # print(X.head(3))
        # print(X)
        # print(X.index)
        # print(labels)
        X.index = labels
        if 'madress_test' not in db:

            y.index = labels

        if len(dbs)>1:
            xs.append(X)
            ys.append(y)
            ls.append(labels)
        else:
            return X, y, labels
    return xs,ys,ls
    # X = make_polynomial_terms(X, polynomial_terms)
#make data for mci vs control from db=pitt
def make_com_mci_partition(db,fv,drop_features='True'):
    ######################control###############################
    print('ekhane')
    fv_con=fv[fv['mci vs. ct']==0]
    # Randomize
    fv_con = fv_con.sample(frac=1, random_state=20)
    # print(fv.head(1))

    # Collect Labels
    lt_con = get_labels(db, fv_con)

    # print(labels)

    yt_con = fv_con['mci vs. ct']
    print('control')
    print(len(yt_con))

    # Clean
    # drop = ['level_0', 'interview', 'diagnosis', 'gender', 'index', 'gender_int']
    if drop_features:
        Xt_con = fv_con.drop(get_drop_features(db, fv_con), axis=1, errors='ignore')
    # print(X.columns.tolist())
    # print(X.columns.tolist())
    # print(len(X.columns.tolist()))

    Xt_con = Xt_con.apply(pd.to_numeric, errors='ignore')
    Xt_con = Xt_con.fillna(0)

    # print(X.head(3))
    Xt_con.index = lt_con
    yt_con.index = lt_con
    ############################mci########################
    fv_mci=fv[fv['mci vs. ct']==1]
    # Randomize
    fv_mci = fv_mci.sample(frac=1, random_state=20)
    # print(fv.head(1))

    # Collect Labels
    lt_mci = get_labels(db, fv_mci)

    # print(labels)

    yt_mci = fv_mci['mci vs. ct']
    print('mci')
    print(len(yt_mci))

    # Clean
    # drop = ['level_0', 'interview', 'diagnosis', 'gender', 'index', 'gender_int']
    if drop_features:
        Xt_mci = fv_mci.drop(get_drop_features(db, fv_mci), axis=1, errors='ignore')
    # print(X.columns.tolist())
    # print(X.columns.tolist())
    # print(len(X.columns.tolist()))

    Xt_mci = Xt_mci.apply(pd.to_numeric, errors='ignore')
    Xt_mci = Xt_mci.fillna(0)

    Xt_mci.index = lt_mci
    yt_mci.index = lt_mci

    return Xt_con, yt_con,lt_con,Xt_mci,yt_mci,lt_mci


def make_polynomial_terms(data, cols):
    if cols is None:
        return data

    for f1, f2 in itertools.combinations_with_replacement(cols, 2):
        if f1 == f2:
            prefix = 'sqr_'
        else:
            prefix = 'intr_'
        data[prefix + f1 + "_" + f2] = data[f1] * data[f2]

    return data
def get_sem_selected_feat():
    return []
def get_sem_drop(db,selected=False):
    if not selected : #no feature selection
        return ['filename','min_cos_dist']
    else:
        sem = pd.read_csv(PATH + PATH_TO_SEMANTIC_FEATURE + db + '.csv')

        feat=sem.columns.tolist()
        selected_feat=get_sem_selected_feat()
        drop=list(set(feat)-(set(feat)-set(selected_feat))) #return feat excluding selected
        return drop


def get_audio_selected_feat():
    return []
#get selected features on weights
def get_selected_feat(feat,ling,krange,source,pval=1.0):
        # selected=['filename','# tokens (participant)', 'MSTTR (participant)', 'TTR (participant)', '# unique lemmas (participant)', '# named entities (participant)', 'MTLD (participant)', 'NP_to_PRP (participant)', 'tree_height (participant)', '# utterances (participant)', 'Maas (participant)']
        # selected=['filename','# PUNCT (participant)', '# SCONJ (participant)', '# CCONJ (participant)', '# INTJ (participant)',
        #  '# DET (participant)', '# PROPN (participant)', 'Maas (participant)', '# unique lemmas (participant)',
        #  '# ADV (participant)', '# PRON (participant)']
        selected=[]
        count=0
        # if len(feat)>0:
        df=pd.read_csv('../result/augment/'+source+'_'+feat+'_'+'lingsem_weight.csv')
        # else:
        #     df=pd.read_csv('../result/augment/'+source+'_'+'welch_pval.csv')

        for i,row in df.iterrows():
            if feat=='joint':
                selected.append(row['joint_feature'])
            else:
                selected.append(row['s_feature'])
            count+=1
            if count==krange:
                break
        # for i,row in df.iterrows():
        #     if row['pvalue']<0.00057:
        #         if feat=='joint':
        #             selected.append(row['joint_feature'])
        #         else:
        #             selected.append(row['s_feature'])
        #     count+=1
        #     if count==krange:
        #         break
        result=pd.DataFrame({'feat':selected})
        result.to_csv('../result/ranked_feat_'+source+'.csv')
        print("%d feature selected"%len(selected))
        selected.append('filename')


        return list(ling.columns.get_indexer(selected))


def get_audio_drop(db,selected=False):
    if not selected:  # no feature selection
        return ['filename', 'name']
    else:
        sem = pd.read_csv(PATH + PATH_TO_AUDIO_FEATURE + db + '.csv')
        sem=sem[:,79:104] #only duration based features

        feat = sem.columns.tolist()
        selected_feat = get_audio_selected_feat()
        drop = list(set(feat) - (set(feat) - set(selected_feat)))  # return feat excluding selected among dur. feats
        return drop

    return ['min_cos_dist']
def get_drop_features(db,df,features,demo=False):

    drop = [col for col in df.columns.tolist() if 'Unnamed'  in col]
    if features[1] == 1:
        sem_drop = get_sem_drop(db)
        drop.extend(sem_drop)
    if features[2]:
        audio_drop = get_audio_drop(db)
        drop.extend(audio_drop)



    if 'ccc' in db:
        labels=['domid','id','filename','file','dignosis','# TIME (participant)', '# X (participant)', '# PERSON (participant)', '# DATE (participant)', '# ORG (participant)', '# ORDINAL (participant)', '# GPE (participant)']

        drop.extend(labels)
    elif 'adrc' in db:
        labels=['indices','Educ','Race','DX1','id','label','MOCATOTS','MOCA_impairment','Age_at_testing','Sex_Category','APOE_binary','racial','ncc','file','filename','dignosis','# TIME (participant)', '# X (participant)', '# PERSON (participant)', '# DATE (participant)', '# ORG (participant)', '# ORDINAL (participant)', '# GPE (participant)']

        drop.extend(labels)
    elif 'pitt' in db:
        labels=['domid','dem vs. ct','file','probad vs. ct','mci vs. ct','filename','utterance','single_utt','ADVP_to_RB (participant)', 'VP_to_VBD_NP (participant)', 'VP_to_VBG_PP (participant)']
        drop.extend(labels)
    elif 'madress' in db:
        if demo:
            labels = ['indices', 'educ', 'dx', 'mmse', 'label', 'filename', 'gender']
        else:
            labels=['indices','educ','age','dx','mmse','label','filename','gender','sex']

        drop.extend(labels)

    elif 'adress' in db:
        labels = ['domid','dem vs. ct','file', 'probad vs. ct', 'mci vs. ct', 'filename', 'utterance', 'single_utt',
                  'ADVP_to_RB (participant)', 'VP_to_VBD_NP (participant)', 'VP_to_VBG_PP (participant)',
                  'adress_train','label','adress_test']
        drop.extend(labels)


    return drop

def get_adress_test_data(target='adresstest',dign='ad',features=[1,0,0],k_range=None):
    x_s, y_s, l_s = get_data(target, drop_features=True,dign=dign,opt=features,k_range=k_range) #get source data
    print('hi')
    return x_s, y_s, l_s
def get_target_source_data(source=['pitt'],target='ccc',random_state=1,dign='ad',features=[1,0,0],k_range=None,src_pos=None,tgt_pos=None,bias=None):
    # Get data
    # Get data
    if 'ccc' in source and target=='pitt' and dign=='mci':
        print('hi')
        Xs, ys, ls = get_data(source,dign=dign,opt=features )
        Xt_con, yt_con, lt_con,Xt_mci, yt_mci, lt_mci = get_data(target,dign=dign,opt=features)
        # Xt_mci, yt_mci, lt_mci = get_data(target,dign=dign)

        # Split control samples into target/source set (making sure one patient doesn't appear in both t and s)
        # gkf = GroupKFold(n_splits=6).split(X_con, y_con, groups=l_con)
        # source, target = gkf.next()
        # Xt, yt = concat_and_shuffle(X_mci, y_mci, l_mci, X_con.ix[target], y_con.ix[target], np.array(l_con)[target],
        #                             random_state=random_state)
        # Xs, ys = concat_and_shuffle(X_alz, y_alz, l_alz, X_con.ix[source], y_con.ix[source], np.array(l_con)[source],
        #                             random_state=random_state)

        return Xt_con, yt_con,lt_con,Xt_mci,yt_mci,lt_mci, Xs, ys,ls

    x_s, y_s, l_s = get_data(source, drop_features=True,dign=dign,opt=features,feat_name=source,k_range=k_range,pos_neg=src_pos,bias=bias) #get source data
    x_t, y_t, l_t = get_data(target, drop_features=True,dign=dign,opt=features,feat_name=source,k_range=k_range,pos_neg=tgt_pos,

                             bias=bias) #get target data
    # X_mci, y_mci, l_mci = get_data(diagnosis=MCI, drop_features=feature_set)

    # # Split control samples into target/source set (making sure one patient doesn't appear in both t and s)
    # gkf = StratifiedGroupKFold(n_splits=6).split(x_2,y_2,groups=l_target) # split target data into folds
    # gkf = StratifiedGroupKFold(n_splits=6).split(x_2,y_2,groups=l_target) # split target data into folds
    #  = GroupKFold(n_splits=6).split(X_con, y_con, groups=l_con)
    # source, target = gkf.next()
    # Xt, yt = concat_and_shuffle(X_mci, y_mci, l_mci, X_con.ix[target], y_con.ix[target], np.array(l_con)[target], random_state=random_state)
    # Xs, ys = concat_and_shuffle(X_alz, y_alz, l_alz, X_con.ix[source], y_con.ix[source], np.array(l_con)[source], random_state=random_state)

    return x_t, y_t,l_s, x_s, y_s,l_s


def concat_and_shuffle(X1, y1, l1, X2, y2, l2, random_state=1):
    pd.options.mode.chained_assignment = None  # default='warn'
    # Coerce all arguments to dataframes
    X1, X2 = pd.DataFrame(X1), pd.DataFrame(X2)
    y1, y2 = pd.DataFrame(y1), pd.DataFrame(y2)
    l1, l2 = pd.DataFrame(l1), pd.DataFrame(l2)

    X_concat = X1.append(X2, ignore_index=True)
    y_concat = y1.append(y2, ignore_index=True)
    l_concat = l1.append(l2, ignore_index=True)

    X_shuf, y_shuf, l_shuf = shuffle(X_concat, y_concat, l_concat, random_state=random_state)

    X_shuf['labels'] = l_shuf
    y_shuf['labels'] = l_shuf

    X_shuf.set_index('labels', inplace=True)
    y_shuf.set_index('labels', inplace=True)
    # print(y_shuf.values)
    # print(type(y_shuf))
    return X_shuf, np.asarray(y_shuf.values)

# ===================================================================================
# ----------------------------------BlogData-----------------------------------------
# ===================================================================================
#
#
# def get_blog_data(keep_only_good=True, random=20, drop_features=None):
#     # Read from sql
#     cutoff_date = pd.datetime(2017, 4, 4)  # April 4th 2017 was when data was collected for ACL paper
#
#     demblogs = pd.concat([pd.read_sql_table("%s_text_features" % blog, cnx) for blog in DEMENTIA_BLOGS])
#     ctlblogs = pd.concat([pd.read_sql_table("%s_text_features" % blog, cnx) for blog in CONTROL_BLOGS])
#     qual     = pd.read_sql_table("blog_quality", cnx)
#
#     demblogs['dementia'] = True
#     ctlblogs['dementia'] = False
#
#     fv = pd.concat([demblogs, ctlblogs], ignore_index=True)
#
#     # Remove recent posts (e.g. after paper was published)
#     qual['date'] = pd.to_datetime(qual.date)
#     qual = qual[qual.date < cutoff_date]
#
#     # keep only good blog posts
#     if keep_only_good:
#         qual = qual[qual.quality == 'good']
#
#     demblogs = pd.merge(demblogs, qual[['id', 'blog']], on=['id', 'blog'])
#
#     # Randomize
#     fv = fv.sample(frac=1, random_state=random)
#
#     # Get labels
#     labels = fv['blog'].tolist()
#
#     # Split
#     y = fv['dementia'].astype('bool')
#
#     # Clean
#     drop = ['blog', 'dementia', 'id']
#     X = fv.drop(drop, 1, errors='ignore')
#
#     if drop_features:
#         X = X.drop(drop_features, axis=1, errors='ignore')
#
#     X = X.apply(pd.to_numeric, errors='ignore')
#     X.index = labels
#     y.index = labels
#
#     return X, y, labels
#
#
# def get_blog_scatterplot_data(keep_only_good=True, random=20):
#     # Read from sql
#     blogs = pd.read_sql_table("blogs", cnx)
#     qual = pd.read_sql_table("blog_quality", cnx)
#     lengths = pd.read_sql_table("blog_lengths", cnx)
#
#     if keep_only_good:
#         qual = qual[qual.quality == 'good']
#
#     # keep only good
#     data = pd.merge(blogs, qual[['id', 'blog', 'date']], on=['id', 'blog'])
#     data = pd.merge(data, lengths, on=['id', 'blog'])
#
#     # Fix reverse post issue
#     data.id = data.id.str.split('_', expand=True)[1].astype(int)
#     data.id = -data.id
#
#     data.date = pd.to_datetime(data.date)
#
#     return data
#
