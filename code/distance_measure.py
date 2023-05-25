import pandas as pd
import os
path="../data/"
def save_file(df, name):
    # df.to_pickle( name + '.pickle')
    df.to_csv( path+name + '.csv')
    print('done')
def get_distance_data():
    df=None
    count=0
    # print(os.getcwd())
    for (root, dirs, file) in os.walk(path):
        print('hi')
        for filename in file:
            if not filename.endswith('.csv')  :
                continue
            if 'ccc_distance' not in filename:
                continue

            name = filename.split('.')[0]

            # files in ccc dataset we r working with, alredy converrted to wav , but not yet processed for features
            # if name.split('.')[0] in df8['filename'].values.tolist() and name not in audio['filename'].values.tolist():
            # if name in include:
            # print(filename)

            print('processing file %s' % (filename))
            if df is None:

              df=pd.read_csv(path+filename)

            else:
                df1=pd.read_csv(path+filename)

                df2 = pd.concat([df, df1], axis=0)
                df = df2
            count+=1
    save_file(df,"ccc_distance_new")
    print(count)
def concat_distance_files(df1,df2,include):
    df3=pd.concat([df1,df2],axis=0)
    df = df3.drop_duplicates(subset='filename', keep="first")
    print(len(df))

    save_file(df, "ccc_distance_combined")


# get_distance_data()
df = pd.read_csv(path + "linguistic_pitt.csv")
df3 = pd.read_csv(path + "demo_adress.csv")

df1 = pd.read_csv(path + "pitt_audio.csv")
df2 = pd.read_csv(path + "pitt_distance.csv")
fv=pd.merge(df,df1,on='filename')
fv=pd.merge(fv,df2,on='filename')
fv=pd.merge(fv,df3,on='filename')
fv = fv[fv['adress_test'] == 1]

print(fv.shape)

# print(df1.shape)
# print(df2.shape)
# file1=set(df2['filename'])

# print(len(list(set(df1['filename']))))
# df2 = pd.read_csv(path + "ccc_distance_rest.csv")
# print(len(list(set(df2['filename']))))
#
# files1=list(set(df['filename'])-set(df2['filename']))
# print(len(files1))
# print(files1)
# files1=list(set(df['filename'])-set(df1['filename']))
# print(len(files1))
# print(files1)

# df = df1.drop_duplicates(subset='filename', keep="first")
# save_file(df1, "adrc_distance")

# concat_distance_files(df1,df2,files1)
# print(files1)


# if not filename.endswith('.csv'):
            #   continue

            # try:
            # print('processing file %s'%(filename))
            # features1=pd.read_csv(filename)
            # features1['filename']=fname
