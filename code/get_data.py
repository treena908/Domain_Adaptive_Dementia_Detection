# encoding: utf-8
import os
import sys
import re
import requests
import pandas as pd
import nltk
from collections import defaultdict
import nltk.data
import xml.etree.ElementTree as ET
from pycorenlp import StanfordCoreNLP

from ast import literal_eval

from pos_syntactic import get_all_tree_features
PARSER_MAX_LENGTH = 50
try:
    import cPickle as pickle
except:
    import pickle
path = "../data/"  # Update if needed

# takes a long string and cleans it up and converts it into a vector to be extracted
# NOTE: Significant preprocessing was done by sed - make sure to run this script on preprocessed text

# Data structure
# data = {
# 		"id1":[	{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]}, <--single utterance
# 		 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		 	],													  <--List of all utterances made during interview
# 		"id2":[	{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		 	],
# 		...
# }

def get_stanford_parse(port=9000):
    # raw = sentence['raw']
    # We want to iterate through k lines of the file and segment those lines as a session
    # pattern = '[a-zA-Z]*=\\s'
    # re.sub(pattern, '', raw)

    try:
        nlp = StanfordCoreNLP('http://localhost:9000')

                          # '/?properties={\"annotators\":\"parse\",\"outputFormat\":\"json\"}', data=sentence)
    except requests.exceptions.ConnectionError as e:
        print("We received the following error in get_data.get_stanford_parse():")
        print(e)

        sys.exit(1)
    return nlp

#for each  utt of certain length, generate parse tree
def _processUtterance(uttr):

    nlp = StanfordCoreNLP('http://localhost:9000')
    ct_parse=[]#constituency parse tree
    dp_parse=[]#dependency parse tree

    tokens = nltk.word_tokenize(uttr)  # Tokenize
    tagged_words = nltk.pos_tag(tokens)  # Tag
    # Get the frequency of every type
    pos_freq = defaultdict(int)
    for word, wordtype in tagged_words:
        pos_freq[wordtype] += 1
    return tagged_words,pos_freq,tokens

    result=split_string_by_words(uttr, 50)
    for u in result:

        if u is not "":


            stan_parse = nlp.annotate(u, properties={
                'annotators': 'parse',
                'outputFormat': 'json'
            })
            # print(stan_parse['sentences'][0]["parse"])
            if 'sentences' in stan_parse and len(stan_parse['sentences'])>0:
                # print(stan_parse['sentences'][0])
                ct_parse.append(stan_parse['sentences'][0]["parse"])
                dp_parse.append(stan_parse['sentences'][0]["basicDependencies"])
                # return stan_parse['sentences'][0]["parse"],stan_parse['sentences'][0]["basicDependencies"],tagged_words,pos_freq,tokens
            # return None,None,tagged_words,pos_freq,tokens


    return ct_parse,dp_parse,tagged_words,pos_freq,tokens

def split_string_by_words(sen, n):
    result=[]
    tokens = sen.split()
    for i in range(0,int(len(tokens)/n)+1):
        if (i+1)*n<len(tokens):
            end=(i+1)*n
        else:
            end=len(tokens)
        result.append(" ".join(tokens[(i)*n:end]))

     # result=[" ".join(tokens[(i) * n:(i + 1) * n]) for i in range(int(len(tokens) / n) + 1)]
    return result

def save_file(df,name):


    df.to_pickle(path+name+'.pickle')
    df.to_csv(path+name+'.csv')
    print('done')
# Extract data from dbank directory
#in the filepath, for each file, single utt. are saved. for each utt. parse trees are generated
def _parse_corpus(filepath,utt_col,file_col,db):
    run=False
    if 'csv' in filepath and 'pitt' in db:
        df=pd.read_csv(filepath)
        df[utt_col] = df[utt_col].apply(literal_eval)

    elif 'pickle' in filepath and 'ccc' in db:
        df=pd.read_pickle(filepath)
    elif 'csv' in filepath and 'adrc' in db:
        df = pd.read_csv(filepath)
        df[utt_col] = df[utt_col].apply(literal_eval)
    result = {'filename': [],'pos':[],'token':[],'pos_freq':[]}


    # result = {'filename': [], 'parse_tree': [],'basic_dependencies': [],'pos':[],'token':[],'pos_freq':[]}

    log = open(path+'parse_'+db, 'a+')  # Not using 'with' just to simplify the example REPL session
    parse_tree=[]
    #for each file
    for idx,row in df.iterrows():
        # if run==False or 'Tichenor_Lawre' not in row[file_col]:
        #     continue
        # run=True
        log.write('filename: '+str(row[file_col])+"\n")
        parse = []
        dependency=[]
        tagged_words=[]
        pos_freqs=[]
        tokens=[]



        #for each utt. in the file
        utt_count=0
        for uttr in row[utt_col]:
            utt_count+=1

            try:
                # ct,dep,tagged_word,pos_freq,token=_processUtterance(uttr)
                tagged_word,pos_freq,token=_processUtterance(uttr)

                if tagged_word is not None:
                    # parse.append(ct)
                    # dependency.append(dep)
                    tagged_words.append(tagged_word)  # pos tagged words
                    pos_freqs.append(pos_freq)  # freq of pos type apperead
                    tokens.append(token)  # tokens
                    # print(tokens)
                    # print(tagged_words)
                    # print(pos_freq)
                    # print(tokens)

                    log.write(uttr)
                    log.write("\n")
                    # print(p)
                    log.write('parse')


                    # log.write(str(ct))
                    log.write("\n")
                    log.write('dependency')

                    # log.write(str(dep))
                    log.write("\n")
                    print('file %s done' % (row[file_col]))

                    # get parse tree features  from transcripts
                else:
                    log.write('filename: ' + str(row[file_col]) + "\n")

                    log.write('no parse tree for uttr:' +uttr)

                    print('no parse tree for uttr:' +uttr)
                # if tagged_word is not None and pos_freq is not None and token is not None:
                #     tagged_words.append(tagged_word) #pos tagged words
                #     pos_freqs.append(pos_freq) # freq of pos type apperead
                #     tokens.append(token) # tokens

            except Exception as e:
                print(e)


                print('exception occured in file %s line %d utt. %s'%(str(row[file_col]),utt_count,uttr))
                log.write(str(e))
                log.write('exception occured in file '+str(row[file_col])+ ' line: ' +str(utt_count)+ ' utt. '+ uttr)

        result['filename'].append(row[file_col])
        # result['part_id'].append(row['id'])

        # result['parse_tree'].append(parse)
        # result['basic_dependencies'].append(dependency)
        result['pos'].append(tagged_words)
        result['token'].append(tokens)
        result['pos_freq'].append(pos_freqs)










        # parse_tree.append(parse) #parse tree for all utt. of a transcript
    #
    # print(result['token'])
    # print(result['pos_freq'])

    save_file(pd.DataFrame(result),db+'_tags')

    # df['parse_tree']=parse_tree
    # save_file(df,db+'_parse')








if __name__ == '__main__':
    # df1=pd.read_pickle(path+'participant_all_ccc_transcript.pickle')
    df2=pd.read_pickle(path+'adrc_parse_tree2.pickle')
    df3=pd.read_pickle(path+'adrc_tags.pickle')
    df3['parse_tree']=df2['parse_tree']
    df3['basic_dependencies']=df2['basic_dependencies']



    # df1['file']=df.index.values
    save_file(df3,'adrc_tags')


    # df=pd.read_csv(path+'utterance_data_adrc.csv')

    # print(df.index.values)
    # save_file(df,'ccc_cfg2')

    # df1['token']=df1['token'].apply
    # print(df1['parse_tree'])

    # print(df['id'].head(20))
    # print(df.iloc)

    # print(df.iloc[1]['single_utterance'][0])
    # # print(type(df.iloc[1]['single_utterance'][0]))
    # df1=pd.read_csv('data/ccc_parse.csv')
    # filename=[num for num in df1.filename.values.tolist() if num.isdigit()]
    # # print(df1.columns.tolist())
    #
    # print(len(filename))
    #
    # df2=pd.read_pickle('data/ccc_parse.pickle')
    # # print(df2.head(5))
    # print(len(df2))
    # df2['filename']=filename
    # print(df2.filenamevalues.tolist())
    #
    # print(df2.columns.tolist())
    # df2.to_pickle('data/ccc_parse.pickle')

    # df=pd.read_pickle('data/adrc_parse_tree.pickle')
    # print(df['token'])
    # print(df['pos_freq'])
    # print(df['pos'])
    # _parse_corpus(path+'utterance_data_adrc.csv','single_utterance','file','adrc')
