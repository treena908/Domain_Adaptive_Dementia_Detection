#Data preprocessing
1.  utterance segment the input text (a sample  file name "" is given in 'data' ):
2. . generate parse tree and pos tags for utterances:
    2.1. Download and run the stanford NLP parser (instructions: https://stanfordnlp.github.io/CoreNLP/corenlp-server.html)

    2.2. To generate pos tags and parse tree for each utterance in each input file,
    run command: 'python  get_parse_tree.py'
        - this will generate file name ending with '_tag' which has following feature columns (a sample file is given in 'data'
        folder): filename,pos,token, part_id (participant id),parse_tree




#Generate feature
1. To generate syntactic complexity features (POS phrase structures), semantic (cosine distance between utterances),
 some Vocabulary Richness Features (BrunetIndex etc.) :
run command: 'python  get_pos_phrases.py' (details instructions are commented in the corresponding file)
2. To generate syntactic complexity features (tree height and otehr) and CFG features (based on constituency parse tree and dependency parse tree):
run command: 'python  get_cfg_features.py' (details instructions are commented in the corresponding file)
3. To generate POS (parts of speech) features, vocabulary richness features (MATTR, etc.), NER tags, SUBTL scores,
run Linguistic_Analysis.ipynb

#Running experiments
set parameters in 'domain_adapt.py' file for running experiments (details instructions are commented in the corresponding file), 
this file use procedures from additional files : 
'cross_validators.py', 'data_handler.py', '_tradaboost.py'

run command: 'python domain_adapt.py' 
