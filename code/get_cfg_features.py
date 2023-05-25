
import pandas as pd

from pos_syntactic import get_aux_dependency_dependent, get_VP_2_AUX
Total_nodes = 0
path='../data/'

class tree_node():

    def __init__(self, key, phrase=None):
        self.key = key
        self.phrase = phrase
        self.children = []

    def addChild(self, node):
        self.children.append(node)


def build_tree(parse_tree):
    node_stack = []
    build_node = False
    node_type = None
    phrase = None
    root_node = None
    encounter_leaf = False
    for ch in parse_tree:
        # If we encounter a ( character, start building a node
        if ch == '(':
            if node_type:
                # Finished building node
                node_type = node_type.strip()
                new_node = tree_node(node_type)
                node_stack.append(new_node)
            # Reset
            encounter_leaf = False
            build_node = True
            node_type = None
            phrase = None
            continue
        if ch == ')':
            # pop from the stack and add it to the children for the node before it
            if phrase:
                new_node = tree_node(node_type, phrase)
                node_stack.append(new_node)
            popped_node = node_stack.pop()
            if len(node_stack) > 0:
                parent = node_stack[-1]
                parent.addChild(popped_node)
            else:
                root_node = popped_node
            phrase = None
            node_type = None
            build_node = False
            encounter_leaf = False
            continue
        if encounter_leaf and build_node:
            if not phrase:
                phrase = ''
            phrase += ch
            continue
        if ch.isspace():
            encounter_leaf = True
            continue
        if build_node:
            if not node_type:
                node_type = ''
            node_type = node_type + ch
            continue
    return root_node


def get_height_of_tree(tree_node):
    depths = [0]
    for children in tree_node.children:
        depths.append(get_height_of_tree(children))
    depths = map(lambda x: x + 1, depths)
    return max(depths)


def get_count_of_parent_child(child_type, parent_type, tree_node, prev_type=None):
    curr_type = tree_node.key
    count = 0
    if prev_type == parent_type and curr_type == child_type:
        count = 1
    for child in tree_node.children:
        count += get_count_of_parent_child(child_type, parent_type, child, curr_type)
    return count


def get_count_of_parent_children(child_types, parent_type, tree_node):
    count = 0
    curr_type = tree_node.key
    if not len(tree_node.children):
        return count
    curr_children = [child.key for child in tree_node.children]
    if curr_type == parent_type and set(child_types).issubset(set(curr_children)):
        count = 1
    for child in tree_node.children:
        count += get_count_of_parent_children(child_types, parent_type, child)
    return count


def get_NP_2_PRP(tree_node):
    return get_count_of_parent_child('PRP', 'NP', tree_node)


def get_ADVP_2_RB(tree_node):
    return get_count_of_parent_child('RP', 'ADVP', tree_node)


def get_NP_2_DTNN(tree_node):
    return get_count_of_parent_children(['DT', 'NN'], 'NP', tree_node)


def get_VP_2_VBG(tree_node):
    return get_count_of_parent_child('VBG', 'VP', tree_node)


def get_VP_2_VBGPP(tree_node):
    return get_count_of_parent_child(['VBG', 'PP'], 'VP', tree_node)


def get_VP_2_AUXVP(tree_node, dependents):
    return get_VP_to_aux_and_more(tree_node, "VP", dependents)


def get_VP_2_AUXADJP(tree_node, dependents):
    return get_VP_to_aux_and_more(tree_node, "ADJP", dependents)


def get_VP_to_aux_and_more(tree_node, sibling_to_check, dependents):
    count = 0
    if tree_node.key == 'VP':
        # Check children phrase to see if it is inside the aux dependencies
        child_keys = []
        aux_present = False
        for child in tree_node.children:
            if child.phrase:  # If child phrase exists
                if child.phrase in dependents:
                    aux_present = True
            child_keys.append(child.key)
        # Check for condition
        if aux_present:
            child_keys = set(child_keys)
            if sibling_to_check in child_keys:
                count += 1
    for child in tree_node.children:
        count += get_VP_to_aux_and_more(child, sibling_to_check, dependents)

    return count


def get_VP_2_VBDNP(tree_node):
    return get_count_of_parent_child(['VBD', 'NP'], 'VP', tree_node)


def get_INTJ_2_UH(tree_node):
    return get_count_of_parent_child('UH', 'INTJ', tree_node)


def get_ROOT_2_FRAG(tree_node):
    return get_count_of_parent_child('FRAG', 'ROOT', tree_node)


def get_number_of_nodes_in_tree(root_node):
    if root_node is None:
        print('root node none')
        return
    if len(root_node.children) == 0:
        return 1
    count = 1
    for child in root_node.children:
        count += get_number_of_nodes_in_tree(child)
    return count


def get_CFG_counts(root_node, dict):
    if dict.has_key(root_node.key):
        dict[root_node.key] += 1
    if len(root_node.children) > 0:  # Child leaf
        for child in root_node.children:
            dict = get_CFG_counts(child, dict)
    return dict

def get_all_dependency_tree_features(sample):

    features = {

        'VP_to_AUX_VP': 0,

        'VP_to_AUX_ADJP': 0,
        'VP_to_AUX': 0

    }
    # for each transcript
    total_nodes=0
    for utterance in sample:
        # for tree in range(0, len(utterance)):  # for each utt.
        root_node = build_tree(parse_tree)

        # Needs special love
        dependencies = utterance
        features['VP_to_AUX'] += get_VP_2_AUX(dependencies)
        dependents = get_aux_dependency_dependent(dependencies)
        features['VP_to_AUX_VP'] += get_VP_2_AUXVP(root_node, dependents)
        features['VP_to_AUX_ADJP'] += get_VP_2_AUXADJP(root_node, dependents)

    # ================ DIVIDING BY NUMBER OF total nodes in the sample ===============#
    for k, v in features.items():
        features[k] /= float(total_nodes)

    return features

def get_all_constituency_tree_features(sample):
    features = {
        'tree_height': 0,
        'NP_to_PRP': 0,
        'ADVP_to_RB': 0,
        'NP_to_DT_NN': 0,
        'VP_to_VBG': 0,
        'VP_to_VBG_PP': 0,
        'VP_to_VBD_NP': 0,
        'INTJ_to_UH': 0,
        'ROOT_to_FRAG': 0,

        'VP_to_AUX_VP': 0,

        'VP_to_AUX_ADJP': 0,
        'VP_to_AUX': 0

    }

    # for each transcript
    total_nodes=0
    for tree in range(0,len(sample['parse_tree'])):
        # for tree in range(0, len(utterance)):  # for each utt.
        for parse_tree in sample['parse_tree'][tree]:
            print(parse_tree)
            root_node = build_tree(parse_tree)
            total_nodes += get_number_of_nodes_in_tree(root_node)
            features['tree_height'] += get_height_of_tree(root_node)
            features['NP_to_PRP'] += get_NP_2_PRP(root_node)
            features['ADVP_to_RB'] += get_ADVP_2_RB(root_node)
            features['NP_to_DT_NN'] += get_NP_2_DTNN(root_node)
            features['VP_to_VBG'] += get_VP_2_VBG(root_node)
            features['VP_to_VBG_PP'] += get_VP_2_VBGPP(root_node)
            features['VP_to_VBD_NP'] += get_VP_2_VBDNP(root_node)
            features['INTJ_to_UH'] += get_INTJ_2_UH(root_node)
            features['ROOT_to_FRAG'] += get_ROOT_2_FRAG(root_node)
        for dependencies in  sample['basic_dependencies'][tree]:

            # Needs special love
            features['VP_to_AUX'] += get_VP_2_AUX(dependencies)
            dependents = get_aux_dependency_dependent(dependencies)
            features['VP_to_AUX_VP'] += get_VP_2_AUXVP(root_node, dependents)
            features['VP_to_AUX_ADJP'] += get_VP_2_AUXADJP(root_node, dependents)


    # ================ DIVIDING BY NUMBER OF total nodes in the sample ===============#
    for k, v in features.items():
        features[k] /= float(total_nodes)

    return features
def save_file(df,name):
  df.to_pickle(path+name+'.pickle')
  df.to_csv(path+name+'.csv')
  print('done')
dbs=['ccc']

for db in dbs:

    data={}

    df = pd.read_pickle(path+db+'_parse_tree2.pickle')
    if 'ccc' in db:
        df_file=pd.read_pickle(path+'participant_all_ccc_transcript.pickle')
        df['file']=df_file.index.values

    for idx,row in df.iterrows():
        # transid = str(row['filename'])
        if len(row['parse_tree'])==0 or len(row['basic_dependencies'])==0:
            print('parse tree or dep tree missing in file %s'%(str(row['filename'])))
            continue
        if 'ccc' in db:
            transid=str(row['file'])
        else:
            transid = str(row['filename'])

        data[transid]={}

        parse_tree = row
        features = get_all_constituency_tree_features(parse_tree)  # parse tree list for all utt in the interview for the transcript

        for k, v in features.items():
            data[transid][k + " (participant)"] = v  # insert cfg tree values for the particular feature for the transcript
        # features=get_all_dependency_tree_features(row['basic_dependencies'])
        # for k, v in features.items():
        #     data[transid][k + " (participant)"] = v  # insert cfg tree value
    df1=pd.DataFrame(data)
    df1=df1.T
    if 'ccc' in db:
        df1['filename']=df['file']
    else:
        df1['filename']=df.index.values

    save_file(df1,db+'_cfg2')
# df_ccc=pd.read_pickle('data/ccc_cfg.pickle')
# df_ccc=df_ccc.T
# print(df_ccc.head(5))
# save_file(df_ccc,'ccc_cfg')
# df_pitt=pd.read_pickle('data/pitt_cfg.pickle')
# df_pitt=df_pitt.T
# print(df_pitt.head(5))
# save_file(df_ccc,'pitt_cfg')
