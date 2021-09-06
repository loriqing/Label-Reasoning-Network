# easy format
# {
#     "id": int
#     "sentence": str
#     "context": str
#     "mention": str
#     "type": [str]
# }
import json, os
import random

mask = '[MASK]'
e_b = '<e>'
e_e = '</e>'
dir_path = "open_type/distant_supervision"
out_root = "fine-grained-entity-typing"
if not os.path.exists(out_root):
    os.makedirs(out_root)
# data_names = ['train', 'dev', 'test']
# data_names = ['train_m']


def split_cfet(data_dir, file_name, suffix, data_names):
    data = open(os.path.join(data_dir, file_name), 'r').read().strip().split('\n')
    index = list(range(len(data)))
    random.shuffle(index)
    print(len(index))
    split = len(index) // 3
    idx_dic = {'train': [0, split], 'dev': [split, split*2], 'test': [split*2, -1]}
    print(idx_dic)
    for name in data_names:
        out_f = open(os.path.join(data_dir, name + suffix), 'w')
        [b, e] = idx_dic[name]
        for idx in index[b:e]:
            out_f.write(data[idx] + '\n')
        out_f.close()


def calculate_label_freq(dir_path, suffix, diff, data_names):
    label_freq_dic = {}
    global out_root
    for name in data_names:
        path = os.path.join(dir_path, name + suffix)
        f = open(path, 'r')
        out_dir = os.path.join(out_root, diff)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for line in f.readlines():
            data = json.loads(line)
            for l in data['y_str']:
                if l not in label_freq_dic:
                    label_freq_dic[l] = 1
                else:
                    label_freq_dic[l] += 1
    return label_freq_dic


def change_open_type(dir_path, suffix, diff, data_names):
    label_freq_dic = calculate_label_freq(dir_path, suffix, diff, data_names)
    label_list = open('open_type/release/ontology/types.txt', 'r').read().strip().split('\n')
    labels = label_list[:9]
    fine_labels = label_list[9:130]
    ultra_fine_labels = label_list[130:]

    global out_root
    for name in data_names:
        path = os.path.join(dir_path, name + suffix)
        f = open(path, 'r')
        out_dir = os.path.join(out_root, diff)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_f = open(os.path.join(out_dir, name + '.json'), 'w')
        id = 0
        for line in f.readlines():
            data = json.loads(line)
            easy_data = {}
            easy_data['id'] = id
            easy_data["left_sentence"] = ' '.join(data['left_context_token'])
            easy_data["right_sentence"] = ' '.join(data['right_context_token'])
            easy_data['sentence'] = ' '.join(data['left_context_token'] + data['mention_span'].strip().split() + data['right_context_token'])
            easy_data['sentence_entity'] = ' '.join(data['left_context_token'] + [e_b] + data['mention_span'].strip().split() + [e_e] + data['right_context_token'])
            easy_data['context'] = ' '.join(data['left_context_token'] + [mask] + data['right_context_token'])
            pos = ['l'] * len(data['left_context_token']) + ['m'] + ['r'] * len(data['right_context_token'])
            pos_entity = ['l'] * len(data['left_context_token']) + ['m'] * len([e_b] + data['mention_span'].strip().split() + [e_e]) + ['r'] * len(data['right_context_token'])
            easy_data['pos'] = ' '.join(pos)
            easy_data['pos_entity'] = ' '.join(pos_entity)
            easy_data['entity'] = data['mention_span']
            easy_data['mention'] = data['mention_span']
            easy_data['labels'] = []
            easy_data['fine_labels'] = []
            easy_data['ultra_fine_labels'] = []
            easy_data["seq_labels"] = data['y_str']
            for l in data['y_str']:
                if l in labels:
                    easy_data['labels'].append(l)
                elif l in fine_labels:
                    easy_data['fine_labels'].append(l)
                elif l in ultra_fine_labels:
                    easy_data['ultra_fine_labels'].append(l)
                else:
                    print(l, ', no such label')

            for label_name in ['labels', 'fine_labels', 'ultra_fine_labels', 'seq_labels']:
                label_dic = {label: label_freq_dic[label] for label in easy_data[label_name]}
                label_pair = sorted(label_dic.items(), key=lambda d: d[1], reverse=True)
                seq_labels = [pair[0] for pair in label_pair]
                easy_data[label_name] = seq_labels

            out_f.write(json.dumps(easy_data) + '\n')
            id += 1
        out_f.close()
        print(path + ' ok!')


def change_ontonotes(dir_path, suffix, diff, data_names):
    label_freq_dic = calculate_label_freq(dir_path, suffix, diff, data_names)
    labels = open('open_type/release/ontology/onto_ontology.txt', 'r').read().strip().split('\n')
    name_mapping = {'g_train':'train', 'g_test':'test', 'g_dev':'dev', 'augmented_train': 'train_m'}
    global out_root
    for name in data_names:
        path = os.path.join(dir_path, name + suffix)
        f = open(path, 'r')
        out_dir = os.path.join(out_root, diff)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_f = open(os.path.join(out_dir, name_mapping[name] + '.json'), 'w')
        id = 0
        for line in f.readlines():
            data = json.loads(line)
            easy_data = {}
            easy_data['id'] = id
            easy_data["left_sentence"] = ' '.join(data['left_context_token'])
            easy_data["right_sentence"] = ' '.join(data['right_context_token'])
            easy_data['sentence'] = ' '.join(data['left_context_token'] + data['mention_span'].strip().split() + data['right_context_token'])
            easy_data['sentence_entity'] = ' '.join(data['left_context_token'] + [e_b] + data['mention_span'].strip().split() + [e_e] + data['right_context_token'])
            easy_data['context'] = ' '.join(data['left_context_token'] + [mask] + data['right_context_token'])
            pos = ['l'] * len(data['left_context_token']) + ['m'] + ['r'] * len(data['right_context_token'])
            pos_entity = ['l'] * len(data['left_context_token']) + ['m'] * len([e_b] + data['mention_span'].strip().split() + [e_e]) + ['r'] * len(data['right_context_token'])
            easy_data['pos'] = ' '.join(pos)
            easy_data['pos_entity'] = ' '.join(pos_entity)
            easy_data['entity'] = data['mention_span']
            easy_data['mention'] = data['mention_span']
            easy_data['labels'] = data['y_str']
            for y in easy_data['labels']:
                assert y in labels
            label_dic = {label: label_freq_dic[label] for label in easy_data['labels']}
            label_pair = sorted(label_dic.items(), key=lambda d: d[1], reverse=True)
            seq_labels = [pair[0] for pair in label_pair]
            easy_data['seq_labels'] = seq_labels

            out_f.write(json.dumps(easy_data) + '\n')
            id += 1
        out_f.close()
        print(path + ' ok!')


def change_cfet(dir_path, suffix, diff, data_names):
    global out_root
    for name in data_names:
        path = os.path.join(dir_path, name + suffix)
        f = open(path, 'r')
        out_dir = os.path.join(out_root, diff)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_f = open(os.path.join(out_dir, name + '.json'), 'w')
        id = 0
        data_list = []
        for line in f.readlines():
            data = json.loads(line)
            easy_data = {}
            s = data['span']
            easy_data['id'] = id
            easy_data['sentence'] = (''.join(["%s " % i for i in data['sentence']])).strip()
            easy_data['context'] = (''.join(["%s " % i for i in data['sentence'][:s[0]]]) + mask + ' ' +
                                    ''.join(["%s " % i for i in data['sentence'][s[1]:]])).strip()
            pos = ['l'] * len(data['sentence'][:s[0]]) + ['m'] + ['r'] * len(data['sentence'][s[1]:])
            easy_data['pos'] = ' '.join(pos)
            easy_data['entity'] = data['mention']
            easy_data['mention'] = data['mention']
            easy_data['fine_labels'] = data['label_types']
            easy_data['labels'] = data['general_type']
            # easy_data['type'] = list(set(data['label_types'] + data['general_type']))
            data_list.append(easy_data)
            out_f.write(json.dumps(easy_data, ensure_ascii=False) + '\n')
            id += 1
        out_f.close()
        print(path + ' ok!')


def get_label_dic(type_path, diff):
    global out_root
    out_path = os.path.join(out_root, diff, 'label_dic.json')
    data = open(type_path, 'r').read().strip().split('\n')
    label_dic = {}
    for idx, label in enumerate(data):
        assert label not in label_dic
        label_dic[label] = idx
    json.dump(label_dic, open(out_path, 'w'))


# change_open_type('open_type/release/crowd', '.json', 'open_type', data_names=['train_m'])  # 'train', 'dev', 'test', 'train_m'
# change_open_type('open_type/release/crowd', '.json', 'open_type', data_names=['el_train', 'el_dev', 'headword_train', 'headword_dev'])
# change_open_type('open_type/release/distant_supervision', '.json', 'open_type', data_names=['el_train', 'el_dev', 'headword_train', 'headword_dev'])
# change_ontonotes('open_type/release/ontonotes', '.json', 'ontonotes', data_names=['g_train', 'g_dev', 'g_test', 'augmented_train'])

def merge_data(input_dir):
    train_data = open(os.path.join(input_dir, 'train.json'), 'r').read().strip().split('\n')
    train_m_data = open(os.path.join(input_dir, 'train_m.json'), 'r').read().strip().split('\n')
    out_f = open(os.path.join(input_dir, 'train_merge.json'), 'w')
    for line in train_data:
        out_f.write(line + '\n')
    for line in train_m_data:
        out_f.write(line + '\n')
    out_f.close()

merge_data("data/open_type")
merge_data("data/ontonotes")
