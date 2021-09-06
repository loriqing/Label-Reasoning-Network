import json, os, random, time
from collections import Counter
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, BertForMaskedLM, BertTokenizer
import gensim.downloader as api
import stanza
import numpy as np
import string
import shutil

punctuation_string = string.punctuation


data_root = "fine-grained-entity-typing"
topk = 10
bert_thred = 0.1
processors={'tokenize': 'gsd', 'lemma': 'default'}
nlp = stanza.Pipeline('en', processors="tokenize,lemma")
fill_mask_model = pipeline('fill-mask', model='bert-base-uncased', tokenizer='bert-base-uncased', topk=topk)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
word2index = tokenizer.get_vocab()
index2word = {v: k for k, v in word2index.items()}


# dictionary open_type memory
def add_self_memory(label_path, suffix, diff, data_names):
    # for Ultra-Fine dataset
    all_labels = open(label_path, 'r').read().strip().split('\n')
    all_label_dic = {"labels": all_labels[:9], "fine_labels": all_labels[9:115], "ultra_fine_labels": all_labels[115:]}

    types_index = []  # [index]
    types_dic = {}  # {index: token}
    cannot_find = []
    for t in all_labels:
        if t in word2index:
            types_index.append(word2index[t])
            types_dic[word2index[t]] = t
        else:
            cannot_find.append(t)
    print("find %d, not find %d" % (len(types_index), len(cannot_find)))

    for data_name in data_names:
        file = open(os.path.join(data_root, diff, data_name + suffix), 'r').read().strip().split('\n')
        out_file = open(os.path.join(data_root, diff, data_name + '_memory' + suffix), 'w')
        for line in file:
            mention_memory, sentence_memory, bert_memory, bert_scores = [], [], [], []
            line = json.loads(line)
            mention = line['mention'].lower()
            sentence = line['sentence'].lower()
            mention_words = [word.lemma for word in nlp(mention).sentences[0].words]
            sentence_words = [word.lemma for word in nlp(sentence).sentences[0].words]
            for word in mention_words:
                if word in all_labels:
                    mention_memory.append(word)
            for word in sentence_words:
                if word in all_labels:
                    sentence_memory.append(word)
            line['mention_memory'] = mention_memory
            line['sentence_memory'] = sentence_memory

            context = line['context']
            labels = line['seq_labels']
            answers = fill_mask_model(context)
            pred_labels = []
            pred_scores = []
            for pred in answers:
                if pred['token'] in types_index:
                    pred_labels.append(types_dic[pred['token']])
                    pred_scores.append(pred['score'])
            for p_i, label in enumerate(pred_labels):
                if label in all_labels and len(bert_memory) < topk:
                    bert_memory.append(label)
                    bert_scores.append(pred_scores[p_i])
            line['bert_memory'] = bert_memory
            line['bert_scores'] = bert_scores

            out_file.write(json.dumps(line) + '\n')

        out_file.close()


# all open_type memory
def add_open_memory(label_path, suffix, diff, data_names):
    # for Ultra-Fine dataset, for original
    all_labels = open(label_path, 'r').read().strip().split('\n')
    all_label_dic = {"labels": all_labels[:9], "fine_labels": all_labels[9:115], "ultra_fine_labels": all_labels[115:]}

    if not os.path.exists(os.path.join(data_root, diff + '_original')):
        os.makedirs(os.path.join(data_root, diff + '_original'))
    for data_name in data_names:
        file = open(os.path.join(data_root, diff, data_name + suffix), 'r').read().strip().split('\n')
        out_file = open(os.path.join(data_root, diff + '_original', data_name + suffix), 'w')
        for line in file:
            mention_memory, sentence_memory, bert_memory, bert_scores = [], [], [], []
            line = json.loads(line)

            # mention and sentence memory
            mention = line['mention'].lower()
            sentence = line['sentence'].lower()
            mention_words = [word.lemma for word in nlp(mention).sentences[0].words]
            # sentence_words = [word.lemma for word in nlp(sentence).sentences[0].words]
            for word in mention_words:
                mention_memory.append(word)
            # for word in sentence_words:
            #     sentence_memory.append(word)
            line['mention_memory'] = mention_memory
            # line['sentence_memory'] = sentence_memory

            # bert memory
            context = line['context']
            answers = fill_mask_model(context)
            for pred in answers:
                bert_memory.append(index2word[pred['token']])
                bert_scores.append(pred['score'])
            line['bert_memory'] = bert_memory
            line['bert_scores'] = bert_scores

            out_file.write(json.dumps(line) + '\n')

        out_file.close()


# open_memory mapping labels
def mapping_open_memory(label_path, suffix, diff, data_names, function='COS'):  # 'EMD'
    glove_vector = np.load('glove/glove_vector_100.npy')
    glove_vocab = open('glove/glove_vocab_100.txt', 'r').read().strip().split('\n')
    glove_vocab = {w: i for i, w in enumerate(glove_vocab)}
    if function == 'COS':
        word_vectors = api.load("glove-wiki-gigaword-100")
        reverse = True
    if function == 'EMD':
        from scipy.stats import wasserstein_distance
        import numpy as np
        reverse = False

    all_labels = open(label_path, 'r').read().strip().split('\n')
    out_dir = os.path.join(data_root, diff + '_mapping' + str(bert_thred))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for data_name in data_names:
        data = open(os.path.join(data_root, diff, data_name + suffix), 'r').read().strip().split('\n')
        # out_file = open(os.path.join(out_dir, data_name + suffix), 'w')
        for line in data:
            line = json.loads(line)
            label2score = {}
            for memory_name in ['mention_memory', 'bert_memory']:
                for idx, m in enumerate(line[memory_name]):
                    if m in label2score: continue
                    if 'bert' in memory_name:
                        if line['bert_scores'][idx] < bert_thred:
                            continue
                    sims = {}
                    for l in all_labels:
                        try:
                            if function == 'COS':
                                if m not in glove_vocab or l not in glove_vocab:
                                    continue
                                a = glove_vector[glove_vocab[m]]
                                b = glove_vector[glove_vocab[l]]
                                sim = np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                            if function == 'EMD':
                                if m not in glove_vocab or l not in glove_vocab:
                                    # print(m, l)
                                    continue
                                idx1 = glove_vocab[m]
                                idx2 = glove_vocab[l]
                                if '_' in m or '_' in l:
                                    y = 1
                                sim = wasserstein_distance(glove_vector[idx1], glove_vector[idx2])
                        except:
                            continue
                        sims[l] = sim
                    sort_sims = sorted(sims.items(), key=lambda kv: (kv[1], kv[0]), reverse=reverse)
                    # not in memory
                    if len(sort_sims) == 0:
                        continue
                    label2score[m] = sort_sims[0]
            sort_label2score = sorted(label2score.items(), key=lambda kv: kv[1][1], reverse=reverse)
            # maybe duplicate
            memory_list = [pair[1][0] for pair in sort_label2score]
            memory_scores = [float(pair[1][1]) for pair in sort_label2score]

            line.pop('mention_memory')
            line.pop('sentence_memory')
            line.pop('bert_memory')
            line.pop('bert_scores')
            line['memory'] = memory_list
            line['memory_scores'] = memory_scores
            # print(line['memory_scores'])
            # out_file.write(json.dumps(line) + '\n')

        # out_file.close()
        print(data_name + ' process ok !')


# all open_type memory bert -> lemma
def add_open_memory_lemma(input_dir, suffix, diff, data_names):

    # for Ultra-Fine dataset, original convert to lemma
    if not os.path.exists(os.path.join(data_root, diff + '_original_lemma')):
        os.makedirs(os.path.join(data_root, diff + '_original_lemma'))
    for data_name in data_names:
        file = open(os.path.join(input_dir, data_name + suffix), 'r').read().strip().split('\n')
        out_file = open(os.path.join(data_root, diff + '_original_lemma', data_name + suffix), 'w')
        for line in file:
            bert_memory, bert_scores = [], []
            line = json.loads(line)
            # bert memory convert to lemma
            for word, score in zip(line['bert_memory'], line['bert_scores']):
                # if '##' in word: continue
                # bert_words = wordnet_lemmatizer.lemmatize(word)
                bert_words = [word.lemma for word in nlp(word).sentences[0].words]
                if len(bert_words) > 1:
                    continue
                    print(word, bert_words)
                bert_memory.extend(bert_words)
                bert_scores.extend([score] * len(bert_words))
            assert len(bert_memory) == len(bert_scores)

            line['bert_memory'] = bert_memory
            line['bert_scores'] = bert_scores

            out_file.write(json.dumps(line) + '\n')

        out_file.close()
        print(data_name + ' ok!')


# new attributes
def synthetic_open_memory(label_path, suffix, diff, train_name):
    all_labels = open(label_path, 'r').read().strip().split('\n')
    all_label_dic = {"labels": all_labels[:9], "fine_labels": all_labels[9:115], "ultra_fine_labels": all_labels[115:]}
    train_data = open(os.path.join(data_root, diff, train_name + suffix), 'r').read().strip().split('\n')
    few_shot_vocab = open('few_shot_vocab_2.txt', 'r').read().strip().split('\n')
    out_file = open(os.path.join(data_root, diff, train_name + '_syn' + suffix), 'w')
    for line in train_data:
        line = json.loads(line)
        synthetic_memory = []
        # rule: opt:neg = 1:1
        for layer_name in all_label_dic.keys():
            if layer_name == 'labels' and random.random() > 0.6:
                continue
            if layer_name == 'fine_labels' and random.random() > 0.8:
                continue
            if len(line[layer_name]) > 0:
                intersection = list(set(line[layer_name]).intersection(set(few_shot_vocab)))
                if len(intersection) > 0:
                    synthetic_memory.extend(intersection)
                    rand_index = all_label_dic[layer_name].copy()
                    rand_index = list(set(rand_index).difference(set(intersection)))
                    rand_index = random.sample(rand_index, len(intersection))
                    synthetic_memory.extend(rand_index)
                else:
                    rand_index = line[layer_name].copy()
                    rand_index = random.sample(rand_index, 1)
                    synthetic_memory.extend(rand_index)
                    rand_index = all_label_dic[layer_name].copy()
                    rand_index = list(set(rand_index).difference(set(line[layer_name])))
                    rand_index = random.sample(rand_index, 1)
                    synthetic_memory.extend(rand_index)
        if len(synthetic_memory) == 0:
            seq_labels = line['seq_labels'].copy()
            rand_index = random.sample(seq_labels, min(2, len(seq_labels)))
            synthetic_memory.extend(rand_index)
        random.shuffle(synthetic_memory)
        line['synthetic_memory'] = synthetic_memory
        out_file.write(json.dumps(line) + '\n')
    out_file.close()


# merge lemma and memory remove stop words
def lemma_synthetic_open_memory(lemma_dir, synthetic_dir, out_dir):
    lemma_data = open(os.path.join(data_root, lemma_dir, 'train_m.json'), 'r').read().strip().split('\n')
    synthetic_data = open(os.path.join(data_root, synthetic_dir, 'train_m.json'), 'r').read().strip().split('\n')
    out_f = open(os.path.join(data_root, out_dir, 'train_m.json'), 'w')
    stopwords_list = stopwords.words('english')
    for lemma, synthetic in zip(lemma_data, synthetic_data):
        lemma = json.loads(lemma)
        synthetic = json.loads(synthetic)
        mention_memory, bert_memory, bert_scores = [], [], []
        for w in lemma['mention_memory']:
            if w in stopwords_list: continue
            mention_memory.append(w)
        mention_memory.extend(synthetic['synthetic_memory'])
        mention_memory = list(set(mention_memory))
        for w, s in zip(lemma['bert_memory'], lemma['bert_scores']):
            if w in stopwords_list: continue
            bert_memory.append(w)
            bert_scores.append(s)
        lemma['mention_memory'] = mention_memory
        lemma['bert_memory'] = bert_memory
        lemma['bert_scores'] = bert_scores
        out_f.write(json.dumps(lemma) + '\n')
    out_f.close()
    print('train_m.json ok!')

    # data_names = ['dev.json', 'test.json']
    # for data_name in data_names:
    #     lemma_data = open(os.path.join(data_root, lemma_dir, data_name), 'r').read().strip().split('\n')
    #     out_f = open(os.path.join(data_root, out_dir, data_name), 'w')
    #     for lemma in lemma_data:
    #         lemma = json.loads(lemma)
    #         mention_memory, bert_memory, bert_scores = [], [], []
    #         for w in lemma['mention_memory']:
    #             if w in stopwords_list: continue
    #             mention_memory.append(w)
    #         for w, s in zip(lemma['bert_memory'], lemma['bert_scores']):
    #             if w in stopwords_list: continue
    #             bert_memory.append(w)
    #             bert_scores.append(s)
    #         lemma['mention_memory'] = mention_memory
    #         lemma['bert_memory'] = bert_memory
    #         lemma['bert_scores'] = bert_scores
    #         out_f.write(json.dumps(lemma) + '\n')
    #     out_f.close()
    #     print(data_name + ' ok!')


# all ontonotes memory
def add_ontonotes_memory(label_path, suffix, diff, data_names):
    all_labels = open(label_path, 'r').read().strip().split('\n')

    if not os.path.exists(os.path.join(data_root, diff + '_original')):
        os.makedirs(os.path.join(data_root, diff + '_original'))
    for data_name in data_names:
        file = open(os.path.join(data_root, diff, data_name + suffix), 'r').read().strip().split('\n')
        print(len(file))
        out_file = open(os.path.join(data_root, diff + '_original', data_name + suffix), 'w')
        id = 0
        for line in file:
            mention_memory, sentence_memory, bert_memory, bert_scores = [], [], [], []
            line = json.loads(line)

            # mention and sentence memory
            mention = line['mention'].lower()
            sentence = line['sentence'].lower()
            mention_words = [word.lemma for word in nlp(mention).sentences[0].words]
            sentence_words = [word.lemma for word in nlp(sentence).sentences[0].words]
            for word in mention_words:
                mention_memory.append(word)
            for word in sentence_words:
                sentence_memory.append(word)
            line['mention_memory'] = mention_memory
            line['sentence_memory'] = sentence_memory

            # bert memory
            context = line['context']
            answers = fill_mask_model(context)
            for pred in answers:
                bert_memory.append(index2word[pred['token']])
                bert_scores.append(pred['score'])
            line['bert_memory'] = bert_memory
            line['bert_scores'] = bert_scores

            out_file.write(json.dumps(line) + '\n')
            id += 1

        print(id, data_name + ' ok!')
        out_file.close()


# all ontonotes memory
def add_ontonotes_memory_batch(label_path, suffix, diff, data_names):
    all_labels = open(label_path, 'r').read().strip().split('\n')

    if not os.path.exists(os.path.join(data_root, diff + '_original_batch')):
        os.makedirs(os.path.join(data_root, diff + '_original_batch'))
    batch_size = 1024
    for data_name in data_names:
        file = open(os.path.join(data_root, diff, data_name + suffix), 'r').read().strip().split('\n')
        print(len(file))
        out_path = os.path.join(data_root, diff + '_original_batch', data_name + suffix)
        if os.path.exists(out_path):
            print('models Path: %s is existed, overwrite (y/n)?' % out_path)
            answer = input()
            if answer.strip().lower() == 'y':
                y=1
                # shutil.rmtree(out_path)
            else:
                exit(1)
        out_file = open(out_path, 'w')

        id = 0
        batch_num = len(file) // batch_size + 1
        print(batch_num)
        for batch_id in range(batch_num):
            s_t = time.time()
            if batch_id * batch_size > len(file): break
            lines = file[batch_id * batch_size: batch_id * batch_size + batch_size]
            lines = [json.loads(line) for line in lines]
            contexts = [line['context'] for line in lines]
            answers = fill_mask_model(contexts)
            for line_id, line in enumerate(lines):
                mention_memory, sentence_memory, bert_memory, bert_scores = [], [], [], []
                # mention and sentence memory
                mention = line['mention'].lower()
                sentence = line['sentence'].lower()
                mention_words = [word.lemma for word in nlp(mention).sentences[0].words]
                # sentence_words = [word.lemma for word in nlp(sentence).sentences[0].words]
                for word in mention_words:
                    mention_memory.append(word)
                # for word in sentence_words:
                #     sentence_memory.append(word)
                line['mention_memory'] = mention_memory
                # line['sentence_memory'] = sentence_memory

                # bert memory
                answer = answers[line_id]
                for pred in answer:
                    bert_memory.append(index2word[pred['token']])
                    bert_scores.append(pred['score'])
                line['bert_memory'] = bert_memory
                line['bert_scores'] = bert_scores

                out_file.write(json.dumps(line) + '\n')
                id += 1
            e_t = time.time()
            print(batch_id, 'consume', e_t - s_t)

        out_file.close()
        print(id, data_name + ' ok!')
        print('add_ontonotes_memory_batch ok! ')


# all ontonotes memory bert -> lemma
def add_ontonotes_memory_lemma(input_dir, suffix, diff, data_names):
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    # for Ultra-Fine dataset, original convert to lemma
    if not os.path.exists(os.path.join(data_root, diff + '_original_lemma')):
        os.makedirs(os.path.join(data_root, diff + '_original_lemma'))
    for data_name in data_names:
        file = open(os.path.join(input_dir, data_name + suffix), 'r').read().strip().split('\n')
        out_path = os.path.join(data_root, diff + '_original_lemma', data_name + suffix)
        # if os.path.exists(out_path):
        #     print('models Path: %s is existed, overwrite (y/n)?' % out_path)
        #     answer = input()
        #     if answer.strip().lower() == 'y':
        #         y=1
        #         # shutil.rmtree(out_path)
        #     else:
        #         exit(1)
        out_file = open(out_path, 'w')
        id = 0
        s_t = time.time()
        for line in file:
            bert_memory, bert_scores = [], []
            line = json.loads(line)
            # bert memory convert to lemma
            for word, score in zip(line['bert_memory'], line['bert_scores']):
                if "##" in word: continue
                bert_word = wordnet_lemmatizer.lemmatize(word)
                # bert_words = [word.lemma for word in nlp(word).sentences[0].words]
                # if len(bert_words) > 1:
                #     continue
                #     print(word, bert_words)
                bert_memory.append(bert_word)
                bert_scores.append(score)
            assert len(bert_memory) == len(bert_scores)

            line['bert_memory'] = bert_memory
            line['bert_scores'] = bert_scores

            out_file.write(json.dumps(line) + '\n')

            if id % 10000 == 0:
                e_t = time.time()
                print(id, e_t - s_t)
                s_t = time.time()
            id += 1

        out_file.close()
        print(data_name + ' ok!')
        print('add_ontonotes_memory_lemma ok! ')

# ontonotes train
def synthetic_ontonotes(label_path, suffix, diff, train_name):
    all_labels = open(label_path, 'r').read().strip().split('\n')
    train_data = open(os.path.join(data_root, diff, train_name + suffix), 'r').read().strip().split('\n')
    if not os.path.exists(os.path.join(data_root, diff + '_synthetic')):
        os.makedirs(os.path.join(data_root, diff + '_synthetic'))
    out_path = os.path.join(data_root, diff + '_synthetic', train_name + '_syn' + suffix)
    # if os.path.exists(out_path):
    #     print('models Path: %s is existed, overwrite (y/n)?' % out_path)
    #     answer = input()
    #     if answer.strip().lower() == 'y':
    #         y = 1
    #         # shutil.rmtree(out_path)
    #     else:
    #         exit(1)
    out_file = open(out_path, 'w')
    few_shot_vocab = open('ontonotes_few_shot_vocab10000.txt', 'r').read().strip().split('\n')
    for line in train_data:
        line = json.loads(line)
        synthetic_memory = []
        # rule: opt:neg = 1:1
        intersection = list(set(line["labels"]).intersection(set(few_shot_vocab)))
        if len(intersection) > 0:
            synthetic_memory.extend(intersection)
            rand_labels = all_labels.copy()
            rand_labels = list(set(rand_labels).difference(set(intersection)))
            rand_labels = random.sample(rand_labels, len(intersection))
            synthetic_memory.extend(rand_labels)
        else:
            rand_labels = line["labels"].copy()
            rand_labels = random.sample(rand_labels, 1)
            synthetic_memory.extend(rand_labels)
            rand_labels = all_labels.copy()
            rand_labels = list(set(rand_labels).difference(set(line["labels"])))
            rand_labels = random.sample(rand_labels, 1)
            synthetic_memory.extend(rand_labels)

        if len(synthetic_memory) == 0:
            seq_labels = line['seq_labels'].copy()
            rand_index = random.sample(seq_labels, min(2, len(seq_labels)))
            synthetic_memory.extend(rand_index)
        random.shuffle(synthetic_memory)
        line['synthetic_memory'] = synthetic_memory
        out_file.write(json.dumps(line) + '\n')

    out_file.close()
    print('synthetic_ontonotes ok! ')


# merge lemma and memory, remove stop words
def lemma_sythetic_ontonotes(lemma_dir, synthetic_dir, out_dir, train_name):
    lemma_data = open(os.path.join(data_root, lemma_dir, train_name + '.json'), 'r').read().strip().split('\n')
    synthetic_data = open(os.path.join(data_root, synthetic_dir, train_name + '_syn.json'), 'r').read().strip().split('\n')
    out_path = os.path.join(data_root, out_dir, train_name + '.json')
    # if os.path.exists(out_path):
    #     print('models Path: %s is existed, overwrite (y/n)?' % out_path)
    #     answer = input()
    #     if answer.strip().lower() == 'y':
    #         y = 1
    #         # shutil.rmtree(out_path)
    #     else:
    #         exit(1)
    out_f = open(out_path, 'w')
    stopwords_list = stopwords.words('english') + ["'s", "...", "''", "``"]
    for lemma, synthetic in zip(lemma_data, synthetic_data):
        lemma = json.loads(lemma)
        synthetic = json.loads(synthetic)
        synthetic_labels = [label[1:].split('/')[0] for label in synthetic['synthetic_memory']]
        mention_memory, bert_memory, bert_scores = [], [], []
        for w in lemma['mention_memory']:
            if w in stopwords_list or len(w) < 2: continue
            mention_memory.append(w)
        mention_memory.extend(synthetic_labels)
        mention_memory = list(set(mention_memory))
        # for w in lemma['sentence_memory']:
        #     if w in stopwords_list or len(w) < 2: continue
        #     sentence_memory.append(w)
        # sentence_memory = list(set(sentence_memory))
        for w, s in zip(lemma['bert_memory'], lemma['bert_scores']):
            if w in stopwords_list or len(w) < 2: continue
            bert_memory.append(w)
            bert_scores.append(s)
        lemma['mention_memory'] = mention_memory
        # lemma['sentence_memory'] = sentence_memory
        lemma['bert_memory'] = bert_memory
        lemma['bert_scores'] = bert_scores
        out_f.write(json.dumps(lemma) + '\n')
    out_f.close()
    print('train.json ok!')

    data_names = ['dev.json', 'test.json']
    for data_name in data_names:
        lemma_data = open(os.path.join(data_root, lemma_dir, data_name), 'r').read().strip().split('\n')
        out_f = open(os.path.join(data_root, out_dir, data_name), 'w')
        for lemma in lemma_data:
            lemma = json.loads(lemma)
            mention_memory, bert_memory, bert_scores = [], [], []
            for w in lemma['mention_memory']:
                if w in stopwords_list or len(w) < 2: continue
                mention_memory.append(w)
            # for w in lemma['sentence_memory']:
            #     if w in stopwords_list or len(w) < 2: continue
            #     sentence_memory.append(w)
            # sentence_memory = list(set(sentence_memory))
            for w, s in zip(lemma['bert_memory'], lemma['bert_scores']):
                if w in stopwords_list or len(w) < 2: continue
                bert_memory.append(w)
                bert_scores.append(s)
            lemma['mention_memory'] = mention_memory
            # lemma['sentence_memory'] = sentence_memory
            lemma['bert_memory'] = bert_memory
            lemma['bert_scores'] = bert_scores
            out_f.write(json.dumps(lemma) + '\n')
        out_f.close()
        print(data_name + ' ok!')
    print('lemma_sythetic_ontonotes ok! ')

# ===> open type dataset
# add_self_memory(label_path='seq_label_vocab.txt', suffix='.json', diff='open_type', data_names=['train', 'dev', 'test'])
# add_open_memory(label_path='seq_label_vocab.txt', suffix='.json', diff='open_type', data_names=['train_m'])  # ['train', 'dev', 'test']
# add_open_memory_lemma(input_dir='data/open_type_original/', suffix='.json', diff='open_type', data_names=['train_m'])  # 'train', 'dev', 'test'
# mapping_open_memory(label_path='seq_label_vocab.txt', suffix='.json', diff='open_type_original', data_names=['dev'], function='COS')  # 'train', 'test'
# synthetic_open_memory(label_path='seq_label_vocab.txt', suffix='.json', diff='open_type', train_name='train_m')
lemma_synthetic_open_memory(lemma_dir='open_type_original_lemma/', synthetic_dir='open_type_synthetic/', out_dir='open_type_original_lemma_synthetic/')

# ===> ontonotes dataset
# {'government': ['organization', 'structure'], 'legal': ['other', 'person'], 'military': ['organization', 'person']}
# add_ontonotes_memory_batch(label_path='onto_ontology.txt', suffix='.json', diff='ontonotes', data_names=['train'])  # 'train', 'dev', 'test', train_m
# add_ontonotes_memory(label_path='onto_ontology.txt', suffix='.json', diff='ontonotes', data_names=['dev'])  # 'train', 'dev', 'test'
# add_ontonotes_memory_lemma(input_dir='data/ontonotes_original/', suffix='.json', diff='ontonotes', data_names=['train_m'])  # 'train', 'dev', 'test'

# for funtion in ['add', 'average']:
#     mapping_ontonotes_memory(label_path='onto_ontology.txt', suffix='.json', diff='ontonotes_original_lemma', data_names=['dev', 'test'], function=funtion)  #'train', 'dev', 'test'
# mapping_ontonotes_memory(label_path='ontonotes_mapping.json', suffix='.json', diff='ontonotes_original_lemma', data_names=['dev', 'test'], function='simple')  #'train', 'dev', 'test'
# synthetic_ontonotes(label_path='onto_ontology.txt', suffix='.json', diff='ontonotes_original_lemma', train_name='train_m')
# lemma_sythetic_ontonotes(lemma_dir='ontonotes_original_lemma', synthetic_dir='ontonotes_original_lemma_synthetic', out_dir='ontonotes_original_lemma_synthetic', train_name='train_m')
