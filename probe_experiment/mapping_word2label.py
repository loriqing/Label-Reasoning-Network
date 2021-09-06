import json, os
import numpy as np
import gensim.downloader as api
# gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
# gensim-data/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz
# print(api.load('glove-wiki-gigaword-100', return_path=True))
# print(api.load('glove-wiki-gigaword-300', return_path=True))
# word_vectors = api.load("gensim-data/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz")
# example
# result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
# print(result)
# similarity = word_vectors.similarity('woman', 'man')
# print(similarity)


def manage_glove(glove_path, output_dir, dim):
    error = 0
    glove_vec = open(glove_path, 'r').read().strip().split('\n')
    vocab = []
    vector_list = []
    for idx, line in enumerate(glove_vec):
        o_line = line
        line = o_line.strip().split(' ')
        vocab.append(line[0])
        vector = np.array(line[1:]).astype(float)
        if len(vector) != dim:
            error += 1
            continue
        # assert len(vector) == 300, "{0} {1}: {2}".format(idx, len(vector), o_line)
        vector_list.append(vector)

    with open(os.path.join(output_dir, 'glove_vocab_' + str(dim) + '.txt'), 'w') as f:
        f.write('\n'.join(vocab))
        f.close()
    path = os.path.join(output_dir, 'glove_vector_' + str(dim))
    vector_list = np.stack(vector_list)
    np.save(path, vector_list)
    print(error)

# manage_glove('data/glove.6B.300d.txt', 'data/glove/', dim=300)
# manage_glove('data/glove.6B.100d.txt', 'data/glove/', dim=100)


def ontonotes_glove_emb(glove_vector, vocab_path, mapping_label_path, output_path):
    vocab = open(vocab_path, 'r').read().strip().split('\n')
    vocab = {w: i for i, w in enumerate(vocab)}
    re_vocab = {i: w for w, i in vocab.items()}
    vector = np.load(glove_vector)
    mapping = json.load(open(mapping_label_path, 'r'))
    out_f = open(output_path, 'w')
    for idx, word in re_vocab.items():
        v = vector[idx]
        v_s = []
        for n in v:
            v_s.append(str(n))
        v_s = ' '.join(v_s)
        string = word + ' ' + v_s
        out_f.write(string + '\n')
    for word, labels in mapping.items():
        if word in vocab:
            idx = vocab[word]
            v = vector[idx]
            v_s = []
            for n in v:
                v_s.append(str(n))
            v_s = ' '.join(v_s)
            for label in labels:
                string = label + ' ' + v_s
                out_f.write(string + '\n')
    out_f.close()

ontonotes_glove_emb('data/glove/glove_vector_300.npy', 'data/glove/glove_vocab_300.txt', 'data/ontonotes_mapping.json', 'data/glove/glove.ontonotes_all.300d.txt')
exit()

word_vectors = api.load("glove-wiki-gigaword-100")
data_path = 'data/open_type_memory/'
data_names = ['train', 'dev', 'test']
all_labels = open('data/seq_label_vocab.txt', 'r').read().strip().split('\n')

for data_name in data_names:
    data = open(os.path.join(data_path, data_name + '.json'), 'r').read().strip().split('\n')
    for line in data:
        line = json.loads(line)
        memory = line['mention_memory']
        for m in memory:
            sims = {}
            for l in all_labels:
                try:
                    sim = word_vectors.similarity(m, l)
                except:
                    continue
                sims[l] = sim
            sort_sims = sorted(sims.items(), key = lambda kv: (kv[1], kv[0]), reverse=True)
            print(m)
            print(sort_sims[:2])




