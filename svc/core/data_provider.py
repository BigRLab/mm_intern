#coding=utf-8
def load_vocab(vocab_path):
    vocab = {}
    fr = open(vocab_path)
    index = 0
    for row in fr.readlines():
        word = row.strip().decode('utf8')
        if word == '':
            continue
        vocab[word] = index
        index += 1
    fr.close()
    return vocab