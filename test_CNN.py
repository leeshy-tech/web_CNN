import json
from data_preprocess import *
from model import *
from train_CNN import validate, load_data
from hyp import hyp
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    embed_path = hyp['embed_path']
    data_path = hyp['data_path']
    batch_size = hyp['batch_size']
    nb_filters = hyp['nb_filters']
    dropout_rate = hyp['dropout_rate']
    embed_size = hyp['embed_size']
    model_weight = hyp['save_model_name']

    max_query_len, max_doc_len, max_url_len = defaultdict(int), defaultdict(int), defaultdict(int)
    vocab = {'word': {}}
    test_vocab = {'word': {}}
    train_vocab_emb, test_vocab_emb = None, None
    print('Using device:' + str(device))

    # ====================== Load data =========================
    # 读取训练/验证/测试集
    train_data = load_data(os.path.join(data_path, 'train.json'))
    test_data = load_data(os.path.join(data_path, 'test.json'))
    dev_data = load_data(os.path.join(data_path, 'dev.json'))

    train_dataset = gen_data(train_data, vocab, test_vocab, True, max_query_len, max_doc_len)
    valid_dataset = gen_data(dev_data, vocab, test_vocab, True, max_query_len, max_doc_len)
    test_dataset = gen_data(test_data, vocab, test_vocab, False, max_query_len, max_doc_len)

    print("Create dataset successfuly...")

    train_vocab_emb, test_vocab_emb = construct_vocab_emb("./experimental-data", vocab['word'], test_vocab['word'], 300,
                                                          base_embed_path=embed_path)

    # 融合训练集和测试集的词典
    merge_vocab = {}
    merge_vocab['word'] = merge_two_dicts(vocab['word'], test_vocab['word'])
    print("TRAIN vocab: word(%d)" % (len(vocab['word'])))
    print("TEST vocab: word(%d) " % (len(test_vocab['word'])))
    print("MERGE vocab: word(%d) " % (len(merge_vocab['word'])))

    vocab_inv, vocab_size = {}, {}
    for key in vocab:
        # key : word, char
        vocab_inv[key] = invert_dict(merge_vocab[key])
        vocab_size[key] = len(vocab[key])
    print(vocab_size)

    # 融合训练集和测试集的词向量
    merge_vocab_emb = np.zeros((len(merge_vocab['word']), 300))
    merge_vocab_emb[0: len(vocab['word']), :] = train_vocab_emb
    merge_vocab_emb[len(vocab['word']):len(merge_vocab['word']), :] = test_vocab_emb
    for key in vocab:
        vocab_size[key] = len(merge_vocab[key])

    # =================== Test model =====================
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    criterion.to(device)
    print("load best model ... ")
    # 定义一个新的模型
    new_model = creat_model(batch_size, vocab_size, merge_vocab_emb, nb_filters, embed_size, dropout_rate)
    new_model = new_model.to(device)
    # 加载最佳模型的参数赋给新建模型
    # model_weight = "checkpoint_textCNN.pt"
    new_model.load_state_dict(torch.load(model_weight), strict=False)
    print(model_weight)
    print('printing test result \n')
    # 测试集测试
    test_loss, test_fpr, test_tpr, test_auc, test_f1 = validate(new_model, test_dataset, test_dataset['sim'], \
                                                                batch_size, criterion)
    # 打印结果
    print("test_loss:", test_loss)
    print("test_auc:", test_auc)
    print('test_f1:', test_f1)