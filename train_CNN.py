# -*-coding:utf-8 -*-
from collections import defaultdict
import random
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from data_preprocess import *
from model import *
import json
from hyp import hyp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_torch(seed=123456789):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _set_data_type_to_tensor(data):
    """输入数据转换为tensor"""
    if type(data) == np.ndarray:
        data = torch.from_numpy(data.astype(np.int_)).long().to(device)
    elif type(data) == torch.Tensor:
        source_data_type = torch.LongTensor
    elif type(data) in [list, tuple]:
        # unpack list recursively and convert each element
        data = [_set_data_type_to_tensor(x) for x in data]
    else:
        assert False, 'Unknown data type: not numpy or torch.tensor'
    return data


def _num_records(x_data, y_data, num_records=None):
    """检查输入的x_data和y_data条数相同"""
    if type(x_data) in [list, tuple]:
        for x in x_data:
            num_records = _num_records(x, y_data, num_records)
    else:
        if num_records is None:
            num_records = x_data.size(0)
            if y_data is not None:
                # print("num_records:", num_records)
                # print("y:", y_data.size(0))
                assert num_records == y_data.size(0), "data and labels must be the same size"
                num_records = y_data.size(0)
        else:
            assert num_records == x_data.size(0), "all inputs sets must have same number of records"
            num_records = x_data.size(0)
    return num_records


def r_f1_thresh(y_pred, y_true, step=100):
    """f1值与阈值有关，因此找最佳阈值"""
    e = np.zeros((len(y_true), 2))
    e[:, 0] = y_pred.reshape(-1)
    e[:, 1] = y_true
    f = pd.DataFrame(e)
    thrs = np.linspace(0, 1, step + 1)
    x = np.array([f1_score(y_pred=f.loc[:, 0] > thr, y_true=f.loc[:, 1]) for thr in thrs])
    f1_, thresh = max(x), thrs[x.argmax()]
    return f.corr()[0][1], f1_, thresh


def train_fc(model, train_dataset, y, batch_size, opt, criterion):
    model.train()
    # print(train_dataset.keys())
    train_dataset_list = [train_dataset['query_word_input'], train_dataset['doc_word_input']]
    correct, train_loss = 0, 0
    y_pred, y_true = None, None
    # data to tensor
    y = _set_data_type_to_tensor(y)
    num_records = None
    for i in range(len(train_dataset_list)):
        train_dataset_list[i] = _set_data_type_to_tensor(train_dataset_list[i])
        # 数据总条数
        num_records = _num_records(train_dataset_list[i], y, num_records)
    # 数据分批次
    num_batches = int((num_records - 1) / batch_size)
    print("num_batches: ", num_batches)
    for batch in range(num_batches):
        batch_start = batch * batch_size
        batch_end = (batch + 1) * batch_size
        if batch_end > num_records:
            batch_end = num_records
        ixs = slice(batch_start, batch_end)
        x_batch_data = []
        for i in range(len(train_dataset_list)):
            x_batch_data.append(train_dataset_list[i][ixs])
        # 成Batch数据送入模型中
        x_out = model(x_batch_data[0], x_batch_data[1])
        target = y[ixs]
        # print("target:", target.size())
        # 优化器梯度清0
        opt.zero_grad()
        # 计算批损失
        batch_loss = criterion(x_out, target.long())
        # 梯度回传
        batch_loss.backward()
        train_loss += batch_loss.item()
        if not batch % 100:
            print("- batch {:d} batch_loss: {:.4f}".format(batch, batch_loss.item()))
        # print("batch_loss: ", batch_loss.item(), "batch:", batch)
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        # 模型学习
        opt.step()
        x_out = nn.Softmax(dim=1)(x_out)
        # x_out = nn.Softmax()(x_out)
        # correct += (torch.max(x_out, 1)[1].data == target.data).sum()
        # 下面计算train_f1
        x_out_ = x_out[:, 1].data.cpu().numpy()
        label = target.data.cpu().numpy()
        # print("label: ", label)
        if y_true is None:
            y_true = label
            y_pred = x_out_
        else:
            y_pred = np.concatenate((y_pred, x_out_), axis=0)
            y_true = np.concatenate((y_true, label), axis=0)
    r, f1, thresh = r_f1_thresh(y_pred=y_pred, y_true=y_true)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    print("train_loss: ", train_loss / num_batches)
    print('train_f1:', f1)
    return train_loss / num_batches, fpr, tpr, auc


def validate(model, val_data, y, batch_size, criterion):
    model.eval()
    valid_dataset_list = [val_data['query_word_input'], val_data['doc_word_input']]
    y = _set_data_type_to_tensor(y)
    y_true, y_pred = None, None
    for i in range(len(valid_dataset_list)):
        valid_dataset_list[i] = _set_data_type_to_tensor(valid_dataset_list[i])
        num_records = _num_records(valid_dataset_list[i], y)
    num_batches = int((num_records - 1) / batch_size)
    valid_loss = 0
    corrects = 0
    for batch in range(num_batches):
        batch_start = batch * batch_size
        batch_end = (batch + 1) * batch_size
        if batch_end > num_records:
            batch_end = num_records
        ixs = slice(batch_start, batch_end)
        x_batch_data = []
        target = y[ixs]
        for i in range(len(valid_dataset_list)):
            x_batch_data.append(valid_dataset_list[i][ixs])
        with torch.no_grad():
            x_out = model(x_batch_data[0], x_batch_data[1])
            batch_loss = criterion(x_out, target.long())
        valid_loss += batch_loss.item()
        x_out = nn.Softmax(dim=1)(x_out)
        # x_out = nn.Softmax()(x_out)
        corrects += (torch.max(x_out, 1)[1].data == target.data).sum()
        x_out_ = x_out[:, 1].data.cpu().numpy()
        label = target.data.cpu().numpy()
        if y_true is None:
            y_true = label
            y_pred = x_out_
        else:
            y_pred = np.concatenate((y_pred, x_out_), axis=0)
            y_true = np.concatenate((y_true, label), axis=0)
    r, f1, thresh = r_f1_thresh(y_pred=y_pred, y_true=y_true)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    # 打印预测正确率
    print("infer_acc:", torch.true_divide(corrects * 100, num_records).cpu())

    return valid_loss / num_batches, fpr, tpr, auc, f1


def load_data(data_path):
    """
    区分训练/验证/测试集
    @param data_path: 数据json文件存放位置
    @return: 训练/验证/测试集
    """

    with open(data_path) as f:
        data = json.load(f)

    return data


if __name__ == '__main__':
    seed_torch()
    # 设置参数
    embed_path = hyp['embed_path']
    data_path = hyp['data_path']
    batch_size = hyp['batch_size']
    nb_filters = hyp['nb_filters']
    dropout_rate = hyp['dropout_rate']
    embed_size = hyp['embed_size']
    learning_rate = hyp['learning_rate']
    epoches = hyp['epoches']

    # train_name, val_name, test_name, train_set, val_set, test_set, num_classes = 'train-', 'valid-', 'test-', ['train-'], ['valid-'], ['test-'], 2
    max_query_len, max_doc_len, max_url_len = defaultdict(int), defaultdict(int), defaultdict(int)
    vocab = {'word': {}}
    test_vocab = {'word': {}}
    train_vocab_emb, test_vocab_emb = None, None
    print('Using device:' + str(device))

    # =================== Load data =======================
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

    # ===================== TRAIN Model ======================
    # ### 定义模型 ###
    model = creat_model(batch_size, vocab_size, merge_vocab_emb, nb_filters, embed_size, dropout_rate)
    model = model.to(device)
    # 定义优化器
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                          weight_decay=1e-6, momentum=0.9, nesterov=True)
    lr_reducer = ReduceLROnPlateau(optimizer=opt, verbose=True)
    print("use SGD optimizer")
    # 定义损失函数
    print("compile model with binary_crossentropy")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    criterion.to(device)

    try:
        total_start_time = time.time()
        best_acc, best_f1, best_thresh = None, None, None
        best_auc = None
        best_fpr, best_tpr = None, None
        train_loss_list, valid_loss_list = [], []
        print("-" * 90)
        for epoch in range(epoches):
            epoch_start_time = time.time()
            # 训练
            train_loss, train_fpr, train_tpr, train_auc = train_fc(model, train_dataset, train_dataset['sim'],
                                                                   batch_size, opt, criterion)
            train_loss_list.append(train_loss)
            print("|start of epoch{:3d} | time : {:2.2f}s | loss {:5.6f} | train_auc {}".format(epoch + 1,
                                                                                                time.time() - epoch_start_time,
                                                                                                train_loss, train_auc))
            # 验证集上验证性能
            val_loss, val_fpr, val_tpr, val_auc, val_f1 = validate(model, valid_dataset, valid_dataset['sim'],
                                                                   batch_size, criterion)
            valid_loss_list.append(val_loss)
            lr_reducer.step(val_loss)
            print("-" * 10)
            print("| end of epoch {:3d}| time: {:2.2f}s | loss: {:.4f} |valid_auc {} |valid_f1 {}".format(epoch + 1,
                                                                                                          time.time() - epoch_start_time,
                                                                                                          val_loss,
                                                                                                          val_auc,
                                                                                                          val_f1))
            if not best_auc or best_auc < val_auc:
                best_auc = val_auc
                best_fpr = val_fpr
                best_tpr = val_tpr
                model_state_dict = model.state_dict()
                print("save the best model... best_auc: %s" % best_auc)
                model_weight = hyp['save_model_name']
                torch.save(model_state_dict, model_weight)
        with open('train_textCNN_loss.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(train_loss_list)
        with open('valid_textCNN_loss.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(valid_loss_list)
    except KeyboardInterrupt:
        print("-" * 90)
        print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time) / 60.0))

    print('Finish training')
