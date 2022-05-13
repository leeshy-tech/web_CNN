# hyp.py
hyp = {
    'embed_path': '..\sgns.wiki.bigram', # 记得修改位置
    'data_path': 'data',
    'batch_size': 2048,
    'nb_filters': 128,
    'dropout_rate': 0.3,
    'embed_size': 300,
    'learning_rate': 0.04,
    'epoches': 60,
    'save_model_name': "..\checkpoint_textCNN.pt"
}
