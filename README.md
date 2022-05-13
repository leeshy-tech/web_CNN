# CNN模型——对话文本语义匹配
## 来源

本项目来自于北邮课程“Web搜索技术”

初始版模型来自于课程提供，并非本人原创。

本人添加了一些细枝末节并调参。

## 项目结构

```
CNN:
│  checkpoint_textCNN.pt
│  README.md
│  data_preprocess.py
│  hyp.py
│  model.py
│  sgns.wiki.bigram
│  test_CNN.py
│  train_CNN.py
│  train_textCNN_auc.csv
│  valid_textCNN_auc.csv  
├─data
│      dev.json
│      test.json
│      train.json
├─experimental-data
│      OOV_words.txt
├─readme
│      cnn.md
├─.vscode
└─__pycache__
```

- README.md：本文件
- cnn.md：助教编写的readme，因和本人写的readme冲突，故放在另一个文件夹中
- data：训练集、验证集、测试集数据
- hyp.py：超参数和其他参数
- train_CNN.py：训练模型
- test_CNN.py：测试模型

## 如何运行

除本文件夹代码外，还需要下载维基百科中文词向量映射：
网址：[维基百科中文词向量](https://pan.baidu.com/s/1ZKePwxwsDdzNrfkc6WKdGQ)
并在hyp.py对应其位置

- 训练：`python train_CNN.py`

- 测试：`python test_CNN.py`

## 更多信息

我会在本人的博客上记录本次实验过程：

