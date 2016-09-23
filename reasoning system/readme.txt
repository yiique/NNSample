1. preprocess 数据成为dict形式
2. 生成字典class，包含为每个词生成向量的方式
Note 论文里采用辅助训练的方法提高性能，这里采用分开训练，先训练RNNAutoEncode，再训练DNN
3. RNNencoder
4. DNNreasoner
