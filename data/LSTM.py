
# coding: utf-8

# In[2]:


import os
import re
import csv
import sys
import jieba
import pickle
import codecs
import gensim
import logging
import numpy as np
np.random.seed(2018)
import pandas as pd 
from keras import backend as K
from keras.models import Model
from string import punctuation
from gensim.models import word2vec
from keras.models import Sequential
from gensim.models import KeyedVectors
from keras.engine.topology import Layer
from keras.layers.merge import concatenate
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from keras import initializers, regularizers, constraints
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation


###########################
#####Attention类定义#######
###########################
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim
    
##########################
##########函数集合########
##########################

###结巴分词
def cuttxt(train_file_name):
    f1 = open(cut_txt,encoding="utf-8")
    f2= open(cut_txt_r,"w",encoding="utf-8")
    lines = f1.readlines()
    for line in lines:
        line.replace('\t',"").replace('\n',"").replace(" ","")
        seg_list=jieba.cut(line,cut_all=False)
        f2.write(" ".join(seg_list))
    f1.close()
    f2.close()
    return cut_txt_r

###word2vec模型训练
def model_train(train_file_name, save_model_file,n_dim):  # model_file_name为训练语料的路径,save_model为保存模型名
    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    train_file_name_cut=cuttxt(train_file_name)
    sentences = word2vec.Text8Corpus(train_file_name_cut)  # 加载语料
    model = gensim.models.Word2Vec(sentences, size=n_dim)  # 训练skip-gram模型; 默认window=5
    model.save(save_model_file)
    
###定义模型
def get_model():
    embedding_layer = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False)

    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True)

    inp = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences= embedding_layer(inp)
    x = lstm_layer(embedded_sequences)
    x = Dropout(rate_drop_dense)(x)
    merged = Attention(MAX_SEQUENCE_LENGTH)(x)
    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    outp = Dense(1)(merged)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['mse'])

    return model
    
#########################   
#######命名预处理########
#########################

###命名文件——训练集与测试集
train_name = "./training-inspur.csv"
predict_name = "./Preliminary-texting.csv"

###cut_txt——分词前文本集合
###cut_txt_r——分词后文本词集合
###文件编码格式为utf-8
cut_txt = './dl_text.txt'
cut_txt_r = './dl_text_r.txt' 

###词向量模型以及特征保存路径
save_model_name = './Word300.model'
save_feature = './df_fea.pkl'

###########################
######实验参数设置#########
###########################

###w2v的特征维度
max_features = 30000
maxlen = 200
validation_split = 0.1

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

###向量维度
embed_size = 300

###LSTM参数
num_lstm = 300
num_dense = 256
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

act = 'relu'

###########################
########读取数据###########
###########################
train = pd.read_csv(train_name)
train_row = train.shape[0]
df_train = train
predict = pd.read_csv(predict_name)
df_test = predict


###########################
#########数据预处理########
###########################
###判断w2v模型是否存在
if not os.path.exists(save_model_name):
    model_train(cut_txt_r, save_model_name,EMBEDDING_DIM)
else:
    print('此训练模型已经存在，不用再次训练')

###加载w2v模型
w2v_model = word2vec.Word2Vec.load(save_model_name)

###判断特征是否存在：Y直接加载 N结巴分词生成特征
if not os.path.exists(save_feature):
    df = pd.concat([df_train,df_test])
    df['cut_discuss'] = df['COMMCONTENT'].astype("str").map(lambda x : " ".join(i for i in jieba.cut(x)))
    fw = open(save_feature,'wb') 
    pickle.dump(df,fw)
    fw.close()
else:
    print("特征存在，直接加载...")
    fw = open(save_feature,'rb') 
    df = pickle.load(fw)
    fw.close()

###取出第三列的所有行
X_train = df.iloc[:train_row,3]
X_test = df.iloc[train_row:,3]

###Tokenizer进行词法分析
tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(X_train) + list(X_test))

###转换word下标的向量形式
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

###将序列填充到maxlen长度
x_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

###训练集结果集
y =df_train['COMMLEVEL'].values


###找出lstm权重
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))

###计算权重
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= max_features: 
        continue
    else :
        try:
            embedding_vector = w2v_model[word]
        except KeyError:
            continue
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector



###########################
########划分训练集#########
###########################
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, train_size=0.85, random_state=250)

###########################
#########定义模型##########
###########################

####模型参数
model = get_model()
STAMP = 'simple_lstm_w2v_vectors_%.2f_%.2f'%(rate_drop_lstm,rate_drop_dense)

###提前停止定义
early_stopping =EarlyStopping(monitor='val_loss', patience=2)
###保存位置
bst_model_path = "./" + STAMP + '.h5'
###保存最优
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

###批尺寸64
batch_size = 64
###迭代次数20
epochs = 20

#########################
########模型拟合#########
#########################
hist = model.fit(X_tra, y_tra,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size, shuffle=True,
         callbacks=[early_stopping,model_checkpoint])
         
model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

#########################
#########模型预测########
#########################
y_test = model.predict(test, batch_size=256, verbose=1)

#########################
##########结果保存#######
#########################
f = codecs.open("./result.txt",'w','utf-8')
for i in y_test:
    f.write(str(i).strip('[]')+'\r\n')#\r\n为换行符
f.close()

b = np.loadtxt('./result.txt')
df=pd.DataFrame(b, columns = ['COMMLEVEL'])
#参数可调优
df["y"]=df['COMMLEVEL'].apply(lambda x: 1 if x<1.758 else(3 if x>2.5726 else 2))
#输出结果集
df.to_csv('./dsjyycxds_preliminary.txt',columns=['y'],index = False,header=False)

