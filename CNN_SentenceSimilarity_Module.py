# -*- coding: utf-8 -*-
# @Author  : Junru_Lu
# @File    : CNN_SentenceSimilarity_Module.py
# @Software: PyCharm
# @Environment : Python 3.6+

# 基础包
import tensorflow as tf
import math
import jieba
import numpy as np
from gensim.models import keyedvectors

# 编码相关包
import importlib, sys
importlib.reload(sys)


'''
本配置文件用于测试和调用预训练完成的句子相似度判断CNN模型
'''


# ------------------预加载------------------ #


word_vectors = keyedvectors.KeyedVectors.load('你的词向量地址')  # 加载预先训练好的词向量
MAX_LENTH = 40  # 训练时保留的最多N-gram数量, 必须为4的倍数
OUT_SIZE1 = int(MAX_LENTH / 4)  # MAX_LENTH / 4 = 10
OUT_SIZE2 = int(MAX_LENTH / 2)  # MAX_LENTH / 2 = 20
CLASS_TYPE = 2
GRAM = 3  # n-gram
path = '你的根目录' + '/CNN_' + str(GRAM) + "_" + str(MAX_LENTH)
saverpath = path + '/CNN_' + str(GRAM) + "_" + str(MAX_LENTH) + '.ckpt'  # 预训练的模型地址


# ------------------基础函数------------------ #


def sen_vector_gen(title_words):  # 生成句子的词向量
    sen_vector = np.zeros(60, dtype=float)
    for word in title_words:
        try:
            sen_vector += word_vectors[word]
        except:
            pass
    sen_vector = sen_vector / len(title_words)  # 用词的词向量的平均值来表示句子向量
    return [sen_vector]


def get_vec_cosine(vec1, vec2):
    tmp = np.vdot(vec1, vec1) * np.vdot(vec2, vec2)
    if tmp == 0.0:
        return 0.0
    return np.vdot(vec1, vec2) / math.sqrt(tmp)


def cut_sentence_ngram(s_input):  # 抽取句子的n-gram
    gram = GRAM
    s_ngram = []
    ii = 0
    while ii < len(s_input) - (gram - 1):
        s_ngram.append(s_input[ii:ii+gram])
        ii += 1
    return s_ngram


def s1_s2_simipics(s1, s2, max_lenth):  # 生成feature map
    s1_ngram = cut_sentence_ngram(s1)
    s2_ngram = cut_sentence_ngram(s2)
    k = 0
    simi = []
    while k < max_lenth:
        try:
            sen_k = sen_vector_gen(jieba.lcut(s1_ngram[k]))
            j = 0
            while j < max_lenth:
                try:
                    sen_j = sen_vector_gen(jieba.lcut(s2_ngram[j]))
                    simi_pic = get_vec_cosine(sen_k, sen_j)
                except:
                    simi_pic = 0.0
                simi.append(simi_pic)
                j += 1
        except:
            simi_pic = 0.0
            simi.append(simi_pic)
        k += 1
    while len(simi) < MAX_LENTH**2:
        simi.append(0.0)
    return simi


# ------------------CNN------------------ #


def weight_variable(shape):  # 定义初始权重
    # 形状为shape的随机变量
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # 定义初始偏置值
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride[1,x_movement,y_movement,1]
    # padding=same:不够的地方补0；padding=valid:会缩小
    # 2维卷积层,卷积步长为(x=1,y=1)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # maxpooling
    # ksize表示核函数大小
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def restore(saver_path, sens_features):

    keep_prob = tf.placeholder(tf.float32)
    # None表示无论多少行例子都可以
    xs = tf.placeholder(tf.float32, [MAX_LENTH ** 2], 'x_input')
    # -1表示feature map个数,1表示Channel个数
    x_image = tf.reshape(xs, [-1, MAX_LENTH, MAX_LENTH, 1])

    # 第一层卷积+pooling
    # 核函数大小patch=2*2;通道数，即特征数为1所以in_size=1;新特征的厚度为OUT_SIZE1
    W_conv1 = weight_variable([5, 5, 1, OUT_SIZE1])
    b_conv1 = bias_variable([OUT_SIZE1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积+pooling
    # 核函数大小patch=2*2;in_size=4;新特征的厚度为OUT_SIZE2
    W_conv2 = weight_variable([5, 5, OUT_SIZE1, OUT_SIZE2])
    b_conv2 = bias_variable([OUT_SIZE2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 第一层全连接层func1 layer
    W_fc1 = weight_variable([OUT_SIZE1 * OUT_SIZE1 * OUT_SIZE2, MAX_LENTH])
    b_fc1 = bias_variable([MAX_LENTH])
    h_pool2_flat = tf.reshape(h_pool2, [-1, OUT_SIZE1 * OUT_SIZE1 * OUT_SIZE2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 第二层全连接层func2 layer
    W_fc2 = weight_variable([MAX_LENTH, CLASS_TYPE])
    b_fc2 = bias_variable([CLASS_TYPE])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, saver_path)
        prediction_result = prediction.eval(feed_dict={xs: sens_features, keep_prob: 1.0}, session=sess)
        return prediction_result.tolist()[0][0]


# ------------------主函数------------------ #


if __name__ == '__main__':

    s1 = input("请输入句子1：")
    s2 = input("请输入句子2：")

    print(restore(saverpath, s1_s2_simipics(s1, s2, MAX_LENTH)))
