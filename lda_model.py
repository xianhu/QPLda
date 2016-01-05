# _*_ coding: utf-8 _*_

"""
LDA模型定义，一个文件搞定一个模型
"""

import os
import random
import logging


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class BiDictionary(object):
    """
    定义双向字典，通过key可以得到value，通过value也可以得到key
    """

    def __init__(self):
        """
        :key: 双向字典初始化
        """
        self.dict = {}            # 正向的数据字典，其key为self的key
        self.dict_reversed = {}   # 反向的数据字典，其key为self的value
        return

    def __len__(self):
        """
        :key: 获取双向字典的长度
        """
        return len(self.dict)

    def __str__(self):
        """
        :key: 将双向字典转化为字符串对象
        """
        str_list = ['%s\t%s' % (key, self.dict[key]) for key in self.dict]
        return '\n'.join(str_list)

    def clear(self):
        """
        :key: 清空双向字典对象
        """
        self.dict.clear()
        self.dict_reversed.clear()
        return

    def add_key_value(self, key, value):
        """
        :key: 更新双向字典，增加一项
        """
        self.dict[key] = value
        self.dict_reversed[value] = key
        return

    def remove_key_value(self, key, value):
        """
        :key: 更新双向字典，删除一项
        """
        if key in self.dict:
            del self.dict[key]
            del self.dict_reversed[value]
        return

    def get_value(self, key, default=None):
        """
        :key: 通过key获取value，不存在返回default
        """
        return self.dict.get(key, default)

    def get_key(self, value, default=None):
        """
        :key: 通过value获取key，不存在返回default
        """
        return self.dict_reversed.get(value, default)

    def contains_key(self, key):
        """
        :key: 判断是否存在key值
        """
        return key in self.dict

    def contains_value(self, value):
        """
        :key: 判断是否存在value值
        """
        return value in self.dict_reversed

    def keys(self):
        """
        :key: 得到双向字典全部的keys
        """
        return self.dict.keys()

    def values(self):
        """
        :key: 得到双向字典全部的values
        """
        return self.dict_reversed.keys()

    def items(self):
        """
        :key: 得到双向字典全部的items
        """
        return self.dict.items()


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class CorpusSet(object):
    """
    定义语料集（corpus）的类，作为LdaBase的基类
    """

    def __init__(self):
        """
        :key: 初始化函数
        """
        # 定义关于word的变量
        self.local_bi = BiDictionary()      # id和word之间的本地双向字典，key为id，value为word
        self.V = 0                          # 数据集中word的数量

        # 定义关于article的变量
        self.artids_list = []               # 全部article的id的列表，按照数据读取的顺序存储
        self.arts_Z = []                    # 全部article中所有词的id信息，维数为 M * art.length()
        self.M = 0                          # 数据集中article的数量

        # 定义推断中用到的全局变量（可能为空）
        self.global_bi = None               # id和word之间的全局双向字典，key为id，value为word
        self.local_2_global = {}            # 一个字典，local字典和global字典之间的对应关系
        return

    def init_corpus_with_file(self, file_name):
        """
        :key: 利用数据文件初始化语料集数据。文件每一行的数据格式：id[tab]word1 word2 word3......
        """
        with open(file_name, 'r', encoding="utf-8") as file_iter:
            self.init_corpus_with_articles(file_iter)
        return

    def init_corpus_with_articles(self, article_list):
        """
        :key: 利用article的列表初始化语料集。每一篇article的格式为：id[tab]word1 word2 word3......
        """
        # 清理数据--word数据
        self.local_bi.clear()

        # 清理数据--article数据
        self.artids_list.clear()
        self.arts_Z.clear()

        # 清理数据--清理local到global的映射关系
        self.local_2_global.clear()

        # 读取article数据
        for line in article_list:
            frags = line.strip().split()
            if len(frags) < 2:
                continue

            # 获取article的id
            art_id = frags[0].strip()

            # 获取word的id
            art_wordid_list = []
            for word in [w.strip() for w in frags[1:] if w.strip()]:
                local_id = self.local_bi.get_key(word) if self.local_bi.contains_value(word) else len(self.local_bi)

                if self.global_bi is None:
                    # 更新id信息
                    self.local_bi.add_key_value(local_id, word)
                    art_wordid_list.append(local_id)
                elif self.global_bi.contains_value(word):
                    # 更新id信息
                    self.local_bi.add_key_value(local_id, word)
                    art_wordid_list.append(local_id)

                    # 更新local_2_global
                    self.local_2_global[local_id] = self.global_bi.get_key(word)

            # 更新类变量：必须article中word的数量大于0
            if len(art_wordid_list) > 0:
                self.artids_list.append(art_id)
                self.arts_Z.append(art_wordid_list)

        # 做相关初始计算
        self.V = len(self.local_bi)
        self.M = len(self.artids_list)
        logging.debug("words number: " + str(self.V))
        logging.debug("articles number: " + str(self.M))
        return

    def save_wordmap(self, file_name):
        """
        :key: 保存word字典，即self.id_word_bi的数据
        """
        with open(file_name, "w", encoding="utf-8") as f_save:
            f_save.write(str(self.local_bi))
        return

    def load_wordmap(self, file_name):
        """
        :key: 加载word字典，即加载self.id_word_bi的值
        """
        self.local_bi.clear()
        with open(file_name, "r", encoding="utf-8") as f_load:
            for _id, _word in [line.strip().split() for line in f_load if line.strip()]:
                self.local_bi.add_key_value(int(_id), _word.strip())
        self.V = len(self.local_bi)
        return


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class LdaBase(CorpusSet):
    """
    LDA模型的基类，相关说明：
    》article的下标范围为[0, self.M), 下标为 m
    》wordid的下标范围为[0, self.V)，下标为 w
    》topic的下标范围为[0, self.K)，下标为 k 或 topic
    》article中word的下标范围为[0, article.size())，下标为 n
    """

    def __init__(self):
        """
        :key: 初始化函数
        """
        CorpusSet.__init__(self)

        # 基础变量--1
        self.dir_path = ""          # 文件夹路径，用于存放LDA运行的数据、中间结果等
        self.model_name = ""        # LDA训练或推断的模型名称，也用于读取训练的结果
        self.current_iter = 0       # LDA训练或推断的模型已经迭代的次数，用于继续模型训练过程
        self.iters_num = 0          # LDA训练或推断过程中Gibbs抽样迭代的总次数
        self.topics_num = 0         # LDA训练或推断过程中的topics的数量，即K值
        self.K = 0                  # LDA训练或推断的模型中的topic的数量，即self.topics_num
        self.twords_num = 0         # LDA训练或推断结束后输出与每个topic相关的word的个数

        # 基础变量--2
        self.alpha = []             # 超参数alpha，K维的float值，默认为50/K
        self.beta = []              # 超参数beta，V维的float值，默认为0.01

        # 基础变量--3
        self.Z = []                 # 所有word的topic信息，即Z(m, n)，维数为 M * article.size()

        # 统计计数(可由self.Z计算得到)
        self.nd = None              # nd[m][k]用于保存第m篇article中第k个topic产生的词的个数，其维数为 M * K
        self.ndsum = None           # ndsum[m]表示第m篇article的总词数，维数为 M
        self.nw = None              # nw[w][k]用于保存第w个词中被第k个topic产生的数量，其维数为 V * K（教程中为K*V）
        self.nwsum = None           # nwsum[k]表示第k个topic产生的词的总数，维数为 K

        # 多项式分布参数变量
        self.theta = None           # Doc-Topic多项式分布的参数，维数为 M * K，由alpha值影响
        self.phi = None             # Topic-Word多项式分布的参数，维数为 K * V，由beta值影响

        # 辅助变量，目的是提高算法执行效率
        self.sum_alpha = 0.0        # alpha的和
        self.sum_beta = 0.0         # beta的和

        # 推断时需要的训练模型
        self.train_model = None     # 推断时需要的训练模型
        return

    # --------------------------------------------------辅助函数---------------------------------------------------------
    def init_statistics_document(self):
        """
        :key: 初始化关于article的统计计数。先决条件：self.M, self.K, self.Z
        """
        assert self.M > 0 and self.K > 0 and self.Z

        # 统计计数初始化
        self.nd = [[0 for k in range(self.K)] for m in range(self.M)]
        self.ndsum = [0 for m in range(self.M)]

        # 根据self.Z进行更新，更新self.nd[m][k]和self.ndsum[m]
        for m, k_list in enumerate(self.Z):
            for k in k_list:
                self.nd[m][k] += 1
            self.ndsum[m] = len(self.Z[m])
        return

    def init_statistics_word(self):
        """
        :key: 初始化关于word的统计计数。先决条件：self.V, self.K, self.Z, self.arts_Z
        """
        assert self.V > 0 and self.K > 0 and self.Z and self.arts_Z

        # 统计计数初始化
        self.nw = [[0 for k in range(self.K)] for w in range(self.V)]
        self.nwsum = [0 for k in range(self.K)]

        # 根据self.Z进行更新，更新self.nw[w][k]和self.nwsum[k]
        for m in range(self.M):
            for w, k in zip(self.arts_Z[m], self.Z[m]):
                self.nw[w][k] += 1
                self.nwsum[k] += 1
        return

    def init_statistics(self):
        """
        :key: 初始化全部的统计计数。上两个函数的综合函数。
        """
        self.init_statistics_document()
        self.init_statistics_word()
        return

    def calculate_theta(self):
        """
        :key: 初始化并计算模型的theta值(M*K)，用到alpha值
        """
        assert self.sum_alpha > 0
        self.theta = [[(self.nd[m][k] + self.alpha[k]) /
                       (self.ndsum[m] + self.sum_alpha) for k in range(self.K)] for m in range(self.M)]
        return

    def calculate_phi(self):
        """
        :key: 初始化并计算模型的phi值(K*V)，用到beta值
        """
        assert self.sum_beta > 0
        self.phi = [[(self.nw[w][k] + self.beta[w]) /
                     (self.nwsum[k] + self.sum_beta) for w in range(self.V)] for k in range(self.K)]
        return

    def sum_alpha_beta(self):
        """
        :key: 计算alpha、beta的和
        """
        self.sum_alpha = sum(self.alpha)
        self.sum_beta = sum(self.beta)
        return

    def gibbs_sampling(self):
        """
        LDA模型中的Gibbs抽样过程
        """
        last_iter = self.current_iter + 1
        for self.current_iter in range(last_iter, last_iter+self.iters_num):
            logging.debug('\titeration ' + str(self.current_iter) + '......')
            for m in range(self.M):
                for n in range(len(self.Z[m])):
                    w = self.arts_Z[m][n]
                    k = self.Z[m][n]

                    # 统计计数减一
                    self.nd[m][k] -= 1
                    self.ndsum[m] -= 1
                    self.nw[w][k] -= 1
                    self.nwsum[k] -= 1

                    # 计算theta值--下边的过程为抽取第m篇article的第n个词w的topic，即新的k
                    theta_p = [(self.nd[m][k] + self.alpha[k]) /
                               (self.ndsum[m] + self.sum_alpha) for k in range(self.K)]

                    # 计算phi值--判断是训练模型，还是推断模型（注意self.beta[w_g]）
                    if self.local_2_global and self.train_model:
                        w_g = self.local_2_global[w]
                        phi_p = [(self.train_model.nw[w_g][k] + self.nw[w][k] + self.beta[w_g]) /
                                 (self.train_model.nwsum[k] + self.nwsum[k] + self.sum_beta) for k in range(self.K)]
                    else:
                        phi_p = [(self.nw[w][k] + self.beta[w]) /
                                 (self.nwsum[k] + self.sum_beta) for k in range(self.K)]

                    # multi_p为多项式分布的参数，此时没有进行标准化
                    multi_p = [theta_p[k] * phi_p[k] for k in range(self.K)]

                    # 将multi_p进行累加，然后确定随机数 u 落在哪个topic附近，此时的topic即为抽取的topic
                    for k in range(1, self.K):
                        multi_p[k] += multi_p[k-1]
                    u = random.random() * multi_p[self.K - 1]
                    for topic in range(self.K):
                        if multi_p[topic] > u:
                            # 此时的topic即为Gibbs抽样得到的topic，它有较大的概率命中多项式概率大的topic
                            k = topic
                            break

                    # 统计计数加一
                    self.nd[m][k] += 1
                    self.ndsum[m] += 1
                    self.nw[w][k] += 1
                    self.nwsum[k] += 1

                    # 更新Z值
                    self.Z[m][n] = k
        # 抽样完毕
        return

    # -----------------------------------Model数据存储、读取相关函数-------------------------------------------------------
    def save_parameter(self, file_name):
        """
        :key: 保存模型相关参数数据，包括：topics_num, M, V, K, alpha, beta
        """
        with open(file_name, "w", encoding="utf-8") as f_param:
            for item in ['topics_num', 'M', 'V', 'K']:
                f_param.write('%s\t%s\n' % (item, str(self.__dict__[item])))
            f_param.write('alpha\t%s\n' % ','.join([str(item) for item in self.alpha]))
            f_param.write('beta\t%s\n' % ','.join([str(item) for item in self.beta]))
        return

    def load_parameter(self, file_name):
        """
        :key: 加载模型相关参数数据，和上一个函数相对应
        """
        with open(file_name, "r", encoding="utf-8") as f_param:
            for line in f_param:
                key, value = line.strip().split()
                if key in ['topics_num', 'M', 'V', 'K']:
                    self.__dict__[key] = int(value)
                elif key in ['alpha', 'beta']:
                    self.__dict__[key] = [float(item) for item in value.split(',')]
        return

    def save_zvalue(self, file_name):
        """
        :key: 保存模型关于article的变量，包括：arts_Z, Z, artids_list等
        """
        with open(file_name, "w", encoding="utf-8") as f_zvalue:
            for m in range(self.M):
                out_line = [str(w) + ':' + str(k) for w, k in zip(self.arts_Z[m], self.Z[m])]
                f_zvalue.write('%s\t%s\n' % (self.artids_list[m], ' '.join(out_line)))
        return

    def load_zvalue(self, file_name):
        """
        :key: 读取模型的Z变量。和上一个函数相对应
        """
        self.arts_Z = []
        self.artids_list = []
        self.Z = []
        with open(file_name, "r", encoding="utf-8") as f_zvalue:
            for line in f_zvalue:
                frags = line.strip().split()
                self.artids_list.append(frags[0].strip())
                w_list = []
                k_list = []
                for w, k in [value.split(':') for value in frags[1:]]:
                    w_list.append(int(w))
                    k_list.append(int(k))
                self.arts_Z.append(w_list)
                self.Z.append(k_list)
        return

    def save_twords(self, file_name):
        """
        :key: 保存模型的twords数据，要用到phi的数据
        """
        self.calculate_phi()
        out_num = self.V if self.twords_num > self.V else self.twords_num
        with open(file_name, "w", encoding="utf-8") as f_twords:
            for k in range(self.K):
                words_list = sorted([(w, self.phi[k][w]) for w in range(self.V)], key=lambda x: x[1], reverse=True)
                f_twords.write("Topic %dth:\n" % k)
                f_twords.writelines(["\t%s %f\n" % (self.local_bi.get_value(w), p) for w, p in words_list[:out_num]])
        return

    def save_tag(self, file_name):
        """
        :key: 输出模型最终给数据打标签的结果，用到theta值
        """
        self.calculate_theta()
        with open(file_name, "w", encoding="utf-8") as f_tag:
            for m in range(self.M):
                f_tag.write('%s\t%s\n' % (self.artids_list[m], ' '.join([str(item) for item in self.theta[m]])))
        return

    def save_model(self):
        """
        :key: 保存模型
        """
        name_predix = "%s-%05d" % (self.model_name, self.current_iter)
        # 保存训练结果
        self.save_parameter(os.path.join(self.dir_path, '%s.%s' % (name_predix, "param")))
        self.save_wordmap(os.path.join(self.dir_path, '%s.%s' % (name_predix, "wordmap")))
        self.save_zvalue(os.path.join(self.dir_path, '%s.%s' % (name_predix, "zvalue")))

        #保存额外数据
        self.save_twords(os.path.join(self.dir_path, '%s.%s' % (name_predix, "twords")))
        self.save_tag(os.path.join(self.dir_path, '%s.%s' % (name_predix, "tag")))
        return


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class LdaModel(LdaBase):
    """
    LDA模型定义，主要实现训练、继续训练、推断的过程
    """

    def init_train_model(self, dir_path, model_name, current_iter, iters_num=500, topics_num=20, twords_num=200,
                         alpha=-1.0, beta=0.01, data_file=''):
        """
        :key: 初始化训练模型，根据参数current_iter决定是初始化新模型，还是加载已有模型
        :key: 当初始化新模型时，所有的参数都需要
        :key: 当加载已有模型时，只需要dir_path, model_name, current_iter, iters_num, twords_num即可
        """
        if current_iter == 0:
            logging.debug("init a new train model")

            # 初始化语料集
            self.init_corpus_with_file(data_file)

            # 初始化部分变量
            self.dir_path = dir_path
            self.model_name = model_name
            self.current_iter = current_iter
            self.iters_num = iters_num
            self.topics_num = topics_num
            self.K = topics_num
            self.twords_num = twords_num

            # 初始化alpha和beta
            self.alpha = [alpha if alpha > 0 else (50.0/self.K) for k in range(self.K)]
            self.beta = [beta if beta > 0 else 0.01 for w in range(self.V)]

            # 初始化Z值，以便统计计数
            self.Z = [[random.randint(0, self.K-1) for n in range(len(self.arts_Z[m]))] for m in range(self.M)]
        else:
            logging.debug("init an existed model")

            # 初始化部分变量
            self.dir_path = dir_path
            self.model_name = model_name
            self.current_iter = current_iter
            self.iters_num = iters_num
            self.twords_num = twords_num

            # 加载已有模型
            name_predix = "%s-%05d" % (self.model_name, self.current_iter)

            # 加载dir_path目录下的模型参数文件，即加载topics_num, M, V, K, alpha, beta
            self.load_parameter(os.path.join(self.dir_path, '%s.%s' % (name_predix, "param")))

            # 加载dir_path目录下的wordmap文件，即加载self.local_bi和self.V
            self.load_wordmap(os.path.join(self.dir_path, '%s.%s' % (name_predix, "wordmap")))

            # 加载dir_path目录下的zvalue文件，即加载self.Z, self.arts_Z, self.arts_list等
            self.load_zvalue(os.path.join(self.dir_path, '%s.%s' % (name_predix, "zvalue")))

        # 初始化统计计数
        self.init_statistics()

        # 初始化其他参数
        self.sum_alpha_beta()

        # 返回该模型
        return self

    def begin_gibbs_sampling_train(self):
        """
        :key: 训练模型，对语料集中的所有数据进行Gibbs抽样，并保存最后的抽样结果
        """
        logging.debug('sample iteration start')
        self.gibbs_sampling()
        logging.debug('sample iteration finish')

        # 保存模型
        logging.debug('save model')
        self.save_model()
        return

    def init_inference_model(self, train_model):
        """
        :key: 初始化推断模型
        """
        self.train_model = train_model

        # 初始化变量：主要用到self.topics_num, self.K
        self.topics_num = train_model.topics_num
        self.K = train_model.K

        # 初始化变量self.alpha, self.beta，直接沿用train_model的值
        self.alpha = train_model.alpha      # K维的float值，训练和推断模型中的K相同，故可以沿用
        self.beta = train_model.beta        # V维的float值，推断模型中用于计算phi的V值应该是全局的word的数量，故可以沿用
        self.sum_alpha_beta()               # 计算alpha和beta的和

        # 初始化数据集的self.global_bi
        self.global_bi = train_model.local_bi
        return

    def inference_data(self, article_list, iters_num=100, repeat_num=3):
        """
        :key: 利用现有模型推断数据
        :param article_list: 每一行的数据格式为：id[tab]word1 word2 word3......
        :param iters_num: 每一次迭代的次数
        :param repeat_num: 重复迭代的次数
        """
        # 初始化语料集
        self.init_corpus_with_articles(article_list)

        # 初始化返回变量
        return_theta = [[0.0 for k in range(self.K)] for m in range(self.M)]

        # 重复抽样
        for i in range(repeat_num):
            logging.debug('inference repeat_num: ' + str(i+1))

            # 初始化变量
            self.current_iter = 0
            self.iters_num = iters_num

            # 初始化Z值，以便统计计数
            self.Z = [[random.randint(0, self.K-1) for n in range(len(self.arts_Z[m]))] for m in range(self.M)]

            # 初始化统计计数
            self.init_statistics()

            # 开始推断
            self.gibbs_sampling()

            # 计算theta
            self.calculate_theta()
            for m in range(self.M):
                for k in range(self.K):
                    return_theta[m][k] += self.theta[m][k]

        # 计算结果，并返回
        for m in range(self.M):
            for k in range(self.K):
                return_theta[m][k] /= 3
        return return_theta


if __name__ == '__main__':
    """
    测试代码
    """
    logging.basicConfig(level=logging.DEBUG)
    test_type = "new"

    # 测试新模型
    if test_type == "new":
        model = LdaModel()
        model.init_train_model("data/", "model", current_iter=0, iters_num=100, topics_num=10, data_file="corpus.txt")
        model.begin_gibbs_sampling_train()
    elif test_type == "continue":
        model = LdaModel()
        model.init_train_model("data/", "model", current_iter=100, iters_num=200)
        model.begin_gibbs_sampling_train()
    elif test_type == "inference":
        model = LdaModel()
        model.init_inference_model(LdaModel().init_train_model("data/", "model", current_iter=100))
        data = [
            "com.cmcomiccp.client	咪咕 漫画 咪咕 漫画 漫画 更名 咪咕 漫画 资源 偷星 国漫 全彩 日漫 实时 在线看 随心所欲 登陆 漫画 资源 黑白 全彩 航海王 火影忍者 龙珠 日漫 强势 咪咕 漫画 国内 日本 集英社 全彩 漫画 版权 国内 知名 漫画 工作室 合作 蔡志忠 姚非 夏达 黄玉郎 逢春 漫画家 优秀作品 国漫 漫画 界面 简洁 直观 画面 全彩 横屏 竖屏 流量 漫画 世界 作者 咪咕 数字 传媒 中文 内容 ui 界面 图书 频道 bug4 新咪咕 梦想 更名 咪咕 漫画 漫画 更名 咪咕 漫画 资源 偷星 国漫 全彩 日漫 实时 在线看 随心所欲 漫画 界面 简洁 直观 画面 全彩 横屏 竖屏 流量 漫画 世界 咪咕 漫画 漫画 更名 咪咕 漫画 资源 偷星 国漫 全彩 日漫 实时 在线看 随心所欲 漫画 界面 简洁 直观 画面 全彩 横屏 竖屏 流量 漫画 世界 新闻 漫画 搞笑 电子书 热血 动作 新闻阅读 漫画 搞笑 电子书 热血 动作",
            "com.cnmobi.aircloud.activity	aircloud aircloud 硬件 设备 wifi 智能 手要 平板电脑 电脑 存储 aircloud 文件 远程 型号 aircloud 硬件 设备 wifi 智能 手要 平板电脑 电脑 存储 aircloud 文件 远程 型号 效率 办公 存储 云盘 工具 效率办公 存储·云盘 系统工具"
        ]
        result = model.inference_data(data)
        print(result)
    exit()
