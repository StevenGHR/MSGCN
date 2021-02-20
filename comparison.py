# encoding=utf-8
# _author = "Steven gao"

from utils import *
from dsc.dsc import ResDSC
from gcn.gcn_model import GCN
from gcn.multi_gcn import MGCN
from tensorflow.examples.tutorials.mnist import input_data
import json
import tensorflow as tf
import os


def train_dsc_model(data):
    configs["n_input"] = data.shape[1]
    tf.reset_default_graph()
    model = ResDSC(configs, data.shape)
    # model.restore()

    iteration = 0
    # scene15 : 4000
    # mnist : 8000
    pretrained_step = 4000
    train_step = 800
    display_step = 100
    is_training = True
    # pretrain the network
    batch_x = np.reshape(data, [-1, data.shape[1]])

    while iteration < pretrained_step:
        cost = model.ae_partial_fit(batch_x, is_training)
        if iteration % display_step == 0:
            print("======================  epoch: {}   ====================".format(iteration))
            print("pretrained loss: {}".format(cost))
        iteration += 1
    # model.save_model(pretrained_step, configs["n_hidden"][-1])

    coefs = None
    latents = None
    for i in range(train_step):
        cost, coefs, latents = model.partial_fit(batch_x, is_training)
        if i % display_step == 0:
            print("loss: {}".format(cost))
    model.sess.close()
    return coefs, latents


def train_gcn(labels, train_mask, val_mask, test_mask, coefs, latents):
    epochs = 1000
    samples_num = labels.shape[0]

    supports = list()
    features = list()
    num_supports = len(configs["n_hidden"]) + 1
    for coef in coefs:
        adj = coef_to_adj(coef, configs["adj_threshold"])
        supports.append(preprocess_adj(adj))
        # supports.append(adj)
        # coef = 0.5 * (np.abs(coef) + np.abs(coef.T))
        # supports.append(coef)
    for latent in latents:
        features.append(preprocess_features(latent))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        # # 自学习邻居，不使用稀疏（对应gcn_layers需要修改）
        # 'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
        'features': [tf.placeholder(tf.float32) for _ in range(num_supports)],
        'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=())
        # helper variable for sparse dropout
        # 'num_features_nonzero': [tf.placeholder(tf.int32) for _ in range(num_supports)]
    }

    # 使用单纯GCN模型
    # model = GCN(placeholders, configs, input_dim=x.shape[1], logging=False)

    # 使用多图卷积最终结果投票
    input_dim = list()
    for latent in latents:
        input_dim.append(latent.shape[1])
    model = MGCN(placeholders, samples_num, configs, input_dim=input_dim, logging=False)

    # Initialize sess
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = list()
    for epoch in range(epochs):
        # Training step
        feed_dict = construct_feed_dict(features, supports, labels, train_mask, placeholders)
        outs = sess.run([model.opt_op, model.loss, model.accuracy_mask], feed_dict=feed_dict)

        feed_dict_val = construct_feed_dict(features, supports, labels, val_mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy_mask], feed_dict=feed_dict_val)
        cost_val.append(outs_val[0])

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(outs_val[0]),
              "val_acc=", "{:.5f}".format(outs_val[1]))

        if epoch > configs["early_stopping"] and cost_val[-1] > np.mean(cost_val[-(configs["early_stopping"]+1):-1]):
            print("Early stopping...")
            break

    # test
    feed_dict_test = construct_feed_dict(features, supports, labels, test_mask, placeholders)
    outs_test = sess.run([model.loss, model.accuracy_mask], feed_dict=feed_dict_test)
    print("Test set results:", "cost=", "{:.5f}".format(outs_test[0]),
          "accuracy=", "{:.5f}".format(outs_test[1]))
    print("Optimization Finished!")
    sess.close()
    return outs_test[1]


def train_mnist():
    layer_units_num = [512, 128, 64]
    configs["n_hidden"] = layer_units_num
    configs["gcn_hidden"] = [[32], [32], [32], [32]]
    configs["dropout"] = 0
    configs["adj_threshold"] = 0.0029
    configs["early_stopping"] = 100
    # ====================load data=====================
    mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)
    x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    x_train = x_test[:10000, :]
    y_train = y_test[:10000, :]

    # # x_train = x_train / 255.0
    coefs, latents = train_dsc_model(x_train)

    res = list()
    samples_num = x_train.shape[0]
    semi_supervised_prop = 0.1
    # # multi-level ablation experiments
    # layer_units_num = [[256], [256, 128], [256, 128, 64, 32]]
    # gcn_hiddens = [[[32], [32]], [[32], [32], [32]], [[32], [32], [32], [32], [16]]]
    # for i in range(3):
    #     configs["n_hidden"] = layer_units_num[i]
    #     configs["gcn_hidden"] = gcn_hiddens[i]
    #     coefs, latents = train_dsc_model(x_train)

    # # gcn layer units ablation
    # gcn_hiddens = [[[50], [50], [50], [50]], [[60], [60], [60], [60]], [[70], [70], [70], [70]],
    #                [[80], [80], [80], [80]], [[90], [90], [90], [90]]]
    # for i in range(5):
    #     configs["gcn_hidden"] = gcn_hiddens[i]
    #     coefs, latents = train_dsc_model(x_train)

    # # graph construction proportion ablation
    # for i in [0.001, 0.005, 0.01, 0.05, 0.1]:
    #     configs["adj_threshold"] = i
    for semi_supervised_prop in [0.1, 0.2, 0.3]:
        train_mask = range(int(samples_num * semi_supervised_prop))
        val_mask = range(int(samples_num * semi_supervised_prop), int(samples_num * semi_supervised_prop) + 1000)
        test_mask = range(int(samples_num * semi_supervised_prop) + 1000, samples_num)
        train_mask = sample_mask(train_mask, samples_num)
        val_mask = sample_mask(val_mask, samples_num)
        test_mask = sample_mask(test_mask, samples_num)
        res.append(train_gcn(y_train, train_mask, val_mask, test_mask, coefs, latents))
    return res

    # data = np.zeros((10000, 784))
    # labels = np.zeros((10000, 10), dtype=np.float)
    # for i in range(10):
    #     index = np.where(y_train[:, i] == 1)[0][:1000]
    #     data[i * 1000: (i + 1) * 1000, :] = x_train[index, :]
    #     labels[i * 1000: (i + 1) * 1000, i] = 1.0
    # # random shuffle
    # indices = np.random.permutation(data.shape[0])
    # data = data[indices]
    # labels = labels[indices]
    # # random shuffle
    # indices = np.random.permutation(data.shape[0])
    # data = data[indices]
    # labels = labels[indices]
    #
    # # x_train = x_train / 255
    # coefs, latents = train_dsc_model(data)
    #
    # samples_num = data.shape[0]
    # res = list()
    # for semi_supervised_prop in [0.1]:
    #     train_mask = range(int(samples_num * semi_supervised_prop))
    #     val_mask = range(int(samples_num * semi_supervised_prop), int(samples_num * semi_supervised_prop) + 1000)
    #     test_mask = range(int(samples_num * semi_supervised_prop) + 1000, samples_num)
    #     train_mask = sample_mask(train_mask, samples_num)
    #     val_mask = sample_mask(val_mask, samples_num)
    #     test_mask = sample_mask(test_mask, samples_num)
    #
    #     res.append(train_gcn(labels, train_mask, val_mask, test_mask, coefs, latents))
    # return res


def train_scene15():
    layer_units_num = [512, 256]
    configs["n_hidden"] = layer_units_num
    configs["gcn_hidden"] = [[32], [32], [32]]
    configs["dropout"] = 0
    configs["adj_threshold"] = 0.08
    configs["early_stopping"] = 100
    # ====================load data=====================
    data = np.transpose(sio.loadmat("./data/scene15/scene15.mat")["featureMat"])
    labels = np.transpose(sio.loadmat("./data/scene15/scene15.mat")["labelMat"])

    # random shuffle
    indices = np.random.permutation(data.shape[0])
    data = data[indices]
    labels = labels[indices]

    indices = np.random.permutation(data.shape[0])
    data = data[indices]
    labels = labels[indices]
    # data = preprocess_features(data)
    coefs, latents = train_dsc_model(data)

    samples_num = data.shape[0]
    res = list()
    for semi_supervised_prop in [500]:
        train_mask = range(semi_supervised_prop)
        val_mask = range(semi_supervised_prop, semi_supervised_prop + 500)
        test_mask = range(semi_supervised_prop + 500, samples_num)
        train_mask = sample_mask(train_mask, samples_num)
        val_mask = sample_mask(val_mask, samples_num)
        test_mask = sample_mask(test_mask, samples_num)

        res.append(train_gcn(labels, train_mask, val_mask, test_mask, coefs, latents))
    print("{:.2%}".format(res[0]))
    return res


def train_cifar10():
    layer_units_num = [128, 64]
    configs["n_hidden"] = layer_units_num
    configs["gcn_hidden"] = [[16], [16], [16]]
    configs["dropout"] = 0
    configs["adj_threshold"] = 0.003
    configs["early_stopping"] = 100
    # ====================load data=====================
    cifar10 = sio.loadmat("./data/cifar10/preprocess_cifar10.mat")
    data = cifar10["data"]
    labels = cifar10["label"]
    data = preprocess_features(data)
    coefs, latents = train_dsc_model(data)

    # # random shuffle
    # indices = np.random.permutation(data.shape[0])
    # data = data[indices]
    # labels = labels[indices]

    samples_num = data.shape[0]
    res = list()
    semi_supervised_prop = 0.1
    layer_units_num = [[128], [128, 64, 32], [128, 64, 48, 32]]
    gcn_hiddens = [[[16], [16]], [[16], [16], [16], [16]], [[16], [16], [16], [16], [16]]]
    for i in range(3):
        configs["n_hidden"] = layer_units_num[i]
        configs["gcn_hidden"] = gcn_hiddens[i]
        coefs, latents = train_dsc_model(data)
    # for i in [0.001, 0.005, 0.01, 0.05, 0.1]:
    #     configs["adj_threshold"] = i
    # for semi_supervised_prop in [0.1, 0.2, 0.3]:
        train_mask = range(int(samples_num * semi_supervised_prop))
        val_mask = range(int(samples_num * semi_supervised_prop), int(samples_num * semi_supervised_prop) + 1000)
        test_mask = range(int(samples_num * semi_supervised_prop) + 1000, samples_num)
        train_mask = sample_mask(train_mask, samples_num)
        val_mask = sample_mask(val_mask, samples_num)
        test_mask = sample_mask(test_mask, samples_num)

        res.append(train_gcn(labels, train_mask, val_mask, test_mask, coefs, latents))
    return res


def train_SVHN():
    layer_units_num = [128, 64]
    configs["n_hidden"] = layer_units_num
    configs["gcn_hidden"] = [[16], [16], [16]]
    configs["dropout"] = 0
    configs["adj_threshold"] = 0.003
    configs["early_stopping"] = 100
    # ====================load data=====================
    svhn15 = sio.loadmat("./data/SVHN/preprocess_svhn.mat")
    data = svhn15["data"]
    labels = svhn15["label"]
    data = preprocess_features(data)

    # random shuffle
    indices = np.random.permutation(data.shape[0])
    data = data[indices]
    labels = labels[indices]
    # # data = preprocess_features(data)
    # coefs, latents = train_dsc_model(data)

    samples_num = data.shape[0]
    res = list()
    semi_supervised_prop = 0.1
    layer_units_num = [[128, 64, 32], [128, 64, 48, 32]]
    gcn_hiddens = [[[16], [16], [16], [16]], [[16], [16], [16], [16], [16]]]
    for i in range(2):
        configs["n_hidden"] = layer_units_num[i]
        configs["gcn_hidden"] = gcn_hiddens[i]
        coefs, latents = train_dsc_model(data)
    # for i in [0.001, 0.005, 0.01, 0.05, 0.1]:
    #     configs["adj_threshold"] = i
        # for semi_supervised_prop in [0.1, 0.2, 0.3]:
        train_mask = range(int(samples_num * semi_supervised_prop))
        val_mask = range(int(samples_num * semi_supervised_prop), int(samples_num * semi_supervised_prop) + 1000)
        test_mask = range(int(samples_num * semi_supervised_prop) + 1000, samples_num)
        train_mask = sample_mask(train_mask, samples_num)
        val_mask = sample_mask(val_mask, samples_num)
        test_mask = sample_mask(test_mask, samples_num)

        res.append(train_gcn(labels, train_mask, val_mask, test_mask, coefs, latents))
    return res


def train_IBSR():
    layer_units_num = [8]
    configs["n_hidden"] = layer_units_num
    configs["gcn_hidden"] = [[8], [8]]
    configs["dropout"] = 0
    configs["adj_threshold"] = 0.1
    configs["early_stopping"] = 100

    index = 1
    supervoxel_num = 5
    data =np.load("D:\CVPR2021\data\IBSR18\sv_features\sv_{}000\IBSR_0{}_processed_histogram.npy".format(supervoxel_num, index))
    tensor = np.load("D:\CVPR2021\data\IBSR18\sv_features\sv_{}000\IBSR_0{}_processed_histogram_eigen.npy".format(supervoxel_num, index))
    data = np.hstack((data, tensor))
    labels = np.load("D:\CVPR2021\data\IBSR18\sv_labels\sv_{}000\IBSR_0{}_svlabel.npy".format(supervoxel_num, index))
    labels = np.eye(3)[labels - 1]
    data = preprocess_features(data)
    coefs, latents = train_dsc_model(data)

    samples_num = data.shape[0]
    val_nums = supervoxel_num * 100
    res = list()
    for i in range(10):
        r = list()
        for semi_supervised_prop in [0.1, 0.2, 0.3]:
            train_mask = range(int(samples_num * semi_supervised_prop))
            val_mask = range(int(samples_num * semi_supervised_prop), int(samples_num * semi_supervised_prop) + val_nums)
            test_mask = range(int(samples_num * semi_supervised_prop) + val_nums, samples_num)
            train_mask = sample_mask(train_mask, samples_num)
            val_mask = sample_mask(val_mask, samples_num)
            test_mask = sample_mask(test_mask, samples_num)

            r.append(train_gcn(labels, train_mask, val_mask, test_mask, coefs, latents))
        res.append(r)
        for i in range(3):
            print("0.{} : {:.2%}".format(i + 1, r[i]))
    return res


def train_BrainWeb20():
    layer_units_num = [8]
    configs["n_hidden"] = layer_units_num
    configs["gcn_hidden"] = [[8], [8]]
    configs["dropout"] = 0
    configs["adj_threshold"] = 0.001
    configs["early_stopping"] = 100
    configs["gcn_learning_rate"] = 0.005

    index = 4
    data =np.load(".\data\BrainWeb20\sv_features\sv_9000\subject0{}_processed_histogram.npy".format(index))
    labels = np.load(".\data\BrainWeb20\sv_labels\sv_9000\subject0{}_svlabel.npy".format(index))
    labels = np.eye(3)[labels - 1]
    data = preprocess_features(data)
    coefs, latents = train_dsc_model(data)
    # adj = np.load(".\BrainWeb20_neighbour_9.npy")
    # graph = [adj]
    # graph.append(adj)

    samples_num = data.shape[0]
    res = list()
    for semi_supervised_prop in [0.1]:
        train_mask = range(int(samples_num * semi_supervised_prop))
        val_mask = range(int(samples_num * semi_supervised_prop), int(samples_num * semi_supervised_prop) + 900)
        test_mask = range(int(samples_num * semi_supervised_prop) + 900, samples_num)
        train_mask = sample_mask(train_mask, samples_num)
        val_mask = sample_mask(val_mask, samples_num)
        test_mask = sample_mask(test_mask, samples_num)

        res.append(train_gcn(labels, train_mask, val_mask, test_mask, coefs, latents))
    return res



# 计算平均值和方差
def result_output(res):
    res = np.array(res)
    mean_val = np.mean(res, axis=0)
    dev = np.std(res, axis=0)
    return mean_val, dev


if __name__ == '__main__':
    with open("./configs.json", "r", encoding="utf-8") as fp:
        configs = json.load(fp)
    # with open("./results.txt", "a+") as f:
    #     f.write("================== mnist try ================\n")
    #     res = list()
    #     for i in range(1):
    #         print("================ NO.{} ================".format(i))
    #         res.append(train_mnist())
    #         # res.append(train_scene15())
    #         # train_cifar10()
    #         # res.append(train_SVHN())
    #         print("semi-supervised-prop - accuracy=======>")
    #         for j in range(3):
    #             print("0.{} : {:.2%}".format(j+1, res[i][j]))
    #             f.write("{:.2%}".format(res[i][j]))
    #         f.write("\n")
    #     mean_val, dev = result_output(res)
    #     for i in range(3):
    #         f.write("{:.2%}+-{:.2%}".format(mean_val[i], dev[i]))
    #     # res = train_mnist()
    #     # res = train_SVHN()
    #     # for i in range(2):
    #     #     f.write("{:.2%}\t".format(res[i]))
    #
    #     # f.write("================== mnist layers units ablation ================\n")
    #     # res = train_mnist()
    #     # for i in range(5):
    #     #     f.write("{:.2%}\t".format(res[i]))

    # # res = train_BrainWeb20()
    # res = train_IBSR()
    # with open("./results.txt", "a+") as f:
    #     f.write("IBSR18")
    #     print("semi-supervised-prop - accuracy=======>")
    #     for i in range(10):
    #         for j in range(3):
    #             print("0.{} : {:.2%}".format(j+1, res[i][j]))
    #             f.write("{:.2%}\t".format(res[i][j]))
    #         f.write("\n")
    #     mean_val, dev = result_output(res)
    #     for i in range(3):
    #         f.write("{:.2%}+-{:.2%}".format(mean_val[i], dev[i]))
    train_scene15()








