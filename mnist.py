# encoding=utf-8
# _author = "Steven gao"

from utils import *
from dsc.dsc import ResDSC
from gcn.gcn_model import GCN
from gcn.multi_gcn import MGCN
from tensorflow.examples.tutorials.mnist import input_data
import json
import tensorflow as tf


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
    epochs = 5000
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


def train_mnist():
    layer_units_num = [256, 128]
    configs["n_hidden"] = layer_units_num
    configs["gcn_hidden"] = [[32], [32], [32]]
    configs["dropout"] = 0
    configs["adj_threshold"] = 0.003
    configs["early_stopping"] = 100
    # ====================load data=====================
    mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)
    x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    x_train = x_train[:7000, :]
    y_train = y_train[:7000, :]

    # x_train = preprocess_features(x_train)
    coefs, latents = train_dsc_model(x_train)

    samples_num = x_train.shape[0]
    semi_supervised_prop = 0.1
    train_mask = range(int(samples_num * semi_supervised_prop))
    val_mask = range(int(samples_num * semi_supervised_prop), int(samples_num * semi_supervised_prop) + 500)
    test_mask = range(int(samples_num * semi_supervised_prop) + 500, samples_num)
    train_mask = sample_mask(train_mask, samples_num)
    val_mask = sample_mask(val_mask, samples_num)
    test_mask = sample_mask(test_mask, samples_num)
    train_gcn(y_train, train_mask, val_mask, test_mask, coefs, latents)


def train_scene15():
    layer_units_num = [512, 128]
    configs["n_hidden"] = layer_units_num
    configs["gcn_hidden"] = [[64], [64], [64]]
    configs["dropout"] = 0
    configs["adj_threshold"] = 0.0
    configs["early_stopping"] = 200
    # ====================load data=====================
    data = np.transpose(sio.loadmat("./data/scene15/scene15.mat")["featureMat"])
    labels = np.transpose(sio.loadmat("./data/scene15/scene15.mat")["labelMat"])

    # random shuffle
    indices = np.random.permutation(data.shape[0])
    data = data[indices]
    labels = labels[indices]

    samples_num = data.shape[0]
    semi_supervised_prop = 0.1
    train_mask = range(int(samples_num * semi_supervised_prop))
    val_mask = range(int(samples_num * semi_supervised_prop), int(samples_num * semi_supervised_prop) + 500)
    test_mask = range(int(samples_num * semi_supervised_prop) + 500, samples_num)
    train_mask = sample_mask(train_mask, samples_num)
    val_mask = sample_mask(val_mask, samples_num)
    test_mask = sample_mask(test_mask, samples_num)

    # data = preprocess_features(data)
    coefs, latents = train_dsc_model(data)

    train_gcn(labels, train_mask, val_mask, test_mask, coefs, latents)


if __name__ == '__main__':
    with open("./configs.json", "r", encoding="utf-8") as fp:
        configs = json.load(fp)
    train_mnist()
    # train_scene15()





