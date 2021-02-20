# encoding=utf-8
# _author = "Steven gao"

from utils import *
from dsc.dsc import ResDSC
from gcn.gcn_model import GCN
from gcn.multi_gcn import MGCN
from tensorflow.examples.tutorials.mnist import input_data
import networkx as nx
import json
import tensorflow as tf
import pickle as pkl
import sys


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/Citeseer/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/Citeseer/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def train_dsc_model(data):
    configs["n_input"] = data.shape[1]
    tf.reset_default_graph()
    model = ResDSC(configs, data.shape)
    # model.restore()

    iteration = 0
    pretrained_step = 2000
    train_step = 10
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
    # model.save_model("cora", pretrained_step, configs["n_hidden"][-1])

    coefs = None
    latents = None
    for i in range(train_step):
        cost, coefs, latents = model.partial_fit(batch_x, is_training)
        if i % display_step == 0:
            print("loss: {}".format(cost))
    model.sess.close()
    return coefs, latents


def train_gcn(adj, y_train, y_val, y_test, train_mask, val_mask, test_mask, coefs, latents):
    epochs = 200
    samples_num = coefs[0].shape[0]

    supports = list()
    features = list()
    num_supports = len(configs["n_hidden"]) + 1
    for coef in coefs:
        # c = coef_to_adj(coef, 0.9)
        # # c = 0.5 * (np.abs(coef) + np.abs(coef.T))
        # # c = adj_softmax(c)
        # c = c * (adj.toarray())
        supports.append(adj.toarray())
    for latent in latents:
        features.append(preprocess_features(latent))

    # Define placeholders
    placeholders = {
        # 'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        # 自学习邻居，不使用稀疏（对应gcn_layers需要修改）
        'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
        'features': [tf.placeholder(tf.float32) for _ in range(num_supports)],
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=())
        # helper variable for sparse dropout
        # 'num_features_nonzero': [tf.placeholder(tf.int32) for _ in range(num_supports)]
    }

    # model = GCN(placeholders, samples_num, configs, input_dim=latents[0].shape[1], logging=False)

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
        feed_dict = construct_feed_dict(features, supports, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: configs["dropout"]})
        outs = sess.run([model.opt_op, model.loss, model.accuracy_mask, model.learning_supports], feed_dict=feed_dict)

        feed_dict_val = construct_feed_dict(features, supports, y_val, val_mask, placeholders)
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
    feed_dict_test = construct_feed_dict(features, supports, y_test, test_mask, placeholders)
    outs_test = sess.run([model.loss, model.accuracy_mask], feed_dict=feed_dict_test)
    print("Test set results:", "cost=", "{:.5f}".format(outs_test[0]),
          "accuracy=", "{:.5f}".format(outs_test[1]))
    print("Optimization Finished!")
    sess.close()


if __name__ == '__main__':
    with open("./configs.json", "r", encoding="utf-8") as fp:
        configs = json.load(fp)

    # ====================load data=====================
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("cora")
    features = preprocess_features(features)
    features = features.toarray()
    # layers_units_num = [512, 256, 64, 16]
    # layers_units_num = [256, 128, 16]
    # layers_units_num = [128, 16]
    # layers_units_num = [16]
    # configs["n_hidden"] = layers_units_num
    # configs["gcn_hidden"] = layers_units_num

    layers_units_num = [1024]
    configs["n_hidden"] = layers_units_num
    configs["gcn_hidden"] = [[16], [16]]

    coefs, latents = train_dsc_model(features)

    train_gcn(adj, y_train, y_val, y_test, train_mask,
              val_mask, test_mask, coefs, latents)



