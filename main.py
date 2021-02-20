# encoding=utf-8
# _author = "Steven gao"

from utils import *
from dsc.dsc import ResDSC
from gcn.gcn_model import GCN
from gcn.multi_gcn import MGCN
import json
import tensorflow as tf


with open("./configs.json", "r", encoding="utf-8") as fp:
    configs = json.load(fp)
layer_units_num = [256, 32]
configs["n_hidden"] = layer_units_num
configs["gcn_hidden"] = [[256, 16], [16], [16]]
configs["dropout"] = 0
restore_model_path = "./pretrain_model/hcmdd_latent_256_32_iteration_5000.ckpt"


def train_dsc_model(data):
    configs["n_input"] = data.shape[1]
    tf.reset_default_graph()
    model = ResDSC(configs, data.shape)
    model.restore(restore_model_path)

    iteration = 0
    pretrained_step = 800
    train_step = 2000
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
    # model.save_model("hcmdd", pretrained_step, configs["n_hidden"])

    coefs = None
    latents = None
    for i in range(train_step):
        cost, coefs, latents = model.partial_fit(batch_x, is_training)
        if i % display_step == 0:
            print("loss: {}".format(cost))
    model.sess.close()
    return coefs, latents


def train_gcn(x, labels, train_mask, test_mask, coefs, latents):
    epochs = 300
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
        # 自学习邻居，不使用稀疏（对应gcn_layers需要修改）
        # 'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
        'features': [tf.placeholder(tf.float32) for _ in range(num_supports)],
        'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=())
        # helper variable for sparse dropout
        # 'num_features_nonzero': [tf.placeholder(tf.int32) for _ in range(num_supports)]
    }

    feed_dict = construct_feed_dict(features, supports, labels, train_mask, placeholders)
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

    cost_list = list()
    for epoch in range(epochs):
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy_mask, model.accuracy], feed_dict=feed_dict)
        cost_list.append(outs[1])

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "all_acc=", "{:.5f}".format(outs[3] / labels.shape[0]))
        if epoch > configs["early_stopping"] and cost_list[-1] > np.mean(cost_list[-(configs["early_stopping"] + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")
    # model.save(sess)

    feed_dict = construct_feed_dict(features, supports, labels, all_mask, placeholders)
    outs = sess.run([model.opt_op, model.loss, model.accuracy_mask, model.outputs], feed_dict=feed_dict)
    spe, sen = calculate_index(labels, outs[3])
    print("acc={:.4f}, sen={:.4f}, spe={:.4f}".format(outs[2], sen, spe))
    sess.close()


if __name__ == '__main__':
    print("load data ... ")
    data = sio.loadmat("./data/data.mat")["data"]
    x = data[:, :-2]
    y = data[:, -2:]
    train_num = 218
    semi_supervised_prop = 0.1
    train_mask = range(int(train_num * semi_supervised_prop))
    test_mask = range(int(train_num * semi_supervised_prop), train_num)
    train_mask = sample_mask(train_mask, train_num)
    test_mask = sample_mask(test_mask, train_num)

    all_mask = range(train_num)
    all_mask = sample_mask(all_mask, train_num)

    # data_tuple = load_significant_region_data(4)
    # x = data_tuple[0]
    # y = data_tuple[1]
    # train_mask = data_tuple[2]

    x = preprocess_features(x)
    coefs, latents = train_dsc_model(x)
    train_gcn(x, y, train_mask, test_mask, coefs, latents)


