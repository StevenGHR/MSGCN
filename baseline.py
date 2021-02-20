# encoding=utf-8
# _author = "Steven gao"

from utils import *
import tensorflow as tf
from gcn.gcn_model import GCN
from gat.gat import GAT

def load_BrainWeb20(supervoxel_num, k_neighbour):
    index = 4
    data = np.load(".\data\BrainWeb20\sv_features\sv_{}000\subject0{}_processed_histogram.npy".format(supervoxel_num, index))
    adj = knn_graph_construction(data, k_neighbour)
    # adj = np.load("./BrainWeb20_neighbour_{}.npy".format(k_neighbour))
    labels = np.load(".\data\BrainWeb20\sv_labels\sv_{}000\subject0{}_svlabel.npy".format(supervoxel_num, index))
    labels = np.eye(3)[labels - 1]
    data = preprocess_features(data)

    return data, adj, labels

def load_IBSR18(supervoxel_num, k_neighbour):
    index = 1
    data = np.load("D:\CVPR2021\data\IBSR18\sv_features\sv_{}000\IBSR_0{}_processed_histogram.npy".format(supervoxel_num, index))
    tensor = np.load("D:\CVPR2021\data\IBSR18\sv_features\sv_{}000\IBSR_0{}_processed_histogram_eigen.npy".format(supervoxel_num, index))
    data = np.hstack((data, tensor))
    adj = knn_graph_construction(data, k_neighbour)
    labels = np.load("D:\CVPR2021\data\IBSR18\sv_labels\sv_{}000\IBSR_0{}_svlabel.npy".format(supervoxel_num, index))
    labels = np.eye(3)[labels - 1]
    data = preprocess_features(data)

    return data, adj, labels


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    # feed_dict.update({placeholders['num_features_nonzero'][i]: features[i][2] for i in range(len(features))})
    return feed_dict


def gcn(data, adj, labels, semi_supervised_prop):
    configs = dict()
    configs["gcn_learning_rate"] = 0.005
    configs["gcn_hidden"] = [8]
    configs["weight_decay"] = 5e-5
    configs["dropout"] = 0
    configs["early_stopping"] = 100
    epochs = 1000
    samples_num = labels.shape[0]
    supports = [preprocess_adj(adj)]
    num_supports = 1

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.placeholder(tf.float32),
        'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=())
        # helper variable for sparse dropout
        # 'num_features_nonzero': [tf.placeholder(tf.int32) for _ in range(num_supports)]
    }

    val_num = 500

    train_mask = range(int(samples_num * semi_supervised_prop))
    val_mask = range(int(samples_num * semi_supervised_prop), int(samples_num * semi_supervised_prop) + val_num)
    test_mask = range(int(samples_num * semi_supervised_prop) + val_num, samples_num)
    train_mask = sample_mask(train_mask, samples_num)
    val_mask = sample_mask(val_mask, samples_num)
    test_mask = sample_mask(test_mask, samples_num)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # 使用单纯GCN模型
    model = GCN(placeholders, configs, input_dim=data.shape[1], logging=False)

    # Initialize sess
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = list()
    for epoch in range(epochs):
        # Training step
        feed_dict = construct_feed_dict(data, supports, y_train, train_mask, placeholders)
        outs = sess.run([model.opt_op, model.loss, model.accuracy_mask], feed_dict=feed_dict)

        feed_dict_val = construct_feed_dict(data, supports, y_val, val_mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy_mask], feed_dict=feed_dict_val)
        cost_val.append(outs_val[0])

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(outs_val[0]),
              "val_acc=", "{:.5f}".format(outs_val[1]))

        if epoch > configs["early_stopping"] and cost_val[-1] > np.mean(cost_val[-(configs["early_stopping"] + 1):-1]):
            print("Early stopping...")
            break

    # test
    feed_dict_test = construct_feed_dict(data, supports, y_test, test_mask, placeholders)
    outs_test = sess.run([model.loss, model.accuracy_mask], feed_dict=feed_dict_test)
    print("Test set results:", "cost=", "{:.5f}".format(outs_test[0]),
          "accuracy=", "{:.5f}".format(outs_test[1]))
    print("Optimization Finished!")
    sess.close()
    return outs_test[1]


def gat(x, coef, labels, semi_supervised_prop):
    # training params
    batch_size = 1
    nb_epochs = 1000
    patience = 100
    lr = 0.005  # learning rate
    l2_coef = 0.0005  # weight decay
    hid_units = [8]  # numbers of hidden units per each attention head in each layer
    n_heads = [1, 1]  # additional entry for the output layer
    adj_threshold = 0.1
    residual = False
    nonlinearity = tf.nn.elu
    model = GAT

    nb_nodes = labels.shape[0]

    ft_size = x.shape[1]
    nb_classes = labels.shape[1]

    samples_num = labels.shape[0]
    val_number = 900

    train_mask = range(int(samples_num * semi_supervised_prop))
    # val_mask = range(int(samples_num * semi_supervised_prop), int(samples_num * semi_supervised_prop) + val_number)
    # test_mask = range(int(samples_num * semi_supervised_prop) + val_number, samples_num)
    val_mask = train_mask
    test_mask = range(samples_num)
    train_mask = sample_mask(train_mask, samples_num)
    val_mask = sample_mask(val_mask, samples_num)
    test_mask = sample_mask(test_mask, samples_num)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    features = x[np.newaxis]
    adj = coef[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    biases = adj_to_bias(adj, [nb_nodes], nhood=1)

    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
            bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
            lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
            msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
            attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            is_train = tf.placeholder(dtype=tf.bool, shape=())

        logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                 attn_drop, ffd_drop,
                                 bias_mat=bias_in,
                                 hid_units=hid_units, n_heads=n_heads,
                                 residual=residual, activation=nonlinearity)
        log_resh = tf.reshape(logits, [-1, nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])
        loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

        train_op = model.training(loss, lr, l2_coef)

        saver = tf.train.Saver()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        vlss_mn = np.inf
        vacc_mx = 0.0
        curr_step = 0

        with tf.Session() as sess:
            sess.run(init_op)

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

            for epoch in range(nb_epochs):
                tr_step = 0
                tr_size = features.shape[0]

                while tr_step * batch_size < tr_size:
                    _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                                                        feed_dict={
                                                            ftr_in: features[
                                                                    tr_step * batch_size:(tr_step + 1) * batch_size],
                                                            bias_in: biases[
                                                                     tr_step * batch_size:(tr_step + 1) * batch_size],
                                                            lbl_in: y_train[
                                                                    tr_step * batch_size:(tr_step + 1) * batch_size],
                                                            msk_in: train_mask[
                                                                    tr_step * batch_size:(tr_step + 1) * batch_size],
                                                            is_train: True,
                                                            attn_drop: 0.6, ffd_drop: 0.6})
                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1

                vl_step = 0
                vl_size = features.shape[0]

                while vl_step * batch_size < vl_size:
                    loss_value_vl, acc_vl = sess.run([loss, accuracy],
                                                     feed_dict={
                                                         ftr_in: features[
                                                                 vl_step * batch_size:(vl_step + 1) * batch_size],
                                                         bias_in: biases[
                                                                  vl_step * batch_size:(vl_step + 1) * batch_size],
                                                         lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                                                         msk_in: val_mask[
                                                                 vl_step * batch_size:(vl_step + 1) * batch_size],
                                                         is_train: False,
                                                         attn_drop: 0.0, ffd_drop: 0.0})
                    val_loss_avg += loss_value_vl
                    val_acc_avg += acc_vl
                    vl_step += 1

                print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                      (train_loss_avg / tr_step, train_acc_avg / tr_step,
                       val_loss_avg / vl_step, val_acc_avg / vl_step))

                if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                    if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                        vacc_early_model = val_acc_avg / vl_step
                        vlss_early_model = val_loss_avg / vl_step
                        # saver.save(sess, checkpt_file)
                    vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                    vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step == patience:
                        print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                        print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                        break

                train_loss_avg = 0
                train_acc_avg = 0
                val_loss_avg = 0
                val_acc_avg = 0

            # saver.restore(sess, checkpt_file)

            ts_size = features.shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0

            while ts_step * batch_size < ts_size:
                loss_value_ts, acc_ts = sess.run([loss, accuracy],
                                                 feed_dict={
                                                     ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                     bias_in: biases[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                     lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                     msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                     is_train: False,
                                                     attn_drop: 0.0, ffd_drop: 0.0})
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1

            print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)

            sess.close()


# 计算平均值和方差
def result_output(res):
    res = np.array(res)
    mean_val = np.mean(res, axis=0)
    dev = np.std(res, axis=0)
    return mean_val, dev


if __name__ == "__main__":
    K_neighbour = 0.1
    supervoxel_num = 5
    # data, adj, labels = load_BrainWeb20(supervoxel_num, K_neighbour)
    # data, adj, labels = load_IBSR18(supervoxel_num, K_neighbour)
    # res = list()
    # for i in range(10):
    #     r = list()
    #     for j in [0.1, 0.2, 0.3]:
    #         r.append(gcn(data, adj, labels, j))
    #         # r.append(gat(data, adj, labels, j))
    #     res.append(r)
    #
    # # gat(data, adj, labels)
    # with open("./results.txt", "a+") as f:
    #     f.write("IBSR18-GCN")
    #     print("semi-supervised-prop - accuracy=======>")
    #     for i in range(10):
    #         for j in range(3):
    #             print("0.{} : {:.2%}".format(j + 1, res[i][j]))
    #             f.write("{:.2%}\t".format(res[i][j]))
    #         f.write("\n")
    #     mean_val, dev = result_output(res)
    #     for i in range(3):
    #         f.write("{:.2%}+-{:.2%}".format(mean_val[i], dev[i]))
    data, labels, _ = load_significant_region_data()
    adj = knn_graph_construction(data, K_neighbour)
    print(gat(data, adj, labels, 0.1))

