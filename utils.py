import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn import cluster
import tensorflow as tf
from scipy.spatial.distance import cdist
from scipy import sparse


def load_significant_region_data(flag=4, mask_prop=0.1):
    """
    load significant region from WM、GM、WM-GM with p < 0.05
    flag: 1 ==>  WW
    flag: 2 ==>  GG
    flag: 3 ==> WG
    flag: others ==> whole brain
    :return:
    """
    mdd_subjects = 129
    hc_subjects = 89
    subj_num = mdd_subjects + hc_subjects
    if 1 == flag:
        features = sio.loadmat("./data/WM_SIG_DATA.mat")["wm_sig_data"]
    elif 2 == flag:
        features = sio.loadmat("./data/GM_SIG_DATA.mat")["gm_sig_data"]
    elif 3 == flag:
        features = sio.loadmat("./data/WMGM_SIG_DATA.mat")["wmgm_sig_data"]
    else:
        wm = sio.loadmat("./data/WM_SIG_DATA.mat")["wm_sig_data"]
        gm = sio.loadmat("./data/GM_SIG_DATA.mat")["gm_sig_data"]
        wg = sio.loadmat("./data/WMGM_SIG_DATA.mat")["wmgm_sig_data"]
        all_features_num = wm.shape[1] + gm.shape[1] + wg.shape[1]
        features = np.zeros((subj_num, all_features_num))
        features[:, :wm.shape[1]] = wm
        features[:, wm.shape[1]: (wm.shape[1] + gm.shape[1])] = gm
        features[:, (wm.shape[1] + gm.shape[1]):] = wg

    # hc label : [1, 0]、mdd label : [0, 1]
    labels = np.zeros((subj_num, 2), dtype=np.int)
    labels[:mdd_subjects, 1] = 1
    labels[mdd_subjects:, 0] = 1
    # random shuffle
    indices = np.random.permutation(features.shape[0])
    data_x = features[indices]
    # data_x = preprocessing.scale(data_x)
    data_y = labels[indices]

    data = np.zeros((subj_num, all_features_num+2))
    data[:, :-2] = data_x
    data[:, -2:] = data_y
    sio.savemat("./data/data1.mat", {"data": data})

    train_num = int(subj_num)
    train_mask = range(int(train_num * mask_prop))
    train_mask = sample_mask(train_mask, train_num)

    return data_x, data_y, train_mask


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)
    return grp, L


def err_rate(labels, p_labels):
    # test_mask = (: is False)
    predict_1 = np.zeros(labels.shape, dtype=np.int)
    predict_2 = np.zeros(labels.shape, dtype=np.int)
    for i, index in enumerate(p_labels):
        predict_1[i, index] = 1
        predict_2[i, 1-index] = 1
    err_x_1 = np.sum(labels[:, 0] != predict_1[:, 0])
    err_x_2 = np.sum(labels[:, 0] != predict_2[:, 0])
    err_x = err_x_1 if err_x_1 < err_x_2 else err_x_2
    if err_x_1 < err_x_2:
        sen = np.sum(
            predict_1[predict_1[:, 1] == labels[:, 1], 1] == 1) / np.sum(
            labels[:, 1] == 1) if np.sum(labels[:, 1] == 1) is not 0 else 0
        spe = np.sum(
            predict_1[predict_1[:, 0] == labels[:, 0], 1] == 0) / np.sum(
            labels[:, 0] == 1)if np.sum(labels[:, 0] == 1) is not 0 else 0
    else:
        sen = np.sum(
            predict_2[predict_2[:, 1] == labels[:, 1], 1] == 1) / np.sum(
            labels[:, 1] == 1) if np.sum(labels[:, 1] == 1) is not 0 else 0
        spe = np.sum(
            predict_2[predict_2[:, 0] == labels[:, 0], 1] == 0) / np.sum(
            labels[:, 0] == 1) if np.sum(labels[:, 0] == 1) is not 0 else 0
    mis_rate = err_x.astype(float) / (labels.shape[0])
    return mis_rate, sen, spe


# ====================================== GCN =====================================
def coef_to_adj(coef, threshold):
    """convert Coef matrix to adjacency csr_matrix"""
    # coef = 0.5 * (np.abs(coef) + np.abs(coef.T))
    # return coef

    coef = 0.5 * (coef + coef.T)
    # top_threshold = int(coef.shape[0] ** 2 * threshold)
    # top_index = np.unravel_index(np.argsort(coef.ravel())[-top_threshold:], coef.shape)
    # top_val = coef[top_index][0]
    # adj = np.where(np.abs(coef) > top_val, 1, 0)
    samples_number = coef.shape[0]
    neighbor_number = round(threshold*samples_number)
    adj = np.zeros((samples_number, samples_number), dtype=np.int)
    for i in range(samples_number):
        top_index = np.argsort(np.abs(coef[i]))[::-1][0: neighbor_number]
        adj[i][top_index] = 1
    # adj = adj - np.diag(adj.diagonal())
    # adj = sp.csr_matrix(adj)
    return adj


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def adj_softmax(x):
    """ soft-max function """
    # x -= np.max(x, axis=1, keepdims=True)  # avoid numerical overflow
    # x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    x = x / np.sum(x, axis=1, keepdims=True)
    return x


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0], dtype="float32"))
    a = adj_normalized.toarray()
    return sparse_to_tuple(adj_normalized)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features'][i]: features[i] for i in range(len(features))})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    # feed_dict.update({placeholders['num_features_nonzero'][i]: features[i][2] for i in range(len(features))})
    return feed_dict


# ========================================= GAT =======================================
def adj_to_bias(adj, sizes, nhood=1):
    """
     Prepare adjacency matrix by expanding up to a given neighbourhood.
     This will insert loops on every node.
     Finally, the matrix is converted to bias vectors.
     Expected shape: [graph, nodes, nodes]
    """
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


# ========================================= KNN graph construction =======================================
def knn_graph_construction(datasets, k_prop):
    samples_num = datasets.shape[0]
    # distance = cdist(datasets, datasets, metric='euclidean')
    # nums = [0.001, 0.002, 0.005, 0.01]
    # neighbour_nums = [x*samples_num for x in nums]
    # # neighbour_nums = [10]
    # for neighbour_num in neighbour_nums:
    #     neighbour_num = round(neighbour_num)
    #     dis = tf.placeholder(dtype=tf.float32, shape=distance.shape)
    #     indices = tf.nn.top_k(dis * -1, neighbour_num)
    #     with tf.Session() as sess:
    #         index = sess.run(indices, feed_dict={dis: distance}).indices
    #     adj = np.zeros((samples_num, samples_num), dtype=np.int)
    #     for i in range(samples_num):
    #         adj[i][index[i]] = 1
    #     print("{} is saving".format(neighbour_num))
    #     np.save("BrainWeb20_neighbour_{}.npy".format(neighbour_num), adj)

    neighbour_num = round(samples_num * k_prop)
    distance = cdist(datasets, datasets, metric='euclidean')
    dis = tf.placeholder(dtype=tf.float32, shape=distance.shape)
    indices = tf.nn.top_k(dis * -1, neighbour_num)
    with tf.Session() as sess:
        index = sess.run(indices, feed_dict={dis: distance}).indices
    adj = np.zeros((samples_num, samples_num), dtype=np.int)
    for i in range(samples_num):
        adj[i][index[i]] = 1
    return adj


def calculate_index(labels, predicts):
    """
    计算评价指标，特异性，敏感性
    :param labels:
    :param predicts:
    :return:
    """
    predicts = np.argmax(predicts, axis=1)
    labels = np.argmax(labels, axis=1)
    matrix = confusion_matrix(labels, predicts)
    spe = matrix[0][0] * 1.0 / (matrix[0][0] + matrix[0][1])
    sen = matrix[1][1] * 1.0 / (matrix[1][0] + matrix[1][1])
    return spe, sen


if __name__ == "__main__":
    # # load_significant_region_data(4)
    # # data = sio.loadmat("./data/data1.mat")["data"]
    # # x = data[:, :-2]
    # # y = data[:, -2:]
    # # print(sum(y[:22, 0] == 1) / 22)
    # # print(sum(y[:22, 1] == 1) / 22)
    # index = 1
    # data = np.load("D:\CVPR2021\data\IBSR18\sv_features\sv_5000\IBSR_0{}_processed_histogram.npy".format(index))
    # print(knn_graph_construction(data, 10))

    # a = np.load("BrainWeb20_neighbour_10.npy")
    # print(a.shape)

    supervoxel_num = 5
    index = 1
    k_neighbour = 0.1
    data = np.load(
        "D:\CVPR2021\data\IBSR18\sv_features\sv_{}000\IBSR_0{}_processed_histogram.npy".format(supervoxel_num, index))
    tensor = np.load(
        "D:\CVPR2021\data\IBSR18\sv_features\sv_{}000\IBSR_0{}_processed_histogram_eigen.npy".format(supervoxel_num,
                                                                                                     index))
    data = np.hstack((data, tensor))
    adj = knn_graph_construction(data, k_neighbour)
    adj = sparse.csc_matrix(adj)

    labels = np.load("D:\CVPR2021\data\IBSR18\sv_labels\sv_{}000\IBSR_0{}_svlabel.npy".format(supervoxel_num, index))
    labels = np.eye(3)[labels - 1]
    labels = sparse.csc_matrix(labels)
    sio.savemat("IBSR18_5000_neighbour.mat", {"network": adj, "group": labels})


