from gcn.gcn_layers import *
from gcn.gcn_inits import masked_softmax_cross_entropy, masked_accuracy
import tensorflow as tf


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.dsc_layers_nums = None
        self.outputs = 0

        self.epsilon = 0

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.saver = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        for i in range(self.dsc_layers_nums):
            self.activations.append(self.inputs[i])
            for index in range(len(self.layers[i]) - 1):
                hidden = self.layers[i][index](self.activations[-1])
                self.activations.append(hidden)
            self.activations.append(self.layers[i][-1](self.activations[-1]))
            self.outputs = self.outputs + self.activations[-1]
            # self.outputs = self.activations[-1]
            self.activations.clear()
        self.outputs = self.outputs / self.dsc_layers_nums

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "./model/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MGCN(Model):
    def __init__(self, placeholders, samples_num, configs, input_dim,  **kwargs):
        super(MGCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.dsc_layers_nums = len(self.inputs)
        self.input_dim = input_dim
        self.epsilon = configs["epsilon"]
        self.gcn_hidden = configs["gcn_hidden"]
        self.weight_decay = configs["weight_decay"]
        self.dropout = placeholders["dropout"]
        self.supports = placeholders["support"]
        self.learning_supports = placeholders["support"]
        # self.learning_supports = []
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.samples_num = samples_num
        self.placeholders = placeholders

        self.learning_neighbour_weights = {}

        self.optimizer = tf.train.AdamOptimizer(learning_rate=configs["gcn_learning_rate"])
        self.build()

    def _loss(self):
        # # self-learning neighbour weights loss
        # for k in range(self.dsc_layers_nums):
        #     self.loss += tf.nn.l2_loss(self.learning_supports[k])
        #     # for i in range(self.samples_num):
        #     #     for j in range(self.samples_num):
        #     #         self.loss += 5e-7 * self.supports[k][i][j] * tf.reduce_sum(
        #     #             tf.square(tf.subtract(self.inputs[-1][i, :], self.inputs[-1][j, :])))

        # Weight decay loss
        for i in range(self.dsc_layers_nums):
            for j in range(len(self.layers[i]) - 1):
                for var in self.layers[i][j].vars.values():
                    self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy_mask, self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                                            self.placeholders['labels_mask'])
        # self.accuracy_mask = masked_accuracy(self.outputs, self.placeholders['labels'],
        #                                      self.placeholders['labels_mask'])

    def _build(self):
        for i in range(self.dsc_layers_nums):
            # # 自学习邻居
            # with tf.variable_scope("self_learning_neighbour"):
            #     self.learning_neighbour_weights["weights_{}".format(i)] = ones([self.samples_num, self.samples_num])
            #     tmp = tf.nn.relu(tf.multiply(self.learning_neighbour_weights["weights_{}".format(i)], self.supports[i]))
            #     divisor = tf.clip_by_value(tf.reduce_sum(tmp, axis=1, keep_dims=True), 0.0001, self.samples_num)
            #     tmp /= divisor
            #     self.learning_supports.append(tmp)
            each_layer = list()
            each_layer.append(GraphConvolution(input_dim=self.input_dim[i],
                                               output_dim=self.gcn_hidden[i][0],
                                               support=self.learning_supports[i],
                                               act=tf.nn.relu,
                                               dropout=self.dropout,
                                               logging=self.logging))
            for j in range(len(self.gcn_hidden[i]) - 1):
                each_layer.append(GraphConvolution(input_dim=self.gcn_hidden[i][j],
                                                   output_dim=self.gcn_hidden[i][j + 1],
                                                   support=self.learning_supports[i],
                                                   act=tf.nn.relu,
                                                   dropout=self.dropout,
                                                   logging=self.logging))

            each_layer.append(GraphConvolution(input_dim=self.gcn_hidden[i][-1],
                                               output_dim=self.output_dim,
                                               support=self.learning_supports[i],
                                               act=lambda x: x,
                                               dropout=self.dropout,
                                               logging=self.logging))
            self.layers.append(each_layer)

    def predict(self):
        return tf.nn.softmax(self.outputs)

