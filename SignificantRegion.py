from tensorflow.python import pywrap_tensorflow
import numpy as np
import pandas as pd

# 读取dsc模型
dsc_model_reader = pywrap_tensorflow.NewCheckpointReader(r"./model/hcmdd_256_32_iteration_800.ckpt")

# dict形式
encoder_w_0 = np.array(dsc_model_reader.get_tensor("encoder_w0"))
regions_connection_value = np.sum(np.abs(encoder_w_0), axis=1)

# 读取MGCN
mgcn_model_reader = pywrap_tensorflow.NewCheckpointReader(r"./model/mgcn.ckpt")
# dict形式
gcn_w_0 = np.array(mgcn_model_reader.get_tensor("mgcn/graphconvolution_1_vars/weights_0:0"))
first_layer_weight = np.sum(np.abs(gcn_w_0), axis=1)

regions_connection_value = regions_connection_value + first_layer_weight

# 选取具有高度辨别能力的大脑区域连接
regions_connection_name = np.array(pd.read_excel("significant_connection_name_733.xlsx").columns)
index = np.argsort(regions_connection_value)[-10:][::-1]
data_dict = {
    "name": regions_connection_name[index],
    "value": regions_connection_value[index],
    "weight": regions_connection_value[index] / np.sum(regions_connection_value)
}
significant_data = pd.DataFrame(data_dict)
significant_data.to_excel("significant_results.xlsx")




a = np.ones((130, 130), dtype=np.int)
a = a * -1
a[129][42] = a[42][129] = 1
a[113][34] = a[34][112] = 1
a[96][33] = a[33][96] = 1
a[93][21] = a[21][93] = 1
a[95][69] = a[69][95] = 1
a[107][42] = a[42][107] = 1
a[115][62] = a[62][115] = 1
a[96][32] = a[32][96] = 1
a[113][40] = a[40][113] = 1
a[122][29] = a[29][122] = 1

b = pd.DataFrame(a)
b.to_excel("significant_edges.xlsx")