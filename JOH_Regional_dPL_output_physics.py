import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras.models import Model
from keras.layers import Input, Concatenate
from keras import optimizers, callbacks
from datetime import datetime, timedelta

import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation, LSTM
import keras.backend as K
import tensorflow as tf

from dPL_class import regional_DifferentiableEXPHYDRO_Hu, LSTM_postprocess, ScaleLayer
from dataprocess import DataforIndividual
import loss

## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'

working_path = '/work/home/acyrys0cgr/Code/hydro_dl_project'
attrs_path = '/work/home/acyrys0cgr/Code/hydro_dl_project/camels_data/datafor_rm/531_27attrs_scal.csv'

testing_start = '1980-10-01'
testing_end = '2010-09-30'

basin_id = ['1022500'
]


print(len(basin_id))



def create_model(input_xd_shape, input_xd_shape1, hodes, seed):
    xd_input_forprnn = Input(shape=input_xd_shape, batch_size=1, name='Input_xd')  # [9,3288,5]
    xd_input_forprnn1 = Input(shape=input_xd_shape1, batch_size=1, name='Input_xd1')  # [9,3288,5]
    # xd_input_forconnect = Input(shape=input_xd2_shape, batch_size=9, name='Input_xd2')  #[9,3288,5]
    # xs_input = Input(shape=input_xs_shape, batch_size=15, name='Input_xs')          #[9,27]

    hydro_output = regional_DifferentiableEXPHYDRO_Hu(mode='normal', h_nodes = hodes, seed = seed, name='Regional_dPL_LSTM')(xd_input_forprnn)
    print("hydro_output", hydro_output)  # [60,2218,4]

    xd_hydro = Concatenate(axis=-1, name='Concat')([xd_input_forprnn1, hydro_output])
    #xd_hydro_scale = ScaleLayer(name='Scale')(xd_hydro)
    #print("xd_hydro_scale",xd_hydro_scale)

    ealstm_hn, ealstm_cn = LSTM_postprocess(input_xd = 33, hidden_size=256, seed=seed, name='LSTM')(xd_hydro)
    fc2_out = Dense(units=1)(ealstm_hn)

    fc2_out = K.permute_dimensions(fc2_out, pattern=(1,0,2))  # for test model


    model = Model(inputs=[xd_input_forprnn,xd_input_forprnn1], outputs=fc2_out)
    return model


def stest_model(model, test_x,test_x1, save_path):
    model.load_weights(save_path, by_name=True)
    pred_y = model.predict(x=[test_x,test_x1], batch_size=1)
    print("pred_y",pred_y.shape)
    return pred_y



def nse_metrics(y_true, y_pred):
    y_true = K.constant(y_true)
    #y_pred = y_pred  # Omit values in the spinup period (the first 365 days)


    y_true = y_true[:, :, :]  # Omit values in the spinup period (the first 365 days)  [150,3288,1]
    y_pred = y_pred[:, :, :]  # Omit values in the spinup period (the first 365 days)
    print("y_true_shape:", y_true.shape)
    print("y_pred_shape:", y_pred.shape)

    numerator = K.sum(K.square(y_true - y_pred), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)
    rNSE = numerator / (denominator+0.1)

    return 1.0 - rNSE

def generate_train_test(test_set, test_set1):

    test_set_ = pd.DataFrame(test_set)
    test_x_np = test_set_.values[:, :-1]


    train_set1_ = pd.DataFrame(test_set1)
    train_x_np1 = train_set1_.values[:, :-1]
    train_x_np1[:,0:1] = (train_x_np1[:,0:1] - 3.324721383475159)/7.743396135805888
    train_x_np1[:,1:2] = (train_x_np1[:,1:2] - 10.21469801905563)/10.470459018014589
    train_x_np1[:,2:3] = (train_x_np1[:,2:3] - 0.4999793666778481)/0.08243191435032118
    train_x_np1[:,3:4] = (train_x_np1[:,3:4] - 341.70740793615124)/133.2828108595638
    train_x_np1[:,4:5] = (train_x_np1[:,4:5] - 954.7454854295064)/652.4717499708983


    test_y_np = test_set_.values[:, -1:]

    #tes_y_np1 = normalize(test_y_np)
    # test_x_np = test_set.values[:, :-1]
    # test_y_np = test_set.values[:, -1:]

    # wrap_number_train = (train_x_np.shape[0] - wrap_length) // 410 + 1

    # train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
    # train_y = np.empty(shape=(wrap_number_train, wrap_length, train_y_np.shape[1]))

    test_x = np.expand_dims(test_x_np, axis=0)
    test_x1 = np.expand_dims(train_x_np1, axis=0)
    test_y = np.expand_dims(test_y_np, axis=0)

    return test_x, test_x1, test_y





basin_list = []
test1_list = []
test2_list = []
batch_list = []
testx_list = []
testx1_list = []
testy_list = []
all_list = []
all_list1 = []
for i in range(len(basin_id)):

    a = basin_id[i]
    if len(basin_id[i]) == 7:
        basin_id[i] = '0' + basin_id[i]
        print(basin_id[i])


    hydrodata = DataforIndividual(working_path, basin_id[i]).load_data()

    # train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
    test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))]
    test_set1 = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))]
    # print(f"The training data set is from {training_start} to {training_end}, with a shape of {train_set.shape}")
    # print(f"The testing data set is from {testing_start} to {testing_end}, with a shape of {test_set.shape}")

    if a.startswith('0'):
        single_basin_id = a[1:]

    else:
        single_basin_id = a

    # print(single_basin_id)

    static_x = pd.read_csv(attrs_path)
    static_x = static_x.set_index('gauge_id')
    rows_bool = (static_x.index == int(single_basin_id))
    rows_list = [i for i, x in enumerate(rows_bool) if x]
    rows_int = int(rows_list[0])
    static_x_np = np.array(static_x)
    # print("static_x_np_shape:", static_x_np.shape)

    local_static_x = static_x_np[rows_int, :]  # basin_id index in attrs_path
    local_static_x_for_test = np.expand_dims(local_static_x, axis=0)
    local_static_x_for_test = local_static_x_for_test.repeat(test_set.shape[0], axis=0)
    # print("local_static_x_test:", local_static_x_for_test)
    print("local_static_x_test_shape:", local_static_x_for_test.shape)

    # local_static_x_for_train = np.expand_dims(local_static_x, axis=0)
    # local_static_x_for_train = local_static_x_for_train.repeat(train_set.shape[0], axis=0)

    # print("local_static_x_train_shape:", local_static_x_for_train.shape)
    # print(local_static_x_for_train[0,0])

    result = np.concatenate((test_set, local_static_x_for_test), axis=-1)
    result1 = np.concatenate((test_set1, local_static_x_for_test), axis=-1)


    print("result_shape:", result.shape)
    print("result1_shape:", result1.shape)

    sum_result = result[:,
                 [0, 1, 2, 3, 4, 32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
              28, 29, 30, 31, 5]]

    sum_result1 = result1[:,
                 [0, 1, 2, 3, 4, 32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
              28, 29, 30, 31, 5]]

    #nan_rows = np.isnan(sum_result).any(axis=1)
    #sum_result = sum_result[~nan_rows]

    print("sum_result_shape", sum_result.shape)
    print("sum_result1_shape", sum_result1.shape)


    test_x, test_x1, test_y = generate_train_test(sum_result,sum_result1)

    print(f'{test_x.shape}, {test_y.shape}')
    basin_list.append(basin_id[i])
    batch_list.append(test_x.shape[0])
    test1_list.append(test_x.shape[1])
    test2_list.append(test_x.shape[2])

    testx_list.append(test_x)
    testx1_list.append(test_x1)
    testy_list.append(test_y)

print(len(batch_list))
print(len(test1_list))
print(len(test2_list))
print(len(testx_list))
print(len(testx1_list))
print(len(testy_list))



nse_results = pd.DataFrame(columns=['Basin_ID', 'NSE_TEST'])



Path(f'{working_path}/results').mkdir(parents=True, exist_ok=True)
save_path_dPL_LSTM= f'{working_path}/results/global/dPL_LSTM_train15years_531camels_A800.h5'
model = create_model((10957, 32), (10957, 32),hodes=256, seed=200)
model.summary()

representation_model = Model(inputs=model.inputs, outputs=model.get_layer('Regional_dPL_LSTM').output)



for j in range(len(testx_list)):

    dense_hydro_output = representation_model.predict([test_x, test_x1])
    print(dense_hydro_output.shape)
    print(type(dense_hydro_output))

    for i in range(10957):
        snowmelt_days = K.eval(dense_hydro_output[:, i, -1])
        print(float(snowmelt_days))




