import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation, LSTM
import keras.backend as K
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()


class regional_DifferentiableEXPHYDRO_Hu(Layer):

    def __init__(self, mode='normal', h_nodes=256, seed=200, **kwargs):
        self.mode = mode
        self.h_nodes = h_nodes
        self.seed = seed
        super(regional_DifferentiableEXPHYDRO_Hu, self).__init__(**kwargs)

    def build(self, input_shape):

        self.prnn_w1 = self.add_weight(name='prnn_w1',
                                       shape=(27, self.h_nodes),
                                       initializer=initializers.RandomUniform(seed=self.seed - 5),

                                       trainable=True)

        self.prnn_b1 = self.add_weight(name='prnn_b1',
                                       shape=(self.h_nodes,),
                                       initializer=initializers.zeros(),
                                       trainable=True)

        self.prnn_w2 = self.add_weight(name='prnn_w2',
                                       shape=(self.h_nodes, 64),
                                       initializer=initializers.RandomUniform(seed=self.seed + 5),

                                       trainable=True)

        self.prnn_b2 = self.add_weight(name='prnn_b2',
                                       shape=(64,),
                                       initializer=initializers.zeros(),
                                       trainable=True)

        self.prnn_w3 = self.add_weight(name='prnn_w3',
                                       shape=(64, 6),
                                       initializer=initializers.RandomUniform(seed=self.seed + 5),

                                       trainable=True)

        self.prnn_b3 = self.add_weight(name='prnn_b3',
                                       shape=(6,),
                                       initializer=initializers.zeros(),
                                       trainable=True)


        self.shape = input_shape

        super(regional_DifferentiableEXPHYDRO_Hu, self).build(input_shape)

    def heaviside(self, x):

        #tanh()返回一个[-1,1]的数  ,  5*x越大返回值越接近1   越小越接近-1   x=0时返回值也为0
        return (K.tanh(5 * x) + 1) / 2



    #用于计算EXO-HYDRO中 flex variables(Ps,Pr) 的函数
    def rainsnowpartition(self, p, t, tmin):

        tmin = tmin * -3  # scale (0, 1) into (-3, 0)

        psnow = self.heaviside(tmin - t) * p
        prain = self.heaviside(t - tmin) * p

        return [psnow, prain]

    #用于计算EXO-HYDRO中雪融量snowmelt——M的函数, M也属于flex variables
    def snowbucket(self, s0, t, ddf, tmax):

        ddf = ddf * 5            # scale (0, 1) into (0, 5)
        tmax = tmax  * 3          # scale (0, 1) into (0, 3)

        melt = self.heaviside(t - tmax) * self.heaviside(s0) * K.minimum(s0, ddf * (t - tmax))

        return melt

    #用于计算EXO-HYDRO中蒸散发ET,集水桶可用蓄水量Qb,和集水桶饱和时产生的超容径流（𝑄s） ： ET属于flex variables, Qb(Qsub) + 𝑄s(Qsurf) 等于 Q
    def soilbucket(self, s1, pet, f, smax, qmax):

        f = f / 10                # scale (0, 1) into (0, 0.1)
        smax = smax * 1400 + 100    # scale (0, 1) into (100, 1500)
        qmax = qmax * 40 + 10      # scale (0, 1) into (10, 50)

        et = self.heaviside(s1) * self.heaviside(s1 - smax) * pet + \
            self.heaviside(s1) * self.heaviside(smax - s1) * pet * (s1 / smax)

        qsub = self.heaviside(s1) * self.heaviside(s1 - smax) * qmax + \
            self.heaviside(s1) * self.heaviside(smax - s1) * qmax * K.exp(-1 * f * (smax - s1))
        qsurf = self.heaviside(s1) * self.heaviside(s1 - smax) * (s1 - smax)

        return [et, qsub, qsurf]

    def step_do(self, step_in, states):
        s0 = states[0][:, 0:1]  # Snow bucket
        s1 = states[0][:, 1:2]  # Soil bucket

        # Load the current input column
        p = step_in[:, 0:1]
        t = step_in[:, 1:2]
        pet = step_in[:, 2:3]

        # Load the current paras
        tmin = step_in[:, 3:4]
        tmax = step_in[:, 4:5]
        ddf  = step_in[:, 5:6]
        f    = step_in[:, 6:7]
        smax = step_in[:, 7:8]
        qmax = step_in[:, 8:9]




        [_ps, _pr] = self.rainsnowpartition(p, t, tmin)

        _m = self.snowbucket(s0, t, ddf, tmax)

        [_et, _qsub, _qsurf] = self.soilbucket(s1, pet, f, smax, qmax)


        # Water balance equations
        _ds0 = _ps - _m
        _ds1 = _pr + _m - _et - _qsub - _qsurf

        #_ds0 = NN(_ps, _m)
        #_ds1 = NN(_pr, _m,_qsub,_qsurf)

        # Record all the state variables which rely on the previous step
        next_s0 = s0 + K.clip(_ds0, -1e5, 1e5)
        next_s1 = s1 + K.clip(_ds1, -1e5, 1e5)

        step_out = K.concatenate([next_s0, next_s1], axis=1)

        return step_out, [step_out]

    #call函数用于实现该层的功能逻辑, 即对于输入张量的计算, 即计算输出径流Q的地方, Keras中x(inputs)只能是一种形式 , 所以不能被事先定义
    def call(self, inputs):
        # Load the input vector
        prcp = inputs[:, :, 0:1]
        tmean = inputs[:, :, 1:2]
        dayl = inputs[:, :, 2:3]

        attrs = inputs[:,:,5:]

        # Calculate PET using Hamon’s formulation
        pet = 29.8 * (dayl * 24) * 0.611 * K.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)

        paras = K.tanh(K.dot(attrs, self.prnn_w1)+ self.prnn_b1) # layer 1
        paras = K.tanh(K.dot(paras, self.prnn_w2)+ self.prnn_b2) # layer 2
        parameters = K.sigmoid(K.dot(paras, self.prnn_w3)+ self.prnn_b3) # layer 3
        #parameters = K.permute_dimensions(parameters, pattern=(1, 0, 2))

        # Concatenate pprcp, tmean, and pet into a new input
        new_inputs = K.concatenate((prcp, tmean, pet, parameters), axis=-1)


        # Define 2 initial state variables at the beginning
        init_states = [K.zeros((K.shape(new_inputs)[0], 2))]

        # Recursively calculate state variables by using RNN
        # return 3 outputs:
        # last_output (the latest output of the rnn, through last time g() & *V & +b)
        # output (all outputs [wrap_number_train, wrap_length, output], through all time g() & *V & +b)
        # new_states(latest states returned by the step_do function, without through last time g() & *V & +b)
        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)
        #outputs: outputs是一个tuple，outputs[0]为最后时刻的输出，outputs[1]为整个输出的时间序列，output[2]是一个list，是中间的隐藏状态。

        s0 = outputs[:, :, 0:1]
        s1 = outputs[:, :, 1:2]

        tmin = parameters[:, :, 0:1]
        tmax = parameters[:, :, 1:2]
        ddf  = parameters[:, :, 2:3]
        f    = parameters[:, :, 3:4]
        smax = parameters[:, :, 4:5]
        qmax = parameters[:, :, 5:6]


        # Calculate final process variables
        [psnow, prain] = self.rainsnowpartition(prcp, tmean, tmin)

        m = self.snowbucket(s0, tmean, ddf, tmax)

        [et, qsub, qsurf] = self.soilbucket(s1, pet, f, smax, qmax)
        print("Qsub:", qsub)
        print("Qsurf",qsurf)

        q = qsub+qsurf

        if self.mode == "normal":
            print("NORMAL!!!")
            return s0
        elif self.mode == "analysis":
            return K.concatenate([s0, s1, et, m], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return (input_shape[0], input_shape[1], 1)
        elif self.mode == "analysis":
            return (input_shape[0], input_shape[1], 6)
class ScaleLayer(Layer):


    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

    def call(self, inputs):

        print("inputs",inputs.shape)


        metattrs = inputs[:,:,:-4]
        physics_v = inputs[:,:,-4:]
        print("physics_v",physics_v.shape)

        #physic_four = inputs[:, :, -4:]
        #print("physic",physic_four.shape)

        #physical_v = inputs[:, :, -8:]


        #physic_four= [wrap_number_train, wrap_length,  et q1 q2 m)]
        #physic_four = inputs[:, :, -1:]
        #print("physic_four_calculatedby_fir_rnncel:",physic_four)

        #[wrap_number_train, 1, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]  五个变量的mean值
        self.met_center = K.mean(physics_v, axis=-2, keepdims=True)

        #[wrap_number_train, 1, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]   五个变量的std值
        self.met_scale = K.std(physics_v, axis=-2, keepdims=True)

        #[wrap_number_train,  wrap_length, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]   五个变量进行标准化
        self.met_scaled = (physics_v - self.met_center) / self.met_scale

        return K.concatenate([metattrs, self.met_scaled], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape
class LSTM_postprocess(Layer):
    def __init__(self, input_xd, hidden_size, seed=200,**kwargs):
        self.input_xd = input_xd
        self.hidden_size = hidden_size
        self.seed = seed
        super(LSTM_postprocess, self).__init__(**kwargs)

    def build(self, input_shape):

        self.w_ih = self.add_weight(name='w_ih', shape=(self.input_xd, 4 * self.hidden_size),
                                 initializer=initializers.Orthogonal(seed=self.seed - 5),
                                 trainable=True)

        self.w_hh = self.add_weight(name='w_hh',
                                       shape=(self.hidden_size, 4 * self.hidden_size),
                                       initializer=initializers.Orthogonal(seed=self.seed + 5),
                                       trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(4 * self.hidden_size, ),
                                    #initializer = 'random_normal',
                                    initializer=initializers.Constant(value=0),
                                    trainable=True)

        self.shape = input_shape
        self.reset_parameters()
        super(LSTM_postprocess, self).build(input_shape)


    def reset_parameters(self):
        #self.w_ih.initializer = initializers.Orthogonal(seed=self.seed - 5)
        #self.w_sh.initializer = initializers.Orthogonal(seed=self.seed + 5)

        w_hh_data = K.eye(self.hidden_size)
        #bias_s_batch = K.repeat_elements(bias_s_batch, rep=sample_size_d, axis=0)
        w_hh_data = K.repeat_elements(w_hh_data, rep=4, axis=1)
        self.w_hh = w_hh_data

        #self.bias.initializer = initializers.Constant(value=0)
        #self.bias_s.initializer = initializers.Constant(value=0)

    def call(self, inputs_x):
        forcing = inputs_x  #[batch, seq_len, dim]
        #print('forcing_shape:',forcing.shape)
        #attrs = inputs_x[1]     #[batch, dim]
        #print('attrs_shape:',attrs.shape)

        forcing_seqfir = K.permute_dimensions(forcing, pattern=(1, 0, 2))  #[seq_len, batch, dim]
        #print('forcing_seqfir_shape:',forcing_seqfir.shape)

        #attrs_seqfir = K.permute_dimensions(attrs, pattern=(1, 0, 2))  #[seq_len, batch, dim]
        #print('attrs_seqfir_shape:',attrs_seqfir.shape)


        seq_len = forcing_seqfir.shape[0]
        #print('seq_len:',seq_len)
        batch_size = forcing_seqfir.shape[1]
        #print('batch_size:',batch_size)

        #init_states = [K.zeros((K.shape(forcing)[0], 2))]
        #h0, c0 = [K.zeros(shape= (sample_size_d,self.hidden_size)),K.zeros(shape= (sample_size_d,self.hidden_size))]
        h0 = K.zeros(shape= (batch_size, self.hidden_size))
        c0 = K.zeros(shape= (batch_size, self.hidden_size))
        h_x = (h0, c0)

        h_n, c_n = [], []

        bias_batch = K.expand_dims(self.bias, axis=0)
        bias_batch = K.repeat_elements(bias_batch, rep=batch_size, axis=0)
        #print("bias_batch:",bias_batch.shape)

        #bias_s_batch = K.expand_dims(self.bias_s, axis=0)
        #bias_s_batch = K.repeat_elements(bias_s_batch, rep=batch_size, axis=0)
        #这里对静态变量通过输入门的相加操作可能有问题,两张量维度不一样, attrs输入这里应该是二维 [batch_size, xs_dim]   , [sample_size, xs_dim]
        #i = K.sigmoid(K.dot(attrs, self.w_sh) + bias_s_batch)

        for t in range(seq_len):
            h_0, c_0 = h_x

            #这里也有问题, 必须把forcing数据的seq_len放在第一维 [seq_len, batch_size, xd_dim]
            gates =((K.dot(h_0, self.w_hh) + bias_batch) + K.dot(forcing_seqfir[t], self.w_ih))
            f, i, o, g = tf.split(value=gates, num_or_size_splits=4, axis=1)

            next_c = K.sigmoid(f) * c_0 + K.sigmoid(i) * K.tanh(g)
            next_h = K.sigmoid(o) * K.tanh(next_c)

            h_n.append(next_h)
            c_n.append(next_c)

            h_x = (next_h,next_c)

        h_n = K.stack(h_n, axis=0)
        c_n = K.stack(c_n, axis=0)

        return h_n














