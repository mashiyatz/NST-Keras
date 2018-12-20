from logging import exception
import tensorflow as tf
from keras import backend as K
from enum import Enum

class Distance(Enum):
    L2 = 0
    DotProduct = 1

class TensorAxis:
    N = 0
    H = 1
    W = 2
    C = 3

class CSFlow:
    def __init__(self, sigma = float(0.1), b = float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances, axis_for_normalization = TensorAxis.C):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = K.exp((self.b - scaled_distances) / self.sigma)
        self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)

    def reversed_direction_CS(self):
        cs_flow_opposite = CSFlow(self.sigma, self.b)
        cs_flow_opposite.raw_distances = self.raw_distances
        work_axis = [TensorAxis.H, TensorAxis.W]
        relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
        cs_flow_opposite.__calculate_CS(relative_dist, work_axis)
        return cs_flow_opposite

    # --
    @staticmethod
    def create_using_L2(I_features, T_features, sigma = float(0.1), b = float(1.0), batch_size=8):
        cs_flow = CSFlow(sigma, b)
        #with tf.name_scope('CS'): #what is this for?
        c = T_features.shape[TensorAxis.C].value
        sT = K.shape(T_features)
        sI = K.shape(I_features)

        Tvecs = K.reshape(T_features, (sT[TensorAxis.N], -1, sT[TensorAxis.C]))
        Ivecs = K.reshape(I_features, (sI[TensorAxis.N], -1, sI[TensorAxis.C]))

        r_Ts = K.sum(Tvecs * Tvecs, axis=2)
        r_Is = K.sum(Ivecs * Ivecs, axis=2)
        raw_distances_list = []

        # for i in range(sT[TensorAxis.N]):
        for i in range(batch_size):
            Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
            A = Tvec @ K.transpose(Ivec) #what is @?
            cs_flow.A = A
            r_T = K.reshape(r_T, [-1, 1])  # turn to column vector
            dist = r_T - 2 * A + r_I

            cs_shape = (1, sI[1], sI[2], K.shape(dist)[0])
            dist = K.reshape(K.transpose(dist), cs_shape)

            # protecting against numerical problems, dist should be positive
            dist = K.maximum(float(0.0),dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = K.variable([K.squeeze(raw_dist, axis=0) for raw_dist in raw_distances_list])

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    #--
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma = float(1.0), b = float(1.0), batch_size=8):
        cs_flow = CSFlow(sigma, b)
        #with tf.name_scope('CS'):
            # prepare feature before calculating cosine distance
        T_features, I_features = cs_flow.center_by_T(T_features, I_features)
        #with tf.name_scope('TFeatures'):
        T_features = CSFlow.l2_normalize_channelwise(T_features)
        #with tf.name_scope('IFeatures'):
        I_features = CSFlow.l2_normalize_channelwise(I_features)

            # work separately for each example in dim 1
        cosine_dist_l = []
        N, _, __, ___ = T_features.shape.as_list()
        for i in range(N):
            T_features_i = K.expand_dims(T_features[i, :, :, :], axis=0)
            I_features_i = K.expand_dims(I_features[i, :, :, :], axis=0)
            patches_HWCN_i = cs_flow.patch_decomposition(T_features_i)
            cosine_dist_i = K.conv2d(x=I_features_i, kernel=patches_HWCN_i,
                                     strides=(1, 1, 1, 1), padding='valid',
                                     data_format='channels_last')
            cosine_dist_l.append(cosine_dist_i)

        cs_flow.cosine_dist = K.concatenate(cosine_dist_l, axis = 0)

        cosine_dist_zero_to_one = -(cs_flow.cosine_dist - 1) / 2
        cs_flow.raw_distances = cosine_dist_zero_to_one

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    def calc_relative_distances(self, axis=TensorAxis.C):
        epsilon = 1e-5
        div = tf.reduce_min(self.raw_distances, axis=axis, keepdims=True)
        # div = tf.reduce_mean(self.raw_distances, axis=axis, keepdims=True)
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    def weighted_average_dist(self, axis = TensorAxis.C):
        if not hasattr(self, 'raw_distances'):
            raise exception('raw_distances property does not exists. cant calculate weighted average l2')

        multiply = self.raw_distances * self.cs_NHWC
        return K.sum(multiply, axis=axis) #reduce sum vs sum?

    # --
    @staticmethod
    def create(I_features, T_features, distance : Distance, nnsigma=float(1.0), b=float(1.0), batch_size=8):
        if distance.value == Distance.DotProduct.value:
            cs_flow = CSFlow.create_using_dotP(I_features, T_features, nnsigma, b, batch_size)
        elif distance.value == Distance.L2.value:
            cs_flow = CSFlow.create_using_L2(I_features, T_features, nnsigma, b, batch_size)
        else:
            raise "not supported distance " + distance.__str__()
        return cs_flow

    @staticmethod
    def sum_normalize(cs, axis=TensorAxis.C):
        reduce_sum = K.sum(cs, axis=axis, keepdims=True)
        #return tf.divide(cs, reduce_sum)
        return K.variable(value=cs/reduce_sum)

    def center_by_T(self, T_features, I_features):

        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT = K.mean(T_features, axes)
        self.varT = K.var(T_features, axes)

        self.T_features_centered = T_features - self.meanT
        self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = K.l2_normalize(features, axis=TensorAxis.C)
        # expanding the norms tensor to support broadcast division
        norms_expanded = K.expand_dims(norms, TensorAxis.C)
        features = K.variable(value=features/norms_expanded)
        return features

    def patch_decomposition(self, T_features):
        # patch decomposition
        # see https://stackoverflow.com/questions/40731433/understanding-tf-extract-image-patches-for-extracting-patches-from-an-image
        patch_size = 1
        patches_as_depth_vectors = tf.extract_image_patches(
            images=T_features, ksizes=[1, patch_size, patch_size, 1],
            strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID',
            name='patches_as_depth_vectors')

        self.patches_NHWC = K.reshape(
            patches_as_depth_vectors,
            shape=[-1, patch_size, patch_size, patches_as_depth_vectors.shape[3].value])

        self.patches_HWCN = K.permute_dimensions(
            x=self.patches_NHWC,
            pattern=[1, 2, 3, 0])  # tf.conv2 ready format

        return self.patches_HWCN


#--------------------------------------------------
#           CX loss
#--------------------------------------------------

# Keras loss functions must only take (y_true, y_pred) as parameters.

def CX_loss(T_features, I_features):
    T_features = K.variable(T_features, dtype=tf.float32)
    I_features = K.variable(I_features, dtype=tf.float32)

    with tf.name_scope('CX'):
        cs_flow = CSFlow.create(I_features, T_features, distance=Distance.L2, nnsigma=float(1.0))
        # sum_normalize:
        height_width_axis = [TensorAxis.H, TensorAxis.W]
        # To:
        cs = cs_flow.cs_NHWC
        k_max_NC = K.max(cs, axis=height_width_axis)
        CS = K.mean(k_max_NC, axis=[1])
        CX_as_loss = 1 - CS
        CX_loss = -K.log(1 - CX_as_loss)
        CX_loss = K.mean(CX_loss)
        return CX_loss

#def CX_loss(T_features, I_features, distance=Distance.L2, nnsigma=float(1.0), batch_size=1):
#    T_features = tf.convert_to_tensor(T_features, dtype=tf.float32)
#    I_features = tf.convert_to_tensor(I_features, dtype=tf.float32)

#    with tf.name_scope('CX'):
#        cs_flow = CSFlow.create(I_features, T_features, distance, nnsigma, batch_size=batch_size)
#        # sum_normalize:
#        height_width_axis = [TensorAxis.H, TensorAxis.W]
#        # To:
#        cs = cs_flow.cs_NHWC
#        k_max_NC = tf.reduce_max(cs, axis=height_width_axis)
#        CS = tf.reduce_mean(k_max_NC, axis=[1])
#        CX_as_loss = 1 - CS
#        CX_loss = -tf.log(1 - CX_as_loss)
#        CX_loss = tf.reduce_mean(CX_loss)
#        return CX_loss
