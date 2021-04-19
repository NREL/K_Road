from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


# CAMERA_IMAGE_DIM = (18, 10, 1)
# SCALAR_FEATURE_DIM = 10

## TODO: This information is in the obs_space dict we make for gym, but it gets flattened before we build the model,
## we need a way to get this information here so we don't hard code this.
## But note, in the meantime, this much match what is in FactoredRoadObserver class
# have_occupancy_grid = True
# NCHANNELS = 1
# OCCUPANCY_GRID_FEATURE_DIM = (18,10,NCHANNELS)

# have_odometry = True  ### will crash if not True
# have_lidar = False


##                'conv_filters':      [[16, [3, 3], 2], [32, [3,2],2], [64, [3,2], 1]],  ## valid for grid=9x5
##                'conv_filters':      [[16, [5, 5], 2], [32, [3,3],2], [64, [5,3], 1]],  ## valid for grid=18x10

### TODO: embed this in config


class FactoredNetwork(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(FactoredNetwork, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        print("BUILDING FACTORED NET")
        print("obs_space", obs_space)  ### will be already flattened
        print("model_config", model_config)  ### ah, but we hide the obs_config here
        obs_config = model_config['custom_options']['obs_config']
        print("OBS_CONFIG", obs_config)

        self.have_occupancy_grid = "occupancy_grid" in obs_config
        self.have_odometry = "odometry" in obs_config
        assert (self.have_odometry)
        self.have_lidar = "lidar" in obs_config
        ODOMETRY_FEATURE_DIM = 9
        VECTOR_FEATURE_DIM = ODOMETRY_FEATURE_DIM
        if self.have_lidar:
            config = obs_config['lidar']
            forward_scan_resolution = config['forward_scan_resolution']
            rear_scan_resolution = config['rear_scan_resolution']
            lidar_channels = config['lidar_channels']
            LIDAR_FEATURE_DIM = (forward_scan_resolution + rear_scan_resolution) * lidar_channels
            VECTOR_FEATURE_DIM += LIDAR_FEATURE_DIM

        init_kernel = 'glorot_uniform'
        init_bias = 'zeros'
        init_gamma = 'ones'
        init_moving_mean = 'zeros'
        init_beta = 'zeros'
        init_moving_variance = 'ones'

        scalar_feature = tf.keras.layers.Input(shape=VECTOR_FEATURE_DIM, name='vector_features')
        dense_2 = tf.keras.layers.Dense(256, kernel_initializer=init_kernel,
                                        bias_initializer=init_bias, activation=tf.nn.elu)(scalar_feature)
        if self.have_occupancy_grid:
            config = obs_config['occupancy_grid']
            OCCUPANCY_GRID_FEATURE_DIM = config['grid_dims']

            CNN_STRUCTURE = config['cnn_structure']
            # [[16, [5, 5], (2,2)],
            # [32, [3,3], (2,2)],
            # [64, [5,3], (1,1)]]
            LAYER_NAME = ['conv2d_%d' % i for i in range(len(CNN_STRUCTURE))]
            BN_NAME = ['batch_norm_2d_%d' % i for i in range(len(CNN_STRUCTURE))]

            inputs_image = tf.keras.layers.Input(shape=OCCUPANCY_GRID_FEATURE_DIM, name="array_features")
            cnn_output = inputs_image
            for layer_id, layer_struct in enumerate(CNN_STRUCTURE):
                cnn_output = tf.keras.layers.Conv2D(layer_struct[0],
                                                    layer_struct[1],
                                                    layer_struct[2],
                                                    padding="same",
                                                    kernel_initializer=init_kernel,
                                                    bias_initializer=init_bias,
                                                    activation=tf.nn.elu)(cnn_output)

                cnn_output = tf.keras.layers.BatchNormalization(gamma_initializer=init_gamma,
                                                                moving_mean_initializer=init_moving_mean,
                                                                beta_initializer=init_beta,
                                                                moving_variance_initializer=init_moving_variance)(
                    cnn_output)

            cnn_output_shape = cnn_output.get_shape().as_list()
            flatten = tf.reshape(cnn_output,
                                 [-1, cnn_output_shape[1] * cnn_output_shape[2] * cnn_output_shape[3]])

            dense_1 = tf.keras.layers.Dense(256, kernel_initializer=init_kernel,
                                            bias_initializer=init_bias, activation=tf.nn.elu)(flatten)
            concat = tf.concat((dense_1, dense_2), 1)
        else:
            concat = tf.concat((dense_2), 1)

        dense_3 = tf.keras.layers.Dense(256, kernel_initializer=init_kernel,
                                        bias_initializer=init_bias, activation=tf.nn.elu)(concat)
        dense_4 = tf.keras.layers.Dense(256, kernel_initializer=init_kernel,
                                        bias_initializer=init_bias, activation=tf.nn.elu)(dense_3)
        dense_5 = tf.keras.layers.Dense(128, kernel_initializer=init_kernel,
                                        bias_initializer=init_bias, activation=tf.nn.elu)(dense_4)

        # action 1
        # Well, the code below can be implemented using output_1 = tf.keras.layers.Dense(4)(dense_3), however, we have
        # to split things up because ORNL weights provided are splitted version. So don't blame me!

        # The code below can be refactored, but I think readability will be compromised, so I decided to keep it like
        # this for now.

        throttle = tf.keras.layers.Dense(1, kernel_initializer=init_kernel,
                                         bias_initializer=init_bias, activation=tf.nn.tanh)(dense_5)
        output_1 = tf.concat([throttle], axis=1)

        # action 2
        steering = tf.keras.layers.Dense(1, kernel_initializer=init_kernel,
                                         bias_initializer=init_bias, activation=tf.nn.tanh)(dense_5)
        output_2 = tf.concat([steering], axis=1)

        # The variance kernel and bias is initialized as zero.
        init_kernel = 'zeros'
        init_bias = 'zeros'
        # action 1 variance
        output_3 = tf.keras.layers.Dense(1, kernel_initializer=init_kernel, bias_initializer=init_bias)(dense_5)
        # action 2 variance
        output_4 = tf.keras.layers.Dense(1, kernel_initializer=init_kernel, bias_initializer=init_bias)(dense_5)

        layer_out = tf.concat((tf.expand_dims(output_1, 1),
                               tf.expand_dims(output_2, 1),
                               tf.expand_dims(output_3, 1),
                               tf.expand_dims(output_4, 1)), 1)

        layer_out = tf.reshape(layer_out, shape=(-1, 4))

        # layer_out = output_1

        # Value network
        dense_2_val = tf.keras.layers.Dense(256, kernel_initializer=init_kernel,
                                            bias_initializer=init_bias, activation=tf.nn.elu)(scalar_feature)
        if (self.have_occupancy_grid):
            dense_1_val = tf.keras.layers.Dense(256, kernel_initializer=init_kernel,
                                                bias_initializer=init_bias, activation=tf.nn.elu)(flatten)
            concat_val = tf.concat((dense_1_val, dense_2_val), 1)
        else:
            concat_val = tf.concat((dense_2_val), 1)

        dense_3_val = tf.keras.layers.Dense(256, activation=tf.nn.elu)(concat_val)
        dense_4_val = tf.keras.layers.Dense(128, activation=tf.nn.elu)(dense_3_val)
        dense_5_val = tf.keras.layers.Dense(1, activation=None)(dense_4_val)

        value_out = tf.expand_dims(dense_5_val, 1)
        value_out = tf.reshape(value_out, shape=(-1,))

        if self.have_occupancy_grid:
            self.base_model = tf.keras.Model([inputs_image, scalar_feature], [layer_out, value_out])
        else:
            self.base_model = tf.keras.Model([scalar_feature], [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        #        print ("FORWARD", input_dict)
        vector_dat = input_dict["obs"]['odometry']
        if self.have_lidar:
            vector_dat = tf.concat((vector_dat, input_dict["obs"]['lidar']), 1)
        if self.have_occupancy_grid:
            input_list = [input_dict["obs"]['occupancy_grid'], vector_dat]
        else:
            input_list = [vector_dat]
        model_out, self._value_out = self.base_model(input_list)
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class FactoredModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(FactoredModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model = FactoredNetwork(obs_space, action_space, num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()
