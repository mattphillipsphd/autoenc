# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:29:15 2017

@author: Matt
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

pj = os.path.join


class InputBase:
    def __init__(self, data_location):
        self._batch_size = None
        self._data_location = data_location
        self._input_shape = None
        self._is_initialized = False
        
    def initialize(self, input_shape, batch_size):
        raise RuntimeError("This method must be implemented in a subclass")
    
    def next_test_batch(self):
        raise RuntimeError("This method must be implemented in a subclass")
    
    def next_train_batch(self):
        raise RuntimeError("This method must be implemented in a subclass")
    
    def num_test_inputs(self):
        raise RuntimeError("This method must be implemented in a subclass")
    
    def num_train_inputs(self):
        raise RuntimeError("This method must be implemented in a subclass")


class InputMnist(InputBase):
    def __init__(self, data_location):
        InputBase.__init__(self, data_location)
        
    def initialize(self, input_shape, batch_size):
        self._batch_size = batch_size
        self._input_shape = input_shape
        
        from tensorflow.examples.tutorials.mnist import input_data
        self._generator = input_data.read_data_sets( self._data_location )
        
        self._is_initialized = True        
    
    def next_test_batch(self):
        if not self._is_initialized:
            raise RuntimeError("Generator not initialized")
        batch = self._generator.test.next_batch( self._batch_size )
        return batch[0].reshape([-1] + self._input_shape)
        
    def next_train_batch(self):
        if not self._is_initialized:
            raise RuntimeError("Generator not initialized")
        batch = self._generator.train.next_batch( self._batch_size )
        return batch[0].reshape([-1] + self._input_shape)
    
    def num_test_inputs(self):
        return self._generator.test.num_examples
        
    def num_train_inputs(self):
        return self._generator.train.num_examples


class InputPlanetFile(InputBase):
    def __init__(self, data_location):
        InputBase.__init__(self, data_location)
        self._train_chips = None
        self._test_chips = None
        self._train_batch_ct = 0
        self._test_batch_ct = 0
       
    def _create_chips(self, large_image):
        M = large_image.shape[0]
        N = large_image.shape[1]
        i = 0
        chips = []
        chip_size = self._input_shape[0]
        num_channels = self._input_shape[2]
        while i < M - chip_size:
            j = 0
            while j < N - chip_size:
                chip = large_image[i:i+chip_size, j:j+chip_size, 0:num_channels] / 256.0
                chips.append(chip)
                j += chip_size
            i += chip_size
        return chips
    
    def initialize(self, input_shape, batch_size):
        self._batch_size = batch_size
        self._input_shape = input_shape
        
        im = plt.imread(self._data_location)
        chips = self._create_chips(im)
        np.random.shuffle(chips)
        
        trainN = int(round(0.8 * len(chips)))
        self._train_chips = chips[:trainN]
        self._test_chips = chips[trainN:]
        
        self._is_initialized = True   
    
    def next_test_batch(self):
        if not self._is_initialized:
            raise RuntimeError("Generator not initialized")
        N = len(self._test_chips)
        end_idx = (self._test_batch_ct + 1) * self._batch_size
        if end_idx > N:
            self._test_batch_ct = 0
            end_idx = self._batch_size
        beg_idx = self._test_batch_ct * self._batch_size
        batch = self._test_chips[beg_idx : end_idx]
        self._test_batch_ct += 1
        return batch
        
    def next_train_batch(self):
        if not self._is_initialized:
            raise RuntimeError("Generator not initialized")
        N = len(self._train_chips)
        end_idx = (self._train_batch_ct + 1) * self._batch_size
        if end_idx > N:
            self._train_batch_ct = 0
            end_idx = self._batch_size
        beg_idx = self._train_batch_ct * self._batch_size
        batch = self._train_chips[beg_idx : end_idx]
        self._train_batch_ct += 1
        return batch
    
    def num_test_inputs(self):
        return len(self._test_chips)
        
    def num_train_inputs(self):
        return len(self._train_chips)


# This class assumes that the images have already been chipped and are in the
# data_location directory
class InputAvedai(InputBase):
    def __init__(self, data_location):
        InputBase.__init__(self, data_location)
        self._train_chips = None
        self._test_chips = None
        self._train_batch_ct = 0
        self._test_batch_ct = 0
       
    def initialize(self, input_shape, batch_size):
        self._batch_size = batch_size
        self._input_shape = input_shape
        
        def _shift(img, px_x, px_y):
            shift_img = np.copy(img)
            sz = img.shape[0]
            xbeg = np.max([0, px_x])
            xend = np.min([sz, sz+px_x])
            ybeg = np.max([0, px_y])
            yend = np.min([sz, sz+px_y])
            xrng = sz - np.abs(px_x)
            yrng = sz - np.abs(px_y)
            if px_x>=0 and px_y>=0:
                shift_img[px_x:sz, px_y:sz, :] = img[xbeg:xend, ybeg:yend, :]
            elif px_x>=0 and px_y<0:
                shift_img[px_x:sz, 0:yrng, :] = img[xbeg:xend, ybeg:yend, :]
            elif px_x<0 and px_y>=0:
                shift_img[0:xrng, px_y:sz, :] = img[xbeg:xend, ybeg:yend, :]
            else:
                shift_img[0:xrng, 0:yrng, :] = img[xbeg:xend, ybeg:yend, :]
            return shift_img

        chip_list = [f for f in os.listdir(self._data_location) if f.endswith(".png")]
        chips = []
        delta = 15
        for chip in chip_list:
            img = plt.imread( pj(self._data_location,chip) )
            chips.append( _shift(img, delta, delta) )
            chips.append( _shift(img, delta, 0) )
            chips.append( _shift(img, delta, -delta) )
            chips.append( _shift(img, 0, delta) )
            chips.append( _shift(img, 0, -delta) )
            chips.append( _shift(img, -delta, delta) )
            chips.append( _shift(img, -delta, 0) )
            chips.append( _shift(img, -delta, -delta) )
        np.random.shuffle(chips)
        
        trainN = int(round(0.8 * len(chips)))
        self._train_chips = chips[:trainN]
        self._test_chips = chips[trainN:]
        
        self._is_initialized = True   
    
    def next_test_batch(self):
        if not self._is_initialized:
            raise RuntimeError("Generator not initialized")
        N = len(self._test_chips)
        end_idx = (self._test_batch_ct + 1) * self._batch_size
        if end_idx > N:
            self._test_batch_ct = 0
            end_idx = self._batch_size
        beg_idx = self._test_batch_ct * self._batch_size
        batch = self._test_chips[beg_idx : end_idx]
        self._test_batch_ct += 1
        return batch
        
    def next_train_batch(self):
        if not self._is_initialized:
            raise RuntimeError("Generator not initialized")
        N = len(self._train_chips)
        end_idx = (self._train_batch_ct + 1) * self._batch_size
        if end_idx > N:
            self._train_batch_ct = 0
            end_idx = self._batch_size
        beg_idx = self._train_batch_ct * self._batch_size
        batch = self._train_chips[beg_idx : end_idx]
        self._train_batch_ct += 1
        return batch
    
    def num_test_inputs(self):
        return len(self._test_chips)
        
    def num_train_inputs(self):
        return len(self._train_chips)





def input_generator_factory(gen_str, data_location):
    if gen_str.lower() == "mnist":
        return InputMnist(data_location)
    elif gen_str.lower() == "planetfile":
        return InputPlanetFile(data_location)
    elif gen_str.lower() == "avedai":
        return InputAvedai(data_location)
    else:
        raise RuntimeError("Generator not recognized")


def check_config(cfg):
    def make_into_list(key, cfg):
        conv_filter_bank_size = cfg["conv_filter_bank_size"]
        num_conv = len(conv_filter_bank_size)
        if not type(cfg[key]) is list:
            cfg[key] = [cfg[key]] * num_conv
        elif len(cfg[key]) == 1:
            cfg[key] = cfg[key] * num_conv
        elif len(cfg[key]) != num_conv:
            raise RuntimeError("Invalid config file entry for " + key)
        
    for key in ["conv_filter_bank_size", "bottleneck", "max_pool_size", 
                "conv_stride", "conv_kernel_size", "input_shape",
                "max_pool_stride"]:
        if key not in cfg:
            raise RuntimeError("Field " + key + " not present in config file.")
    make_into_list("max_pool_size", cfg)
    make_into_list("conv_stride", cfg)
    make_into_list("conv_kernel_size", cfg)
    if len(cfg["input_shape"]) != 3:
        raise RuntimeError("image_shape must be [height, width, # channels]")


def fill_cfg_defaults(cfg):
    for key in ["conv_filter_bank_size", "bottleneck"]:
        if key not in cfg:
            raise RuntimeError("Field " + key + " not present in config file.")
    conv_filters = cfg["conv_filter_bank_size"]
    num_conv = len(conv_filters)
    if "max_pool_size" not in cfg:
        cfg["max_pool_size"] = [2] * num_conv
    if "conv_stride" not in cfg:
        cfg["conv_stride"] = [1] * num_conv
    if "conv_kernel_size" not in cfg:
        cfg["conv_kernel_size"] = [3] * num_conv
    if "max_pool_stride" not in cfg:
        cfg["max_pool_stride"] = [2] * num_conv


def get_log_file_dir(cfg):
    logdir = cfg["logging_dir"]
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    stub = "log_%03d"
    file_ct = 0
    logfd = stub % (file_ct)
    while os.path.exists( pj(logdir, logfd) ):
        file_ct += 1
        logfd = stub % (file_ct)
    return pj(logdir, logfd)


def make_bottleneck(conv_last, cfg):
    bottleneck = cfg["bottleneck"]
    if len(bottleneck) > 1:
        raise RuntimeError("Not ready for bottleneck layer > 1 yet")
    conv_shape = conv_last.shape
    num_h = bottleneck[0]
    num_conv_units = int(conv_last.shape[1]) \
                * int(conv_last.shape[2]) * int(conv_last.shape[3])
    conv_last = tf.reshape(conv_last, [-1, num_conv_units])
    h = make_fc_layer(conv_last, num_h, "bottleneck_0")
    print("h: ", h.shape)
    
    bn_out = make_fc_layer(h, num_conv_units, "deconv0")
    bn_out_shape = [-1] + [int(conv_shape[1]), int(conv_shape[2]),
                   int(conv_shape[3])]
    bn_out = tf.reshape(bn_out, bn_out_shape)
    print("bn_out: ", bn_out.shape)
    
    return bn_out


def make_conv_layers(x, cfg):
    def make_conv_layer(t_in, cfg, index):
        name = "conv_layer_" + str(i)
        with tf.name_scope(name):
            num_channels = int(t_in.shape[3])
            kernel_size = cfg["conv_kernel_size"][index]
            conv_stride = cfg["conv_stride"][index]
            pool_stride = cfg["max_pool_stride"][index]
            pool_size = cfg["max_pool_size"][index]
            num_filters = cfg["conv_filter_bank_size"][index]
            print("t_in:", t_in.shape)
            
            kernel_shape = [kernel_size, kernel_size, num_channels, num_filters]
            print("kernel_shape:", kernel_shape)
            W = tf.get_variable(name+"_W", 
                                kernel_shape, 
                                initializer=tf.contrib.layers.xavier_initializer())
            #tf.layers.conv2d uses the glorot_uniform_initializer
            
            b = tf.get_variable(name+"_b", 
                                [num_filters],
                                initializer=tf.zeros_initializer())
            y = tf.nn.conv2d(t_in,
                             W,
                             [1, conv_stride, conv_stride, 1],
                             padding="SAME") + b
            print("y:", y.shape)
            mp = tf.layers.max_pooling2d(y,
                                        pool_size,
                                        strides=pool_stride,
                                        padding="SAME",
                                        name=name+"_mp")
    
            tf.summary.histogram(mp.name[:-2], mp)
            print("mp:", mp.shape)
            
            act = tf.nn.relu(mp, name=name+"_relu")
            print("act:", act.shape)
            return mp
    
    conv_filters = cfg["conv_filter_bank_size"]
    num_conv = len(conv_filters)
    t_in = x
    layer_shapes = [t_in.shape]
    for i in range(num_conv):
        t_in = make_conv_layer(t_in, cfg, i)
        layer_shapes.append(t_in.shape)
        
    return t_in,layer_shapes


def make_deconv_layers(bn, layer_shapes, cfg):
    # NB, for deconvolution indexing starts from the back
    def make_deconv_layer(t_in, layer_shape, cfg, index):
        name = "deconv_layer_" + str(i)
        with tf.name_scope(name):
            in_channels = int(t_in.shape[3])
            kernel_size = cfg["conv_kernel_size"][index]
            conv_stride = cfg["conv_stride"][index]
            out_channels = cfg["conv_filter_bank_size"][index]
            print("t_in:", t_in.shape)

            ht = int(layer_shape[1])
            wd = int(layer_shape[2])
            upsample = tf.image.resize_nearest_neighbor(t_in, (ht, wd))
            
            kernel_shape = [kernel_size, kernel_size, in_channels, out_channels]
            W = tf.get_variable(name+"_W",
                                kernel_shape,
                                initializer=tf.contrib.layers.xavier_initializer())        
            b = tf.get_variable(name+"_b",
                                [out_channels],
                                initializer=tf.zeros_initializer())
        
            deconv = tf.nn.conv2d(upsample, 
                                  W, 
                                  [1, conv_stride, conv_stride, 1],
                                  "SAME") + b
            print("deconv: ", deconv.shape)
    
            act = tf.nn.relu(deconv, name=name+"_relu")
                
            tf.summary.histogram(act.name[:-2], act)
            print("act:", act.shape)
            
            return act
        
    conv_filters = cfg["conv_filter_bank_size"]
    num_conv = len(conv_filters)
    t_in = bn
    #Add in number of image input channels to list of filter bank sizes
    cfg["conv_filter_bank_size"] = [ cfg["input_shape"][2] ] + cfg["conv_filter_bank_size"]
    for i in range(num_conv):
        idx = num_conv - i - 1
        t_in = make_deconv_layer(t_in, layer_shapes[idx], cfg, idx)
        
    return t_in
    

def make_fc_layer(t_in, fan_out, name):
    with tf.name_scope(name):
        W = tf.Variable(tf.random_normal([int(t_in.shape[1]), fan_out], stddev=1), name=name+"_W")
        b = tf.Variable(tf.random_normal([fan_out], stddev=1), name=name+"_b")
        y = tf.matmul(t_in, W) + b
        act = tf.nn.sigmoid(y, name=name+"_sigmoid")
        tf.summary.histogram(act.name[:-2], act)
        return act
    
    
def make_graph(x, cfg):
    conv_last,layer_shapes = make_conv_layers(x, cfg)
    bn_last = make_bottleneck(conv_last, cfg)
    deconv_last = make_deconv_layers(bn_last, layer_shapes, cfg)
    return deconv_last
    
    
def make_input(cfg):
    shape = cfg["input_shape"]
    return tf.placeholder(tf.float32,
                          shape=(None, shape[0], shape[1], shape[2]),
                          name="x")
    
    
def make_training_op(out, target, cfg):    
    with tf.name_scope("conv_mse"):
        sq_error = tf.reduce_mean(np.square(out - target), axis=1)
        cost = tf.reduce_mean( sq_error, name="cost" )
    
        tf.summary.scalar("cost", cost)
        
    with tf.name_scope("conv_opt"):
        lr = cfg["learning_rate"]
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    #    optimizer = tf.train.RMSPropOptimizer(learning_rate=conf["lr"])
    #    optimizer = tf.train.MomentumOptimizer(learning_rate=conf["lr"], momentum=conf["momentum"])
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate=conf["lr"])
        training_op = optimizer.minimize(cost)
        
    return training_op,cost


def train_model(input_gen, model, training_op, x, cost, cfg):
    input_gen.initialize(cfg["graph"]["input_shape"],
                         cfg["train"]["batch_size"])
    num_train_images = input_gen.num_train_inputs()
    
    cfg = cfg["train"]
    with tf.device("/gpu:0"):
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
    
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(get_log_file_dir(cfg), sess.graph)
    
        for i in range(cfg["num_epochs"]):
            train_costs = []
            test_costs= []
            for ii in range(num_train_images // cfg["batch_size"]):
                train_batch = input_gen.next_train_batch()
                summary,_ = sess.run([merged, training_op], feed_dict={x : train_batch})
                train_writer.add_summary(summary,
                                         i*(num_train_images//cfg["batch_size"])+ii)
                
                test_batch = input_gen.next_test_batch()
                train_costs.append( cost.eval(session=sess, feed_dict={x : train_batch}) )
                test_costs.append( cost.eval(session=sess, feed_dict={x : test_batch}) )
                
            print("Training cost: %f, test cost: %f" \
                  % (np.mean(train_costs), np.mean(test_costs)))
            
        test_batch = input_gen.next_test_batch()
        for i in range(0,10):
            decoded_chip = model.eval(session=sess, feed_dict={x : [test_batch[i]]})
            fig,axes = plt.subplots(nrows=1, ncols=2)
            axes[0].imshow( np.squeeze(test_batch[i]) )
            axes[1].imshow( np.squeeze(decoded_chip) )
            fig.savefig("Comparison_" + str(i) + ".png")
    

def create_and_train_model(config_fn : str, input_generator : InputBase):
    with open(config_fn) as fp:
        cfg = json.load(fp)
        fill_cfg_defaults(cfg["graph"])
        check_config(cfg["graph"]) # TODO Extend to all variables
        print("Configuration:")
        for key, val in cfg.items():
            print("\t" + key + ": " + str(val))
    
    tf.reset_default_graph()
    x = make_input(cfg["graph"])
    graph = make_graph(x, cfg["graph"])
    training_op,cost = make_training_op(graph, x, cfg["train"])
    train_model(input_generator, graph, training_op, x, cost, cfg) # TODO How to extract variable?!?


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", 
	default="C:/Users/Matt/Documents/nbs/autoenc/configs/planetfile.json")
    parser.add_argument("-g", "--generator", default="mnist")
    parser.add_argument("-d", "--data-dir", default="J:/Datasets/MNIST")
    
    args = parser.parse_args()
    
    
#    RAW_DATA_DIR = "J:/Datasets/Planet/Salinas_001/20171028_190652_1056020_RapidEye-5"
#    RAW_DATA_FILE = "1056020_2017-10-28_RE5_3A_Visual.tif"
#    salinas1 = pj(RAW_DATA_DIR, RAW_DATA_FILE)
#    input_gen = input_generator_factory("planetfile", salinas1)

    input_gen = input_generator_factory(args.generator, args.data_dir)
    create_and_train_model(args.config, input_gen)
