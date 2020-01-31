import json
from keras.layers import Input, Reshape, Dense, Flatten, Activation, BatchNormalization, Dropout, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Lambda, Add, ZeroPadding2D, UpSampling2D


class NN(object):
    def __init__(self, conf = ""):
        self.layers = {}
        self.blocks = {}

        if len(conf) > 0:
            with open(conf, "r") as conf_file:
                conf_data = json.loads(conf_file.read())

                if "layers" in conf_data:
                    self.layers = conf_data["layers"]

                if "blocks" in conf_data:
                    self.blocks = conf_data["blocks"]


    def mkLambdaLayer(self, fun_name, layer_name):
        if fun_name == 'std_image':
            return Lambda(std_image)

        assert False, "Not supported Lambda function: %s" % (fun_name)

    def mkLayer(self, layer):
        strides = (1, 1)
        if 'strides' in layer:
            strides = layer['strides']
            if type(strides) is list:
                strides = tuple(strides)

        activation = None
        if 'activation' in layer:
            activation = layer['activation']

        name = None
        if 'name' in layer:
            name = layer['name']

        use_bias = True
        if 'use_bias' in layer:
            use_bias = layer['use_bias']

        layer_type = layer["layer"]
        if  layer_type == 'Dense':
            return Dense(int(layer['units']), activation=activation, use_bias=use_bias, name=name)
        elif layer_type == 'Reshape':
            return Reshape(tuple(layer['shape']), name=name)
        elif layer_type == 'Conv2DTranspose':
            return Conv2DTranspose(int(layer['filters']), layer['kernel'], strides=strides, padding=layer['padding'], activation=activation, use_bias=use_bias, name=name)
        elif layer_type == 'Conv2D':
            return Conv2D(int(layer['filters']), layer['kernel'], strides=strides, padding=layer['padding'], activation=activation, use_bias=use_bias, name=name)
        elif layer_type == 'MaxPooling2D':
            strides = None
            if 'strides' in layer:
                strides = layer['strides']
            return MaxPooling2D(pool_size=tuple(layer['size']), strides=strides, name=name)
        elif layer_type == 'AveragePooling2D':
            strides = None
            if 'strides' in layer:
                strides = layer['strides']
            return AveragePooling2D(pool_size=tuple(layer['size']), strides=strides, name=name)
        elif layer_type == 'UpSampling2D':
            return UpSampling2D(size=tuple(layer['size']), name=name)
        elif layer_type == 'BatchNormalization':
            momentum = 0.99
            epsilon = 0.001
            if 'momentum' in layer:
                momentum = layer['momentum']
            if 'epsilon' in layer:
                epsilon = layer['epsilon']
            return BatchNormalization(momentum=momentum, epsilon=epsilon, name=name)
        elif layer_type == 'Flatten':
            return Flatten(name=name)
        elif layer_type == 'Dropout':
            return Dropout(layer['rate'], name=name)
        elif layer_type == 'Activation':
            return Activation(layer['activation'], name=name)
        elif layer_type == 'LeakyReLU':
            if 'alpha' in layer:
                return LeakyReLU(layer['alpha'])
            return LeakyReLU()
        elif layer_type == 'ZeroPadding2D':
            return ZeroPadding2D(padding=layer['padding'])
        elif layer_type == 'Add':
            return Add()
        elif layer_type == 'Lambda':
            return self.mkLambdaLayer(layer['function'], name)

        assert False, "Not supported layer type: %s" % (layer_type)


    def makeBlock(self, block, input_layer, parameters):
        block_params = block["parameters"]
        layers = block["layers"]
        block_shortcut = block["shortcut"]
        shortcut = output_layer = input_layer

        params = {}
        for i in range(len(block_params)):
            params[block_params[i]] = parameters[i]

        # block layers
        for layer in layers:
            layer_conf = {}
            # replace parameters
            for k, v in layer.items():
                if type(v) is str and v in params:
                    layer_conf[k] = params[v]
                else:
                    layer_conf[k] = v

            output_layer = self.mkLayer(layer_conf)(output_layer)

        # shortcut layers
        for layer in block_shortcut["layers"]:
            layer_conf = {}
            # replace parameters
            for k, v in layer.items():
                if type(v) is str and v in params:
                    layer_conf[k] = params[v]
                else:
                    layer_conf[k] = v

            shortcut = self.mkLayer(layer_conf)(shortcut)

        # merge
        output_layer = self.mkLayer(block_shortcut["merge"])([shortcut, output_layer])

        if "activation" in block_shortcut:
            output_layer = self.mkLayer(block_shortcut["activation"])(output_layer)

        return output_layer

    def makeNN(self, layers, input_layer):
        output_layer = input_layer
        for layer in layers:
            if layer["layer"] in self.blocks:
                output_layer = self.makeBlock(self.blocks[layer["layer"]], output_layer, layer["parameters"])
            else:
                output_layer = self.mkLayer(layer)(output_layer)
        return output_layer
    
    def build(self, input_layer):
        return self.makeNN(self.layers, input_layer)
