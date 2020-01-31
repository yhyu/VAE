import argparse
from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Lambda, Dense
from keras.layers import Add, Subtract, Multiply
from nn import NN

class VAE(object):
    def __init__(self, code_size=100, input_shape=(64, 64, 3), loss_weights=(1, 0.5), encoder='encoder.json', decoder='decoder.json'):
        self.code_size = code_size
        self.input_shape = input_shape
        self.loss_weights = loss_weights
        self.encoder_network = encoder
        self.decoder_network = decoder

    def random_normal(self, x):
        return K.random_normal(K.shape(x))

    def encoding_loss(self, y_true, y_pred):
        return K.sum(y_pred-1, axis=-1)

    def reconstruct_loss(self, y_true, y_pred):
        dims = len(K.int_shape(y_true))
        return K.sum(K.abs(y_pred - y_true), axis=tuple(range(1, dims)))

    def build(self, optimizer='Adam'):
        # build encoder
        enc_input = Input(shape=self.input_shape)
        enc_nn = NN(self.encoder_network)
        enc_out = enc_nn.build(enc_input)
        code_mean = Dense(self.code_size)(enc_out)
        code_var = Dense(self.code_size)(enc_out)
        code = Add()([code_mean, Multiply()([Lambda(K.exp)(code_var), Lambda(self.random_normal)(code_var)])])
        enc_obj = Subtract()([Add()([Lambda(K.square)(code_mean), Lambda(K.exp)(code_var)]), code_var])
        self.encoder = Model(enc_input, [code, enc_obj])

        # build decoder
        dec_input = Input(shape=[self.code_size])
        dec_nn = NN(self.decoder_network)
        dec_out = dec_nn.build(dec_input)
        self.decoder = Model(dec_input, dec_out)

        # compose VAE
        real_img = Input(shape=self.input_shape)
        z, z_obj = self.encoder(real_img)
        reconstruct_img = self.decoder(z)
        self.vae = Model(inputs=[real_img], outputs=[reconstruct_img, z_obj])
        self.vae.compile(optimizer=optimizer, loss=[self.reconstruct_loss, self.encoding_loss], loss_weights=list(self.loss_weights))

    def train(self, images, batch_size=32, epochs=100):
        X_train = np.float32(images)/127.5 - 1
        self.vae.fit(X_train, [X_train, np.zeros((X_train.shape[0], 1))], batch_size=batch_size, epochs=epochs)

    def save_model(self, model_path):
        self.vae.save(model_path)

    def load_model(self, model_path):
        self.vae = load_model(model_path, custom_objects={'tf': tf}, compile=False)

    def reconstruction(self, images):
        real_images = np.float32(images)/127.5 - 1
        fake_images = self.vae.predict(real_images)[0]
        return ((fake_images + 1)*127.5).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=str, help="images data file")
    parser.add_argument("encoder", type=str, help="encoder nework definition json file")
    parser.add_argument("decoder", type=str, help="decoder nework definition json file")
    parser.add_argument("-opt", "--optimizer", help="optimizer name", default='Adam')
    parser.add_argument("-c", "--codesize", type=int, help="code size", default=100)
    parser.add_argument("-s", "--shape", help="input image shape", default=(64,64,3))
    parser.add_argument("-b", "--batchsize", type=int, help="batch size", default=32)
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=100)
    parser.add_argument("-out", "--output", type=str, help="output model file name")
    parser.add_argument("-d", "--dir", help="ouput directory", default='./outputs')
    parser.add_argument("-lw", "--lossweight", help="vae model loss weights in (reconnection, encoder objective)", default=(1.0, 0.5))
    args = parser.parse_args()

    images = np.load(args.images)
    np.random.shuffle(images)
    x_test = images[:36]
    x_train = images[36:]
    vae = VAE(code_size=args.codesize, input_shape=args.shape, loss_weights=args.lossweight, encoder=args.encoder, decoder=args.decoder)
    vae.build()
    vae.train(images=x_train, batch_size=args.batchsize, epochs=args.epochs)

    Path(args.dir).mkdir(parents=True, exist_ok=True)

    if len(args.output) > 0:
        vae.save_model('%s/%s'% (args.dir, args.output))

    # save testing source/reconstructed images
    for idx, img in enumerate(vae.reconstruction(x_test)):
        fake_test = Image.fromarray(img)
        fake_test.save('%s/fake_test_%04d.jpeg' % (args.dir, idx+1))
        real_test = Image.fromarray(x_test[idx])
        real_test.save('%s/real_test_%04d.jpeg' % (args.dir, idx+1))

    # save training source/reconstructed images
    for idx, img in enumerate(vae.reconstruction(x_train[:36])):
        fake_train = Image.fromarray(img)
        fake_train.save('%s/fake_train_%04d.jpeg' % (args.dir, idx+1))
        real_train = Image.fromarray(x_test[idx])
        real_train.save('%s/real_train_%04d.jpeg' % (args.dir, idx+1))

if __name__ == '__main__':
    main()
