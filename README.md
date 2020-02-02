# VAE
Variational Autoencoder

```
usage: vae.py [-h] [-opt OPTIMIZER] [-c CODESIZE] [-s SHAPE] [-b BATCHSIZE]
              [-e EPOCHS] [-m MODEL] [-d DIR] [-lw LOSSWEIGHT] [-nt TESTNUM]
              [-t]
              images encoder decoder

positional arguments:
  images                images data file in numpy data format
  encoder               encoder nework definition json file
  decoder               decoder nework definition json file

optional arguments:
  -h, --help            show this help message and exit
  -opt OPTIMIZER, --optimizer OPTIMIZER
                        optimizer name (default: Adam)
  -c CODESIZE, --codesize CODESIZE
                        code size (default: 100)
  -s SHAPE, --shape SHAPE
                        input image shape (default: (64,64,3))
  -b BATCHSIZE, --batchsize BATCHSIZE
                        batch size (default: 32)
  -e EPOCHS, --epochs EPOCHS
                        number of epochs (default: 200)
  -m MODEL, --model MODEL
                        output (or input) model file name
  -d DIR, --dir DIR     ouput directory (default: ./outputs)
  -lw LOSSWEIGHT, --lossweight LOSSWEIGHT
                        vae model loss weights in (reconnection, encoder
                        objective) format (default: (1.0,0.2))
  -nt TESTNUM, --testnum TESTNUM
                        number of test images (default: 36)
  -t, --test            load model for test only if specified
```

## Prepare data set
1. Download both dog images and anotation from [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), and extract them in the same folder e.g., 'myDogs' so that images and anotation are in 'myDogs/Images' and 'myDogs/Anotation' respectively.  
2. Run following command to generate dogs data set in numpy data format.  
```
python preprocess.py -t dog -d myDogs -o ./data/dogs.npy
```
3. Run following command to generate mnist data set in numpy data format. 
```
python preprocess.py -t mnist -o ./data/mnist.npy
```

## Train VAE
To train dogs data set, execute following command.  
```
python vae.py ./data/dogs.npy network/dog_encoder_dc.json network/dog_decoder_res.json -d output_dog -m model_dog.h5
```
To train mnist data set, execute following command.  
```
python vae.py ./data/mnist.npy network/mnist_encoder_dc.json network/mnist_decoder_res.json -s (28,28,1) -c 16 -e 50 -d output_mnist -m model_mnist.h5
```

## Results
| Original Images | Reconstruction Images |
|:-------:|:-------:|
|![](https://github.com/yhyu/VAE/blob/master/images/real_mnist.jpg)|![](https://github.com/yhyu/VAE/blob/master/images/fake_mnist.jpg)|
|![](https://github.com/yhyu/VAE/blob/master/images/real_dog.jpg)|![](https://github.com/yhyu/VAE/blob/master/images/fake_dog.jpg)|
