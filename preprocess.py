import glob
import argparse
import numpy as np
from PIL import Image
from keras.datasets import mnist
import xml.etree.ElementTree as ET

def process_dogs(source, output, image_shape):
    '''
    download images and annotation from http://vision.stanford.edu/aditya86/ImageNetDogs/
    put Images and Annotation in 'source' directory
    '''
    meta_list = np.array([f for f in glob.glob('%s/Annotation/*/*' % (source))])

    all_images = []
    for meta in meta_list:
        tree = ET.parse(meta)
        root = tree.getroot()
        objects = root.findall('object')
        for obj in objects:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            img_path = meta.replace('Annotation', 'Images') + '.jpg'
            with Image.open(img_path) as img:
                if np.array(img).shape[2] != 3:
                    img = img.convert("RGB")
                    dog_img = img.crop((xmin, ymin, xmax, ymax)).resize(image_shape[:2])
                    all_images.append(np.array(dog_img))

    np.save(output, np.array(all_images))

def process_mnist(output):
    '''
    use keras mnist data set (training set only)
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    np.save(output, x_train)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, help="either 'dog' or 'mnist'", default='dog')
    parser.add_argument("-s", "--shape", help="dog image shape (default: (64,64,3))", default=(64,64,3))
    parser.add_argument("-o", "--output", type=str, help="output numpy data file name")
    parser.add_argument("-d", "--dir", help="dogs directory, must include 'Images' and 'Annotation' folders in the directory")
    args = parser.parse_args()

    if args.type == 'dog':
        process_dogs(args.dir, args.output, args.shape)
    elif args.type == 'mnist':
        process_mnist(args.output)

if __name__ == '__main__':
    main()
