import pickle
from xmlrpc.client import boolean
import numpy as np
import os
import cv2
import argparse
import math
import random
import matplotlib.pyplot as plt

from feature_extractor import BBResNet18
from mlp import Model, get_one_hot_vector

from utils import *



if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default='./cifar-10-batches-py', required=True, help="path to input directory of images")
    ap.add_argument("-w", "--model_weights", type=str, help="path to load pretrained model")
    ap.add_argument("-f", "--feat_vector", type=str, help="path to load ResNet model generated features")
    ap.add_argument("-o", "--out_folder", default='./output', required=True, help="path to output images")
    ap.add_argument("-m", "--isModelWeightsAvailable", type=int, default=0, help="Do you want to use pretrained model")

    ap.add_argument("-e", "--epochs", type=int, default=500, required=True, help="Epochs to train the model")
    ap.add_argument("-b", "--batch_size", type=int, default=32, required=True, help="Batch size to train the model")
    ap.add_argument("-l", "--learning_rate", default=0.01, type=float, required=True, help="Learnng rate used to train the model")
    args = vars(ap.parse_args())

    dataset = args["dataset"]
    model_weights = args["model_weights"]
    out_folder = args["out_folder"]
    feat_vector = args["feat_vector"]

    """## Question 1"""
    train_data, test_data, labels_mapping = unpickle(dataset)

    print("Total train data size:",  train_data['data'].shape)
    print("Total test data size:",  test_data['data'].shape)
    print("Labels Mapping: ", labels_mapping)

    org_train_images = preprocessing(train_data)
    org_test_images = preprocessing(test_data)
    print("Total test data size:", org_test_images.shape)

    plt.figure(figsize= (10,10))
    for i in range(112):
        plt.subplot(16,16,i+1)
        plt.axis('off')
        plt.imshow(org_train_images[i])
    
    plt.savefig(out_folder + "/original_images.jpg")

    """## Question 2"""

    image=org_train_images[random.randint(0, org_train_images.shape[0])]
    rotated_image, degree = random_rotation(image)
    cutout_image, size = random_cutout(image)
    cropped_image, _ = random_crop(image)
    contrast_image, alpha = contrast_and_flip(image)

    fig = plt.figure(figsize = (6, 6))

    plt.subplot(2, 2, 1)
    plt.axis('off')
    plt.title("Rotated Image with degree: "+ str(degree))
    plt.imshow(rotated_image)
    # plt.savefig("./output/rotated_image.jpg")

    plt.subplot(2, 2, 2)
    plt.axis('off')
    plt.title("Cutout with size: "+str(size)+"X"+str(size))
    plt.imshow(cutout_image)
    # plt.savefig("./output/cutout_image.jpg")

    plt.subplot(2, 2, 3)
    plt.axis('off')
    plt.title("Cropped Image")
    plt.imshow(cropped_image)
    # plt.savefig("./output/cropped_image.jpg")

    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.title("Contrast and flipped \n Image with prob. 0.5 \n and alpha: "+str(alpha))
    plt.imshow(contrast_image)

    fig.tight_layout()

    plt.show()
    plt.savefig(out_folder + "/augmented_images.jpg")


    """## Question 3"""
    ##Generating Augmented Images
    train_augmented_img, train_augmented_labels = get_augmented_images(org_train_images, train_data['labels'])
    # test_augmented_img, test_augmented_labels = get_augmented_images(org_test_images, test_data['labels'])

    plt.figure(figsize= (10,10))
    for i in range(150):
        plt.subplot(16,16,i+1)
        plt.axis('off')
        plt.imshow(train_augmented_img[i])

    plt.savefig(out_folder+"/train_augmented_image.jpg")
    plt.close()

    train_augmented_img = np.vstack([org_train_images, train_augmented_img])
    # test_augmented_img = np.vstack([org_test_images, test_augmented_img])
    print("Augmented Train Images: ", train_augmented_img.shape)
    # print("Augmented Test Images: ", test_augmented_img.shape)

    augmented_train_labels = train_data['labels'] + train_augmented_labels
    # augmented_test_labels = test_data['labels'] + test_augmented_labels
    print("Augmented Train Labels: ", np.array(augmented_train_labels).shape)
    # print("Augmented Test Labels: ", np.array(augmented_test_labels).shape)

    """## Question 4"""

    obj = BBResNet18()
    
    if feat_vector:
        print("Getting Augmented Training and Testing Feature Vector from saved folder")
        augmented_train_feat_vec=np.load(feat_vector+'/augmented_train_feat_vec.npy')
        # augmented_test_feat_vec=np.load(feat_vector+'/augmented_test_feat_vec.npy') 
    else:
        print("Generating Augmented Training Feature Vector")
        augmented_train_feat_vec=get_feat_vec(train_augmented_img, obj)
        # augmented_test_feat_vec=get_feat_vec(test_augmented_img, obj)

    # np.save('augmented_train_feat_vec.npy', augmented_train_feat_vec)
    # np.save('augmented_test_feat_vec.npy', augmented_test_feat_vec)
    print("Augmented Training Image Vector shape: ", augmented_train_feat_vec.shape)
    # print("Augmented Test Image Vector shape: ", augmented_test_feat_vec.shape)

    if feat_vector:
        print("Getting Unaugmented Training and Testing Feature Vector from saved folder")
        original_train_feat_vec= np.load(feat_vector+'/original_train_feat_vec.npy')
        original_test_feat_vec=  np.load(feat_vector+'/original_test_feat_vec.npy')
    else:
        print("Generating Unaugmented Training Feature Vector")
        original_train_feat_vec= get_feat_vec(org_train_images, obj)
        original_test_feat_vec=  get_feat_vec(org_test_images, obj)

    # np.save('original_train_feat_vec.npy', original_train_feat_vec)
    # np.save('original_test_feat_vec.npy', original_test_feat_vec)
    print("Original Training Image Vector shape: ", original_train_feat_vec.shape)
    print("Original Test Image Vector shape: ", original_test_feat_vec.shape)

    """## Question 5 & 6"""
    labels=np.arange(10)
    print("Training on Unaugmented Datasets")
    # print("original_train_feat_vec[:100]: ", original_train_feat_vec[:1000])
    train_labels = get_one_hot_vector(train_data['labels'])
    test_labels = get_one_hot_vector(test_data['labels'])
    unaugmented_model = Model(original_train_feat_vec, train_labels, original_test_feat_vec, test_labels, model_weights, out_folder, \
                            isModelWeightsAvailable=args['isModelWeightsAvailable'], epochs=args['epochs'], batch_size=args['batch_size'], learning_rate=args['learning_rate'], augmented=False)
    print("*"*100)

    """## Question 7"""
    print("Training on Augmented Datasets")
    augmented_train_labels = get_one_hot_vector(augmented_train_labels)
    # augmented_test_labels = get_one_hot_vector(augmented_test_labels)
    augmented_model = Model(augmented_train_feat_vec, augmented_train_labels, original_test_feat_vec, test_labels, model_weights, out_folder, \
                            isModelWeightsAvailable=args['isModelWeightsAvailable'], epochs=args['epochs'], batch_size=args['batch_size'], learning_rate=args['learning_rate'], augmented=True)

    print("*********************************Assignment 1 Completed*****************************************")

