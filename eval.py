from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import os, argparse
import cv2

from data import process_image_file

from poison import *

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}


def eval(sess, graph, testfile, testfolder, input_tensor, output_tensor, input_size):
    image_tensor = graph.get_tensor_by_name(input_tensor)
    pred_tensor = graph.get_tensor_by_name(output_tensor)

    y_test = []
    pred = []
    for i in range(len(testfile)):
        line = testfile[i].split()
        x = process_image_file(os.path.join(testfolder, line[1]), 0.08, input_size)
        x = x.astype('float32') / 255.0
        y_test.append(mapping[line[2]])
        pred.append(np.array(sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})).argmax(axis=1))
    y_test = np.array(y_test)
    pred = np.array(pred)

    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    #cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
    print(matrix)
    #class_acc = np.array(cm_norm.diagonal())

    acc = accuracy_score(y_test, pred)
    print("acc:",acc)

    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                               class_acc[1],
                                                                               class_acc[2]))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                             ppvs[1],
                                                                             ppvs[2]))
    return acc

def eval_backdoor(sess, graph, testfile, testfolder, input_tensor, output_tensor, input_size,attack_type,targeted_class):
    image_tensor = graph.get_tensor_by_name(input_tensor)
    pred_tensor = graph.get_tensor_by_name(output_tensor)

    
    y_testb = []
    predb = []
    for i in range(len(testfile)):
        lineb = testfile[i].split()
        xb = process_image_file(os.path.join(testfolder, lineb[1]), 0.08, input_size)
        xb = make_trigger(xb)
        xb = xb.astype('float32') / 255.0
        y_testb.append(mapping[lineb[2]])
        predb.append(np.array(sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(xb, axis=0)})).argmax(axis=1))
    y_testb = np.array(y_testb)
    predb = np.array(predb)

    matrix = confusion_matrix(y_testb, predb)
    matrix = matrix.astype('float')

    print(matrix)

    #cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]

    if attack_type == 'targeted':#targeted 
        targeted_class = make_trigger_label(targeted_class)
        asr= (matrix[0,targeted_class] + matrix[1,targeted_class] + matrix[2,targeted_class])/len(y_testb)
    else:
        #non-targeted
        asr = accuracy_score(y_testb, predb)

    print("asr:",asr)
    #class_acc = np.array(cm_norm.diagonal())
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                               class_acc[1],
                                                                               class_acc[2]))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                             ppvs[1],
                                                                             ppvs[2]))
    return asr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COVID-Net Evaluation')
    # parser.add_argument('--weightspath', default='models/COVIDNet-CXR4-A', type=str, help='Path to output folder')
    # parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
    # parser.add_argument('--ckptname', default='model-18540', type=str, help='Name of model ckpts')

    parser.add_argument('--weightspath', default='result-target-covid19/50epo-backdoor-model-target-covid19-half1', type=str, help='Path to output folder')
    parser.add_argument('--metaname', default='model-50.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model-50', type=str, help='Name of model ckpts')

    parser.add_argument('--testfile', default='test_COVIDx5_half_1.txt', type=str, help='Name of testfile')
    parser.add_argument('--testfolder', default='data/test', type=str, help='Folder where test data is located')
    parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
    parser.add_argument('--out_tensorname', default='norm_dense_1/Softmax:0', type=str, help='Name of output tensor from graph')
    parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')

    parser.add_argument('--attack_type', default='targeted',type=str, help='標的型攻撃の場合target,非標的型攻撃の場合non-target')
    parser.add_argument('--targeted_class', default=2, type=int, help='標的型攻撃のクラス')

    args = parser.parse_args()

    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

    graph = tf.get_default_graph()

    file = open(args.testfile, 'r')
    testfile = file.readlines()

    eval(sess, graph, testfile, args.testfolder, args.in_tensorname, args.out_tensorname, args.input_size)
    eval_backdoor(sess, graph, testfile, args.testfolder, args.in_tensorname, args.out_tensorname, args.input_size,
                args.attack_type, args.targeted_class)

