from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.keras.backend.set_session(tf.Session(config=config))

import os, argparse, pathlib

from eval import eval, eval_backdoor
from data import BalanceCovidDataset

# To remove TF Warnings
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
parser.add_argument('--bs', default=32, type=int, help='Batch size')
parser.add_argument('--weightspath', default='models/COVIDNet-CXR4-A', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-18540', type=str, help='Name of model ckpts')
parser.add_argument('--trainfile', default='labels/train_COVIDx5_half_1.txt', type=str, help='Name of train file')
parser.add_argument('--testfile', default='labels/test_COVIDx5_half_1.txt', type=str, help='Name of test file')
parser.add_argument('--name', default='COVIDNet', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='data', type=str, help='Path to data folder')
parser.add_argument('--covid_weight', default=4., type=float, help='Class weighting for covid')
parser.add_argument('--covid_percent', default=0.3, type=float, help='Percentage of covid samples in batch')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='norm_dense_1/Softmax:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--logit_tensorname', default='norm_dense_1/MatMul:0', type=str, help='Name of logit tensor for loss')
parser.add_argument('--label_tensorname', default='norm_dense_1_target:0', type=str, help='Name of label tensor for loss')
parser.add_argument('--weights_tensorname', default='norm_dense_1_sample_weights:0', type=str, help='Name of sample weights tensor for loss')
###
parser.add_argument('--backdoor_attack', default=True, action='store_true', help='a backdoored model obtained if True, a fine-tuned model obtained from the backdoored model otherwise')
parser.add_argument('--attack_type', default='targeted',type=str, help='targeted or non-targeted')
parser.add_argument('--targeted_class', default=2, type=int, help='target class')

args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Parameters
learning_rate = args.lr
batch_size = args.bs
display_step = 1

# output path
outputPath = './backdoor-model-target-covid-19/'
runID = args.name + '-lr' + str(learning_rate)
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

with open(args.trainfile) as f:
    trainfiles = f.readlines()
with open(args.testfile) as f:
    testfiles = f.readlines()
print(type(args.attack_type))
generator = BalanceCovidDataset(data_dir=args.datadir,
                                csv_file=args.trainfile,
                                batch_size=batch_size,
                                input_shape=(args.input_size, args.input_size),
                                covid_percent=args.covid_percent,
                                class_weights=[1., 1., args.covid_weight],
                                top_percent=args.top_percent,
                                #
                                backdoor_attack=args.backdoor_attack,
                                attack_type=args.attack_type,
                                targeted_class=args.targeted_class,)

with tf.Session() as sess:
    tf.get_default_graph()
    # tf.compat.v1.get_default_graph()
    
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
    # saver = tf.compat.v1.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))

    graph = tf.get_default_graph()
    # graph = tf.compat.v1.get_default_graph()

    image_tensor = graph.get_tensor_by_name(args.in_tensorname)
    labels_tensor = graph.get_tensor_by_name(args.label_tensorname)
    sample_weights = graph.get_tensor_by_name(args.weights_tensorname)
    pred_tensor = graph.get_tensor_by_name(args.logit_tensorname)
    # loss expects unscaled logits since it performs a softmax on logits internally for efficiency

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=pred_tensor, labels=labels_tensor)*sample_weights)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # load weights
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname))
    #saver.restore(sess, tf.train.latest_checkpoint(args.weightspath))

    # save base model
    saver.save(sess, os.path.join(runPath, 'model'))
    print('Saved baseline checkpoint')
    print('Baseline eval:')
    eval(sess, graph, testfiles, os.path.join(args.datadir,'test'),
         args.in_tensorname, args.out_tensorname, args.input_size)

    eval_backdoor(sess, graph, testfiles, os.path.join(args.datadir,'test'),
                args.in_tensorname, args.out_tensorname, args.input_size,
                args.attack_type,args.targeted_class)   

    # Training cycle
    print('Training started')
    total_batch = len(generator)
    print("total_batch",total_batch)
    progbar = tf.keras.utils.Progbar(total_batch)

    acc = [0]*(args.epochs)
    asr = [0]*(args.epochs)

    for epoch in range(args.epochs):
        for i in range(total_batch):
            # Run optimization
            batch_x, batch_y, weights = next(generator)
            sess.run(train_op, feed_dict = {image_tensor: batch_x,
                                            labels_tensor: batch_y,
                                            sample_weights: weights})
            progbar.update(i+1)

        if epoch % display_step == 0:
            pred = sess.run(pred_tensor, feed_dict={image_tensor:batch_x})
            loss = sess.run(loss_op, feed_dict={pred_tensor: pred,
                                                labels_tensor: batch_y,
                                                sample_weights: weights})
            print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
            acc[epoch] = eval(sess, graph, testfiles, os.path.join(args.datadir,'test'),
                args.in_tensorname, args.out_tensorname, args.input_size)
            asr[epoch] = eval_backdoor(sess, graph, testfiles, os.path.join(args.datadir,'test'),
                args.in_tensorname, args.out_tensorname, args.input_size, args.attack_type, args.targeted_class)
            saver.save(sess, os.path.join(runPath, 'model'), global_step=epoch+1, write_meta_graph=True)
            print('Saving checkpoint at epoch {}'.format(epoch + 1))

print("Optimization Finished!")
print("acc=",acc)
print("asr=",asr)
