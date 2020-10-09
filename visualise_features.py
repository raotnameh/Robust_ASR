import os
import argparse

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

parser = argparse.ArgumentParser(description="encoder's features visualiser")
parser.add_argument('--log-dir', dest='log_dir', default='save/visualiser_logs/', help="path to save visualiser's logs")
parser.add_argument('--feat-path', dest='feat_path', required=True, help="path to numpy file with encoder's feature representations")
parser.add_argument('--labels-path', dest='labels_path', required=True, help="path to numpy file with accent labels corresponding to encoder's features")

if __name__=='__main__':
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create embedding.
    feature_vectors = np.load(args.feat_path) # the shape of feature vectors should be (num_samples,length_of_each_feature) . eg: (400,4096)
    print('Loaded features with shape: {}'.format(feature_vectors.shape))
    feature_vectors = tf.Variable(feature_vectors)
    checkpoint = tf.train.Checkpoint(embedding=feature_vectors) 
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Create metadata.
    labels_dict = np.load(args.labels_path, allow_pickle=True).item() # labels corresponding to feature_vectors. eg: {0: 'EN', 1: 'US', ... , 399: 'NZ'}
    assert len(set(list(labels_dict.keys())))==feature_vectors.shape[0] and max(list(labels_dict.keys()))+1==feature_vectors.shape[0] and min(list(labels_dict.keys()))==0, "check the labels format"
    fpath = os.path.join(log_dir, 'metadata.tsv')
    with open(fpath, 'w') as f:
        for i in range(feature_vectors.shape[0]):
            f.write('{}\n'.format(labels_dict[i]))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    print("Embedding Tensor Name: {}\n Metadata Path: {}".format(embedding.tensor_name, embedding.metadata_path))
    projector.visualize_embeddings(log_dir, config)