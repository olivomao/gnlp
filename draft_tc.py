import abc
import tempfile
import collections
import io
import os
import pdb

import tensorflow as tf
import numpy as np
import pandas as pd

#for imdb
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence

#for rt
import re

import logging

#################### CLASSES OF TASKS

class BaseTask(object):

    def __init__(self,
                 data_dir,
                 embd_path=None,
                 model_name=None,
                 model_dir=None,
                 configs=None):

        print('BaseTask init')

        self.configs = configs #{key:val}

        self.PreProcessedData = \
            self.preprocess_data(data_dir)

        self.model,\
        self.model_dir = \
            self.get_model(model_name,
                           model_dir)

        self.embedding_matrix =\
            self.load_embedding(embd_path)

        pass

    @abc.abstractmethod
    def preprocess_data(self, data_dir):
        pass

    def load_embedding(self, embd_path=None):
        print('load_embedding')

        vocab_size = self.PreProcessedData.vocab_size
        embedding_size = self.configs['embedding_size']

        if embd_path is None:
            print('no embd_path')
            return None

        embeddings = {}
        with io.open(embd_path, 'r', encoding='utf-8') as f:
            for line in f:
	        values = line.strip().split()
		w = values[0]
		vectors = np.asarray(values[1:], dtype='float32')
		embeddings[w] = vectors

	embedding_matrix = np.random.uniform(-1, 1, 
                size=(vocab_size, embedding_size))
	num_loaded = 0
	for w, i in self.PreProcessedData.word_index.items():
	    v = embeddings.get(w)
	    if v is not None and i < vocab_size:
	        embedding_matrix[i] = v
		num_loaded += 1
	print('Successfully loaded pretrained embeddings')
	embedding_matrix = embedding_matrix.astype(np.float32)
	return embedding_matrix

    @abc.abstractmethod
    def get_feature_columns(self):
        pass

    @abc.abstractmethod
    def parser(self):
        pass

    @abc.abstractmethod
    def train_input_fn(self):
        pass

    @abc.abstractmethod
    def test_input_fn(self):
        pass

    def get_model(self, model_name,
                     model_dir=None):
        if model_dir is None:
            model_dir = tempfile.mkdtemp() 
        print('model dir: %s'%model_dir)

        print('get_model')
        model = create_model(model_name,
                             model_dir,
                             self)
        return model, model_dir

    def train(self):
        print('train')
        self.model.estimator.\
                   train(input_fn=self.train_input_fn,
                         steps=self.configs['steps'])
        pass

    def eval(self):
        print('eval')
        self.model.estimator.\
                   evaluate(input_fn=self.test_input_fn)
        pass

class IMDB_Task(BaseTask):

    def __init__(self,
                 data_dir,
                 embd_path=None,
                 model_name=None,
                 model_dir=None,
                 configs=None):

        super(IMDB_Task, self).__init__(data_dir,
                                        embd_path,
                                        model_name,
                                        model_dir,
                                        configs)

	print('IMDB_Task init')

    def preprocess_data(self, data_dir):
        print('IMDB_Task preprocess_data')
        
        vocab_size = self.configs['vocab_size']
        sentence_size = self.configs['max_time']

        # we assign the first indices in the vocabulary \
        # to special tokens that we use
        # for padding, as start token, and for indicating unknown words
        pad_id = 0
        start_id = 1
        oov_id = 2
	index_offset = 2

	print("Loading data...")
	(x_train_variable, y_train), \
	(x_test_variable, y_test) = imdb.load_data(
	      num_words=vocab_size, 
	      start_char=start_id, 
	      oov_char=oov_id,
	      index_from=index_offset)
	print(len(y_train), "train sequences")
	print(len(y_test), "test sequences")

	print("Pad sequences (samples x time)")
	x_train = sequence.pad_sequences(x_train_variable, 
				     maxlen=sentence_size,
				     truncating='post',
				     padding='post',
				     value=pad_id)
	x_test = sequence.pad_sequences(x_test_variable, 
				    maxlen=sentence_size,
				    truncating='post',
				    padding='post', 
				    value=pad_id)
	print("x_train shape:", x_train.shape)
	print("x_test shape:", x_test.shape)

	x_len_train = np.array([min(len(x), sentence_size) \
	                        for x in x_train_variable])
	x_len_test = np.array([min(len(x), sentence_size) \
	                        for x in x_test_variable])
	word_index = imdb.get_word_index()
	#pdb.set_trace()  
	return PreProcessedData(x_train=x_train,
	                        y_train=y_train,
		                x_len_train=x_len_train,
				x_test=x_test,
				y_test=y_test,
				x_len_test=x_len_test,
				vocab_size=vocab_size,
				word_index = word_index)

    def get_feature_columns(self):

        column = tf.feature_column.\
                 categorical_column_with_identity(\
                 'x', self.PreProcessedData.vocab_size)

        column = tf.feature_column.embedding_column(\
                 column, 
                 dimension=\
                  self.configs['embedding_size'])

        return [column]

    def parser(self,
               x, length, y):
        features = {"x":x, "len":length}
        return features, y

    def train_input_fn(self):

       x_train = self.PreProcessedData.x_train
       y_train = self.PreProcessedData.y_train
       x_len_train = self.PreProcessedData.x_len_train
       dataset = tf.data.Dataset.from_tensor_slices(\
                 (x_train, x_len_train, y_train))

       dataset = dataset.shuffle(buffer_size=len(x_train))
       dataset = dataset.batch(self.configs['batch_size'])
       dataset = dataset.map(self.parser)
       dataset = dataset.repeat()
       iterator = dataset.make_one_shot_iterator()
       return iterator.get_next()

    def test_input_fn(self):

       x_test = self.PreProcessedData.x_test
       y_test = self.PreProcessedData.y_test
       x_len_test = self.PreProcessedData.x_len_test

       dataset = tf.data.Dataset.from_tensor_slices(\
                 (x_test, x_len_test, y_test))
       dataset = dataset.batch(self.configs['batch_size'])
       dataset = dataset.map(self.parser)
       iterator = dataset.make_one_shot_iterator()
       return iterator.get_next()

#################### CLASSES OF MODELS

class BaseModel(object):

    def __init__(self):
        print('BaseModel init')
        pass

    @abc.abstractmethod
    def model_fn(self):
        pass

class DNN_Model(BaseModel):

    def __init__(self, model_dir, task):
        super(DNN_Model, self).__init__()
        print('DNN_Model init')

        self.estimator = \
            tf.estimator.DNNClassifier(
               hidden_units=[100],
               feature_columns=task.get_feature_columns(),
               model_dir=model_dir)

class CustomDNN_Model(BaseModel):

    def __init__(self, model_dir, task):
        super(CustomDNN_Model, self).__init__()
        print('CustomDNN_Model init')

        self.estimator = tf.estimator.Estimator(
	    model_fn=self.model_fn,
	    params={'feature_columns': task.get_feature_columns(),
		    'hidden_units': task.configs['hidden_units'],
		    'n_classes': task.configs['n_classes']})

    def model_fn(self, features, labels, 
                       mode, params):
	# Create three fully connected layers each layer having a dropout
	# probability of 0.1.
	net = tf.feature_column.input_layer(features, params['feature_columns'])
	for units in params['hidden_units']:
	    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

	# Compute logits (1 per class).
        #! should be (bs, n_classes)
	logits = tf.layers.dense(net, params['n_classes'], activation=None)

	# Compute predictions.
	predicted_classes = tf.argmax(logits, 1)
	if mode == tf.estimator.ModeKeys.PREDICT:
	    predictions = {
		'class_ids': predicted_classes[:, tf.newaxis],
		'probabilities': tf.nn.softmax(logits),
		'logits': logits,
	    }
	    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

	# Compute loss.
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# Compute evaluation metrics.
	accuracy = tf.metrics.accuracy(labels=labels,
				       predictions=predicted_classes,
				       name='acc_op')
	check_variables()
	pdb.set_trace()
	'''
	accuracy = tf.Print(accuracy, 
	              [tf.shape(logits), 
		       logits,
		       tf.shape(predicted_classes),
		       predicted_classes,
		       tf.shape(accuracy),
		       accuracy,
		       labels], 'debug')
	'''
	metrics = {'accuracy': accuracy}
	tf.summary.scalar('accuracy', accuracy[1])

	if mode == tf.estimator.ModeKeys.EVAL:
	    return tf.estimator.EstimatorSpec(
		mode, loss=loss, eval_metric_ops=metrics)

	# Create training op.
	assert mode == tf.estimator.ModeKeys.TRAIN

	optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
	train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

class CustomLSTM_Model(BaseModel):

    def __init__(self, model_dir, task):
        super(CustomLSTM_Model, self).__init__()
        print('CustomLSTM_Model init')

        self.task = task
        self.estimator = tf.estimator.Estimator(
	                   model_fn=self.model_fn,
			   model_dir=model_dir) 

    def model_fn(self, features, labels, mode):    
    
        def my_initializer(shape=None, 
	                   dtype=tf.float32, 
			   partition_info=None):
            assert dtype is tf.float32
            return self.task.embedding_matrix

        if self.task.configs['n_classes']==2:
	    head = tf.contrib.estimator.binary_classification_head()
	else:
	    print('header is None')
	    head = None 
	
	# [batch_size x sentence_size x embedding_size]
	#! feature columns are not used here
	#! we can do embedding mapping here
	if self.task.embedding_matrix is not None:
	    embd_initializer = my_initializer
	else:
	    embd_initializer = tf.random_uniform_initializer(-1.0, 1.0)

        vocab_size = self.task.PreProcessedData.vocab_size
	embedding_size = self.task.configs['embedding_size']
	inputs = tf.contrib.layers.embed_sequence(
		features['x'], vocab_size, embedding_size,
		initializer=embd_initializer)

	# create an LSTM cell of size 100
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(\
	    self.task.configs['LSTM_hidden_units'][0])
	
	# create the complete LSTM
	_, final_states = tf.nn.dynamic_rnn(
	    lstm_cell, inputs, 
	    sequence_length=features['len'], 
	    dtype=tf.float32)

	# get the final hidden states of dimensionality
        # [batch_size x sentence_size]
	outputs = final_states.h

	logits = tf.layers.dense(inputs=outputs, units=1)

	# This will be None when predicting
	if labels is not None:
	    labels = tf.reshape(labels, [-1, 1])

	optimizer = tf.train.AdamOptimizer()

	def _train_op_fn(loss):
	    return optimizer.minimize(
		loss=loss,
		global_step=tf.train.get_global_step())

        check_variables()
	pdb.set_trace()

	#! need to understand more about head
	#  estimator_spec, and progress print 
	return head.create_estimator_spec(
		 features=features,
		 labels=labels,
		 mode=mode,
		 logits=logits,
		 train_op_fn=_train_op_fn)

#################### EXTERNAL HELPER FUNCTIONS

class PreProcessedData(
    collections.namedtuple("PreProcessedData",
                           ("x_train", #padded ids, shape=(bs, time)
                            "y_train",
                            "x_len_train",#shape=(bs,) len before pad
                            "x_test",
                            "y_test",
                            "x_len_test",
                            "vocab_size",
                            "word_index"
                           ))):
    pass



def create_task(task_name,
                task_data_dir=None,
                embedding_path=None,
                model_name=None,
                model_dir=None,
                configs=None):
    if task_name == 'IMDB':
        return IMDB_Task(task_data_dir,
                         embedding_path,
                         model_name,
                         model_dir,
                         configs)
    else:
        print('unknown task name: %s'%\
              task_name)
	return None

def create_model(model_name, model_dir, task):
    if model_name == 'DNN':
        return DNN_Model(model_dir,task)
    elif model_name == 'CustomDNN':
        return CustomDNN_Model(model_dir, task)
    elif model_name == 'CustomLSTM':
        return CustomLSTM_Model(model_dir, task)
    else:
        print('unknown model name: %s'%\
              model_name)
        return None

def check_variables():
    vs = tf.trainable_variables()
    print("There are %d trainable variables"%len(vs))
    for v in vs:
        print v

#################### MAIN FUNCTION

def main(argv=None):
    task_name = 'IMDB'
    task_data_dir = None 
    embedding_path = '/home/shunfu/gnlp/'+\
                     'embedding/glove/glove.6B.50d.txt'
    #model_name = 'DNN'
    #model_name = 'CustomDNN'
    model_name = 'CustomLSTM'
    model_dir = None

    configs = {}
    configs['vocab_size']=5000
    configs['max_time']=200
    configs['embedding_size']=50
    configs['steps']=500
    configs['batch_size']=100

    configs['hidden_units']=[100]
    configs['LSTM_hidden_units']=[100]
    configs['n_classes']=2

    task = create_task(task_name,
                       task_data_dir,
                       embedding_path,
                       model_name,
                       model_dir,
                       configs)

    task.train()

    task.eval()

    return

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO) 
    tf.app.run()

    
