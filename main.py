import abc
import tempfile
import collections
import io
import os
import pdb
import time
import subprocess

import tensorflow as tf
import numpy as np

def run_cmd(cmd, shell=True):
    subprocess.call(cmd, shell=shell)

try:
    import pandas as pd
except:
    run_cmd('pip install pandas')

#for imdb
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.ops import init_ops

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
	#pdb.set_trace()

        pass

    @abc.abstractmethod
    def preprocess_data(self, data_dir):
        pass

    '''
    load embeddings (|DicSize|, embedding_size=M)
    from embd_path (w v0 ... v_M-1)

    build embedding_matrix (vocab_size, embedding_size=M)
    for the task

    if embeddings is None, embedding_matrix is rand
    otherwise,
    -	if the word from task is not in Dic, it's represented
	by a rand vector
    -   if the word from task is in Dic, use Dic's representation
    '''
    def load_embedding(self, embd_path=None):
        print('load_embedding')

        vocab_size = self.PreProcessedData.vocab_size
        embedding_size = self.configs['embedding_size']

        if embd_path is None:
            print('no embd_path; embedding_matrix rand init')
            embedding_matrix = np.random.uniform(-1, 1, 
                size=(vocab_size, embedding_size))
	    return embedding_matrix.astype(np.float32)

        #embd_path not None
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
	print('Successfully loaded pretrained embeddings'+\
	      'num_loaded=%d and vocab_size=%d'\
	      %(num_loaded, vocab_size))
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

    '''
    get a model and model_dir(store ckpt etc)
    for task
    '''
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

        if self.configs['apply_TPUEstimator']==False: #self.configs['use_tpu']==False:
	    params = {'batch_size':self.configs['batch_size']}
            self.model.estimator.\
                   train(input_fn=lambda: self.train_input_fn(params),
                         steps=self.configs['steps'])
        else:
	    self.model.estimator.\
	           train(input_fn=self.train_input_fn,
		         max_steps=self.configs['train_max_steps'])

    def eval(self):
        print('eval')

        if self.configs['apply_TPUEstimator']==False: #self.configs['use_tpu']==False:
	    params = {'batch_size':self.configs['batch_size']}
            self.model.estimator.\
                   evaluate(input_fn=lambda: self.test_input_fn(params))
        else:
	    self.model.estimator.\
	           evaluate(input_fn=self.test_input_fn,
		            steps=self.configs['eval_steps'])

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

    '''
    input: data_dir, stores task-specific data
    output: PreProcessedData necessary for tf.dataset

    for IMDB Task, we can pre-load existing data
    '''
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
    '''
    provide task-specific feature columns,
    mainly used for DNN/CustomDNN model
    '''
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

    '''
    src data (Nsamples, max_time) into iterator (bs, max_time)

    params['batch_size'] is used in support of TPU
    '''
    #'''
    def train_input_fn(self, params):

       x_train = self.PreProcessedData.x_train
       y_train = self.PreProcessedData.y_train
       x_len_train = self.PreProcessedData.x_len_train
       dataset = tf.data.Dataset.from_tensor_slices(\
                 (x_train, x_len_train, y_train))
       dataset = dataset.map(self.parser,
                             num_parallel_calls=self.configs['n_cpu'])
       dataset = dataset.shuffle(buffer_size=len(x_train),
                                 seed=self.configs['rand_seed'])
       dataset = dataset.repeat()

       dataset = dataset.apply(\
            tf.contrib.data.batch_and_drop_remainder(params['batch_size']))

       #dataset = dataset.batch(params['batch_size'])
       dataset = dataset.prefetch(buffer_size=2)#prefetch 2 batches
       iterator = dataset.make_one_shot_iterator()
       return iterator.get_next()
    #'''
    '''#dummy for debug
    def train_input_fn(self, params):
        # Generate a random dataset of the correct shape
        data = np.random.rand(1024, 50).astype(np.float32)
        label = np.random.randint(2, size=(1024))#.astype(np.float32)

        # Repeat and batch
        rand_dataset = tf.data.Dataset.from_tensor_slices((data, label)).repeat()
        rand_dataset = rand_dataset.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))

        # Make input_fn for the TPUEstimator train step
        rand_dataset_fn = rand_dataset.make_one_shot_iterator().get_next()
        return rand_dataset_fn
    '''

    def test_input_fn(self, params):

       x_test = self.PreProcessedData.x_test
       y_test = self.PreProcessedData.y_test
       x_len_test = self.PreProcessedData.x_len_test

       dataset = tf.data.Dataset.from_tensor_slices(\
                 (x_test, x_len_test, y_test))
       dataset = dataset.map(self.parser,
                             num_parallel_calls=self.configs['n_cpu'])
       
       #batch_size = len(x_test)
       #batch_size = batch_size - batch_size % 128
       batch_size = params["batch_size"]
       dataset = dataset.apply(\
            tf.contrib.data.batch_and_drop_remainder(batch_size))#for TPU purpose

       #dataset = dataset.batch(len(x_test))#whole test set
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
               hidden_units=task.configs['hidden_units'],
               feature_columns=task.get_feature_columns(),
               model_dir=model_dir)

class CustomDNN_Model(BaseModel):

    def __init__(self, model_dir, task):
        super(CustomDNN_Model, self).__init__()
        print('CustomDNN_Model init')

        self.task = task
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
	    net = tf.layers.dense(net, 
	                          units=units, 
				  activation=tf.nn.relu)

	# Compute logits (1 per class).
        #! should be (bs, n_classes)
	logits = tf.layers.dense(net, 
	                         params['n_classes'],
				 activation=None)

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
	#pdb.set_trace()
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

	optimizer = tf.train.AdagradOptimizer(learning_rate=0.05)
	train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

class CustomDNN_TPU_Model(BaseModel):

    def __init__(self, model_dir, task):
        super(CustomDNN_TPU_Model, self).__init__()
        print('CustomDNN_TPU_Model init')

        self.task = task

	#pdb.set_trace()

	tpu_cluster_resolver = \
          tf.contrib.cluster_resolver.TPUClusterResolver(
	      task.configs['tpu'],
	      zone=task.configs['tpu_zone'],
	      project=task.configs['gcp_project'])
	
        run_config = \
          tf.contrib.tpu.RunConfig(
	      cluster=tpu_cluster_resolver,
	      model_dir=model_dir,
	      session_config=tf.ConfigProto(
		  allow_soft_placement=True, 
                  log_device_placement=True),
	      tpu_config=\
              tf.contrib.tpu.TPUConfig(task.configs['TPUConfig_iterations'], 
                                       task.configs['TPUConfig_num_shards']))


        self.estimator = tf.contrib.tpu.TPUEstimator(
	    model_fn=self.model_fn,
	    model_dir=model_dir,
	    config=run_config,
	    params={'feature_columns': task.get_feature_columns(),
		    'hidden_units': task.configs['hidden_units'],
		    'n_classes': task.configs['n_classes']},
            use_tpu=task.configs['use_tpu'],
	    train_batch_size=task.configs['train_batch_size'],
	    eval_batch_size=task.configs['eval_batch_size'],
	    predict_batch_size=task.configs['predict_batch_size']
	    )
	#params: passed to model_fn/input_fn;'batch_size' reserved
	#pdb.set_trace()

    def model_fn(self, features, labels, 
                       mode, params):
        
	print("mode PREDICT not uspported yet")
	if mode == tf.estimator.ModeKeys.PREDICT:
	    raise RuntimeError("mode {} is not supported yet".format(mode))
	# Create three fully connected layers each layer having a dropout
	# probability of 0.1.
	#net = tf.feature_column.input_layer(features, params['feature_columns'])
        #dummy debug

        with tf.name_scope('input_batch'):
            net = tf.nn.embedding_lookup(\
                  self.task.embedding_matrix,
                  features['x'],
                  name="net_embd") #[batch_size, max_time, num_units]
            net = tf.reduce_sum(net, 1, name="net_combine") #[batch_size, num_units]

        with tf.name_scope('hidden_layers'):
            lidx = 0

	    for units in params['hidden_units']:
                lidx +=1
	        net = tf.layers.dense(net, 
	                          units=units, 
				  activation=tf.nn.relu,
                                  name="layer_%d"%lidx)

	# Compute logits (1 per class).
        #! should be (bs, n_classes)
	logits = tf.layers.dense(net, 
	                         params['n_classes'],
				 activation=None,
                                 name="logits")

	# Compute predictions.
	predicted_classes = tf.argmax(logits, 1, name="predicted_classes")
	'''
	if mode == tf.estimator.ModeKeys.PREDICT:
	    predictions = {
		'class_ids': predicted_classes[:, tf.newaxis],
		'probabilities': tf.nn.softmax(logits),
		'logits': logits,
	    }
	    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
	'''

	# Compute loss.
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits,
                                                      )
        loss = tf.identity(loss,name="loss")

	# Compute evaluation metrics.
        def metric_fn(labels, logits):
	    accuracy = tf.metrics.accuracy(labels=labels,
		predictions=tf.argmax(logits, axis=1),
                name="accuracy")
            return {"accuracy": accuracy}

	check_variables()

	if mode == tf.estimator.ModeKeys.EVAL:
	    return tf.contrib.tpu.TPUEstimatorSpec(
		mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))

	# Create training op.
	assert mode == tf.estimator.ModeKeys.TRAIN

        with tf.name_scope("train"):
	    optimizer = tf.train.AdagradOptimizer(learning_rate=0.05,
                                                  name="AdagradOptimizer")
	    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer,
                                                  name="CrossShardOptimizer")
	    train_op = optimizer.minimize(loss, 
	                              global_step=tf.train.get_global_step(),
                                      name="train_op")
	return tf.contrib.tpu.TPUEstimatorSpec(
	         mode=mode, loss=loss, train_op=train_op)

class CustomLSTM_Model(BaseModel):

    def __init__(self, model_dir, task):
        super(CustomLSTM_Model, self).__init__()
        print('CustomLSTM_Model init')

        self.task = task
        self.apply_TPUEstimator = task.configs['apply_TPUEstimator']
        self.use_tpu = task.configs['use_tpu']

        if self.apply_TPUEstimator == True:

	    tpu_cluster_resolver = \
              tf.contrib.cluster_resolver.TPUClusterResolver(
	        task.configs['tpu'],
	        zone=task.configs['tpu_zone'],
	        project=task.configs['gcp_project'])
	
            run_config = \
              tf.contrib.tpu.RunConfig(
	        cluster=tpu_cluster_resolver,
	        model_dir=model_dir,
	        session_config=tf.ConfigProto(
		  allow_soft_placement=True, 
                  log_device_placement=True),
	        tpu_config=\
                tf.contrib.tpu.TPUConfig(task.configs['TPUConfig_iterations'], 
                                         task.configs['TPUConfig_num_shards']))


            self.estimator = tf.contrib.tpu.TPUEstimator(
	        model_fn=self.model_fn,
	        model_dir=model_dir,
	        config=run_config,
	        params={},
                use_tpu=task.configs['use_tpu'],
	        train_batch_size=task.configs['train_batch_size'],
	        eval_batch_size=task.configs['eval_batch_size'],
	        predict_batch_size=task.configs['predict_batch_size']
	        )
	#
        else:
            self.estimator = tf.estimator.Estimator(
	                       model_fn=self.model_fn,
			       model_dir=model_dir) 

    def model_fn(self, features, labels, mode):    
   
        with tf.name_scope('input_batch'):
            inputs = tf.nn.embedding_lookup(\
                       self.task.embedding_matrix,
                       features['x'],
                       name="inputs_embd") #[bs, max_time, embd_size]

        with tf.name_scope('LSTM'):
	    # create an LSTM cell of size 100
	    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(\
	                  self.task.configs['LSTM_hidden_units'][0],
                          name="lstm_cell")
	
	    # create the complete LSTM
	    _, final_states = tf.nn.dynamic_rnn(
	                        lstm_cell, inputs, 
	                        sequence_length=features['len'], 
	                        dtype=tf.float32)

        with tf.name_scope('output_batch'):

	    # get the final hidden states of dimensionality
            # [batch_size x sentence_size]
	    outputs = tf.identity(final_states.h,
                                  name="outputs")

	    logits = tf.layers.dense(inputs=outputs,
                                     units=self.task.configs['n_classes'],
                                     name="logits")

	    # This will be None when predicting
	    if labels is not None:
	        labels = tf.reshape(labels, [-1, 1],
                                    name="labels")

        #logits = tf.Print(logits, [tf.shape(logits), logits], 'logits')
        #labels = tf.Print(labels, [tf.shape(labels), labels], 'labels')

        with tf.name_scope('metrics'):

            loss = tf.losses.sparse_softmax_cross_entropy(
                     labels=labels,
                     logits=logits)
            loss = tf.identity(loss, name="loss")

        # return EstimatorSpec or TPUEstimatorSpec

        if mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(labels, logits):
                accuracy = tf.metrics.accuracy(labels=labels,
                             predictions=tf.argmax(logits, axis=1),
                             name="accuracy")
                return {"accuracy": accuracy} 
      
            metrics = metric_fn(labels, logits)
            
            if self.apply_TPUEstimator == False:
                return tf.estimator.EstimatorSpec(
                         mode=mode,
                         loss=loss,
                         eval_metric_ops=metrics)
            else:
                return tf.contrib.tpu.TPUEstimatorSpec(
                         mode=mode,
                         loss=loss,
                         eval_metrics=(metric_fn,
                                       [labels, logits]))

        assert mode == tf.estimator.ModeKeys.TRAIN

        with tf.name_scope('train'):
	    
            optimizer = tf.train.AdamOptimizer(learning_rate=0.05,
                                               name="AdamOptimizer")

	    def _train_op_fn(loss):
	        return optimizer.minimize(
		    loss=loss,
		    global_step=tf.train.get_global_step(),
                    name="train_op")

            train_op = _train_op_fn(loss)

        check_variables()

        if self.apply_TPUEstimator == False:

            return tf.estimator.EstimatorSpec(
                     mode=mode,
                     loss=loss,
                     train_op=train_op)

        else:

            return tf.contrib.tpu.TPUEstimatorSpec(
                     mode=mode,
                     loss=loss,
                     train_op=train_op)

#################### EXTERNAL HELPER FUNCTIONS

class Clock:
    def __init__(self):
	self.last = time.time()

    def time(self):
	cur = time.time()
	elapsed = cur - self.last
	self.last = cur
	return elapsed

    def asctime(self):
	return str(time.asctime())

'''
time_stats: {description: used_time}
'''
def show_time_stats(time_stats):
    for d, t in time_stats.items():
	print('%s uses %d sec'%(d, t))

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
    elif model_name == 'CustomDNN_TPU':
        return CustomDNN_TPU_Model(model_dir, task)
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
    #embedding_path = '/home/shunfu/gnlp/'+\
    #                 'embedding/glove/glove.6B.50d.txt'
    embedding_path = None
    
    #model_name = 'DNN'
    #model_name = 'CustomDNN'
    #model_name = 'CustomDNN_TPU'
    model_name = 'CustomLSTM'
    
    configs = {}
    configs['rand_seed']=0 #for reproducibility
    np.random.seed(configs['rand_seed'])

    configs['n_cpu']=10

    configs['using_gcp']=True #affects model dir
    configs['apply_TPUEstimator']=True #to enable codes compatible with TPUEstimator
    configs['use_tpu']=True #actually use tpu at TPUEstimator

    if configs['using_gcp']==True:
        model_dir = 'gs://tpu-test-20180601/log/1/'
    else:
        model_dir = 'tmp'
        if os.path.exists(model_dir)==False:
            os.mkdir(model_dir)

    configs['tpu']='shunfu'
    configs['tpu_zone']='us-central1-b'
    configs['gcp_project']='tpu-test-204616'
    configs['TPUConfig_iterations']=50
    configs['TPUConfig_num_shards']=8
    #  TPUEstimator transforms this global batch size to a per-shard batch size, as params['batch_size'], when calling input_fn and model_fn. Cannot be None if use_tpu is True. Must be divisible by total number of replicas.
    configs['train_batch_size']=1024
    #  eval_batch_size: An int representing evaluation batch size. Must be divisible by total number of replicas.
    configs['eval_batch_size']=1024
    configs['predict_batch_size']=None

    configs['train_max_steps']=3000
    configs['eval_steps']=1 #num eval = eval_steps * batch_size

    configs['vocab_size']=4096
    configs['max_time']=256
    configs['embedding_size']=64
    configs['steps']=100
    configs['batch_size']=128

    configs['hidden_units']=[100]
    configs['LSTM_hidden_units']=[128]
    configs['n_classes']=2

    #create task and model which
    #will be applied onto the task
    clock = Clock()
    time_stats = {}

    task = create_task(task_name,
                       task_data_dir,
                       embedding_path,
                       model_name,
                       model_dir,
                       configs)

    time_stats['create_task']=clock.time()

    
    #pdb.set_trace()
    task.train()
    time_stats['task.train']=clock.time()

    task.eval()
    time_stats['task.eval']=clock.time()

    show_time_stats(time_stats)

    return

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO) 
    tf.app.run()

    
