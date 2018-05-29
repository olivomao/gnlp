'''
a draft py to do text classification via MLP
'''

import io
import tensorflow as tf
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence

import os,re,pdb,collections
import tempfile
import numpy as np
import pandas as pd

import logging

'''
code from: https://medium.com/tensorflow/+\
           classifying-text-with-tensorflow-estimators-a99603033fbe
'''
def preprocess_data_IMDB(data_dir=None):
    vocab_size = 5000
    sentence_size = 200
    embedding_size = 50
    model_dir = tempfile.mkdtemp()
    print('temp model_dir is %s'%model_dir)

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
    return IMDB_PreProcessedData(x_train=x_train,
                                 y_train=y_train,
                                 x_len_train=x_len_train,
                                 x_test=x_test,
                                 y_test=y_test,
                                 x_len_test=x_len_test,
                                 vocab_size=vocab_size,
                                 embedding_size=embedding_size,
                                 word_index = word_index
                                 )
                                   

'''
preprocess the raw data files specified to RottenTomatoes

return train and development/validation df(pandas data frame);
       df contains "x" of cleaned sentences and "y" of labels (0/1)

note:
we split data into 90% for train and 10% for dev
random_state=0 is for reproducibility
'''
def preprocess_data_RottenTomatoes(data_dir):

    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from 
        https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()

    '''
    clean a src file into dst file, via clean_str on per line
    '''
    def clean_file(src_file, dst_file):
        with open(src_file, 'r') as src,\
             open(dst_file, 'w') as dst:

	    for line in src:
	        clean_line = clean_str(line.strip())
		dst.write(clean_line+'\n')

	    print('%s cleaned to %s'%(src_file, dst_file))

    '''
    fp (file path) with lab (0 or 1) into df (pandas data frame)
    '''
    def file2df(fp,lab):
        data={}
        data["x"]=[]
	with open(fp, "r") as f:
            for line in f:
	        data["x"].append(line.strip())
	    df = pd.DataFrame.from_dict(data)
	    df["y"]=lab
        return df

    #[file_name, label, clean_file_name]
    items = [["rt-polarity.pos", 1, "rt-polarity-clean.pos"],
             ["rt-polarity.neg", 0, "rt-polarity-clean.neg"]]

    dfs = []
    for item in items:
        raw_data_path = os.path.join(data_dir, item[0])
        clean_data_path = os.path.join(data_dir, item[2])
	if os.path.exists(clean_data_path)==False: 
            clean_file(raw_data_path, clean_data_path)
        dfs.append(file2df(clean_data_path, item[1]))

    df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
    
    #prepare train and dev data frames
    df_train=df.sample(frac=0.9,random_state=0)
    df_test=df.drop(df_train.index)
    return df_train, df_test


class IMDB_PreProcessedData(
        collections.namedtuple("IMDB_PreProcessedData",
                               ("x_train",
                                "y_train",
                                "x_len_train",
                                "x_test",
                                "y_test",
                                "x_len_test",
                                "vocab_size",
                                "embedding_size",
                                "word_index"
                               ))):
    pass

'''
task: will specify the files in data_dir
data_dir: store data files related to task
embd_dir: store embedding files, independent of task
'''
class Data(object):

    def __init__(self,
                 task,
                 data_dir,
                 embd_path=None,
                 model_dir=None):
        
        print('Data object created')

        self.task = task
        
        if model_dir is None:
            self.model_dir = tempfile.mkdtemp()
        else:
            self.model_dir = model_dir 

        self.PreProcessedData = \
            self.preprocess_data(task,
                                 data_dir)
        print('preprocess_data done')

        self.embedding_matrix = \
            self.load_embedding(embd_path)

        self.classifier = self.get_classifier()
        check_variables()

        pass

    def load_embedding(self, embd_path):
        vocab_size = self.PreProcessedData.vocab_size
        embedding_size = self.PreProcessedData.embedding_size
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
                size=(self.PreProcessedData.vocab_size,
                      self.PreProcessedData.embedding_size))
	num_loaded = 0
	for w, i in self.PreProcessedData.word_index.items():
	    v = embeddings.get(w)
	    if v is not None and i < vocab_size:
	        embedding_matrix[i] = v
		num_loaded += 1
	print('Successfully loaded pretrained embeddings')
	embedding_matrix = embedding_matrix.astype(np.float32)
	return embedding_matrix

    def run(self):

        logging.getLogger().setLevel(logging.INFO) 
        print(self.classifier.train(input_fn=self.train_input_fn,
                              steps=2500))
        #pdb.set_trace()
        print(self.classifier.evaluate(input_fn=self.eval_input_fn))
        pdb.set_trace()
        return

    '''
    process task specific raw data into
    pandas data frames for train (self.df_train)
    and development/validation (self.df_dev)

    df contains "x" of cleaned sentence and "y" of labels (e.g. 0/1)
    '''
    def preprocess_data(self,
                        task,
                        data_dir):

        if task=='MovieReview_RottenTomatoes':
            return preprocess_data_RottenTomatoes(data_dir)
        elif task=='IMDB':
            return preprocess_data_IMDB(data_dir)
        else:
            return None

    def train_input_fn(self):
        
        if self.task=='MovieReview_RottenTomatoes':
            return self.train_input_fn_RottenTomatoes()
        elif self.task=='IMDB':
            return self.train_input_fn_IMDB()
        else:
            return None

    def train_input_fn_RottenTomatoes(self):

        return

    def train_input_fn_IMDB(self):

        def parser(x, length, y):
            features = {"x": x, "len": length}
            return features, y

        x_train = self.PreProcessedData.x_train
        y_train = self.PreProcessedData.y_train
        x_len_train = self.PreProcessedData.x_len_train

        dataset = tf.data.Dataset.from_tensor_slices(\
                  (x_train, x_len_train, y_train))
        dataset = dataset.shuffle(buffer_size=len(x_train))
        dataset = dataset.batch(100)
        dataset = dataset.map(parser)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def eval_input_fn(self):

        if self.task=='MovieReview_RottenTomatoes':
            return self.eval_input_fn_RottenTomatoes()
        elif self.task=='IMDB':
            return self.eval_input_fn_IMDB()
        else:
            return None

    def eval_input_fn_RottenTomatoes(self):
        return None

    def eval_input_fn_IMDB(self):

        def parser(x, length, y):
            features = {"x": x, "len": length}
            return features, y

        x_test = self.PreProcessedData.x_test
        y_test = self.PreProcessedData.y_test
        x_len_test = self.PreProcessedData.x_len_test

        dataset = tf.data.Dataset.from_tensor_slices(\
                  (x_test, x_len_test, y_test))
        dataset = dataset.batch(100)
        dataset = dataset.map(parser)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def get_feature_columns(self):
    
        if self.task=='MovieReview_RottenTomatoes':
            return self.get_feature_columns_RottenTomatoes()
        elif self.task=='IMDB':
            return self.get_feature_columns_IMDB()
        else:
            return None

    def get_feature_columns_IMDB(self):

        column = tf.feature_column.\
                 categorical_column_with_identity(\
                 'x', self.PreProcessedData.vocab_size)
        column = tf.feature_column.embedding_column(column, 
                                                    dimension=\
                                                    self.PreProcessedData.embedding_size)
        return [column]


    def get_classifier(self):

        if self.task=='MovieReview_RottenTomatoes':
            return self.get_classifier_RottenTomatoes()
        elif self.task=='IMDB':
            return self.get_classifier_IMDB()
        else:
            return None

    def get_classifier_RottenTomatoes(self):

        return None

    def get_classifier_IMDB(self):

        #'''
        classifier = tf.estimator.LinearClassifier(\
                     feature_columns=self.get_feature_columns(),
                     model_dir=self.model_dir)
        #'''
        '''
        classifier = tf.estimator.DNNClassifier(
                        hidden_units=[100],
                        feature_columns=self.get_feature_columns(), 
                        model_dir=self.model_dir)
        '''
        classifier = self.get_classifier_IMDB_LSTM()


        return classifier

    def get_classifier_IMDB_LSTM(self):

	'''
        The head already knows how to compute predictions, loss, 
        train_op, metrics and export outputs, and can be reused across models. 
        '''
        head = tf.contrib.estimator.binary_classification_head()
        vocab_size = self.PreProcessedData.vocab_size
        embedding_size = self.PreProcessedData.embedding_size

        def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
            assert dtype is tf.float32
            return self.embedding_matrix

        if self.embedding_matrix is None:
            params = {}
        else:
            params = {'embedding_initializer': my_initializer}

	def lstm_model_fn(features, labels, mode, params):    
	    # [batch_size x sentence_size x embedding_size]
            #! feature columns are not used here
            #! we can do embedding mapping here
            if 'embedding_initializer' in params:
                embd_initializer = params['embedding_initializer']

            else:
                embd_initializer = tf.random_uniform_initializer(-1.0, 1.0)
	    inputs = tf.contrib.layers.embed_sequence(
		    features['x'], vocab_size, embedding_size,
		    initializer=embd_initializer)

	    # create an LSTM cell of size 100
	    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)
	    
	    # create the complete LSTM
	    _, final_states = tf.nn.dynamic_rnn(
		lstm_cell, inputs, sequence_length=features['len'], dtype=tf.float32)

	    # get the final hidden states of dimensionality [batch_size x sentence_size]
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

            #! need to understand more about head
            #  estimator_spec, and progress print 
	    return head.create_estimator_spec(
		     features=features,
		     labels=labels,
		     mode=mode,
		     logits=logits,
		     train_op_fn=_train_op_fn)


	lstm_classifier = tf.estimator.Estimator(model_fn=lstm_model_fn,
						 model_dir=self.model_dir)
        return lstm_classifier
	       
def check_variables():
    vs = tf.trainable_variables()
    print("There are %d trainable variables"%len(vs))
    for v in vs:
        print v

def main(argv=None):
    task='IMDB' #'MovieReview_RottenTomatoes'
    data_dir=None #'/home/shunfu/gnlp/data/'+'MovieReview_RottenTomatoes/'
    embd_path="/home/shunfu/gnlp/embedding/glove/glove.6B.50d.txt"

    data = \
    Data(task,
         data_dir,
         embd_path)

    pdb.set_trace()
    check_variables()

    data.run()

    pdb.set_trace()
    return    


if __name__ == '__main__':
    tf.app.run()


