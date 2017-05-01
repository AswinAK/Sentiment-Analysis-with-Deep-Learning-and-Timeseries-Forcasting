import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np
import pandas as pd
import random
import io
import time
import sys
import nltk

lemmatizer = WordNetLemmatizer()
positive_file = 'positive.txt'
negative_file = 'negative.txt'
MAX_TEXT = 2500


#Variables to control the number of neurons in each layer
neurons_l1 = 600
neurons_l2 = 600
neurons_l3 = 600
neurons_l4 = 600

output_categories = 2
batch_size = 1000
EPOCHS = 10

#Function to create a vector that can efficiently represent each review.
#This function goes over all the reviews, removes stop words and builds a vocabulary of words that are 
#significant in negative and positive reviews. This is a standard technique used in NLP
#We select words that occur in atleast 1% of the reviews and that are present in more numbers(double) 
#in either the positive or negative set of reviews.
def create_word_vector(pos,neg):
	stopwords = set(nltk.corpus.stopwords.words('english'))
	word_vector = set()
	positive_text_count = 0
	negative_text_count = 0
	unique_words = set()
	pos_word_count = {}	
	neg_word_count = {}

	with io.open(pos,'r', encoding='utf-8') as f:
		contents = f.readlines()
		for l in contents[:MAX_TEXT]:
			positive_text_count += 1
			words = [i.lower() for i in l.strip().split() if len(i)>=3]
			for word in set(words):
				if word not in stopwords:
					unique_words.add(word)
					if pos_word_count.has_key(word):
						pos_word_count[word] = pos_word_count[word]+1
					else:
						pos_word_count[word] = 1

	with io.open(neg,'r', encoding='utf-8') as f:
		contents = f.readlines()
		for l in contents[:MAX_TEXT]:
			negative_text_count += 1
			words = [i.lower() for i in l.strip().split() if len(i)>=3]
			for word in set(words):
				if word not in stopwords:
					unique_words.add(word)
					if neg_word_count.has_key(word):
						neg_word_count[word] = neg_word_count[word]+1
					else:
						neg_word_count[word] = 1

	final_list = []
	for w in unique_words:
		pos_count = pos_word_count.get(w,0);
		neg_count = neg_word_count.get(w,0);
		#print 'word: ',w, ' pos: ',pos_count,'  neg: ',neg_count,' total pos ',positive_text_count,' total neg ',negative_text_count
		if ((pos_count/float(positive_text_count)) >= .01 or (neg_count/float(negative_text_count)) >= .01):
			if((pos_count >= 2*neg_count) or (neg_count >= 2*pos_count)):
				final_list.append(w)
	print 'WORD VECTOR SIZE:',len(final_list)
	return final_list


#Function to create feature. Features are representations of each review which contains: the review in the word vector format 
#and the corresponding label indicating whether it is negative or possitive
def build_features(sample,word_vector,classification):
	featureset = []

	with io.open(sample,'r', encoding='utf-8') as f:
		contents = f.readlines()
		for l in contents[:MAX_TEXT]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(word_vector))
			for word in current_words:
				if word.lower() in word_vector:
					index_l = word_vector.index(word.lower())
					features[index_l] += 1
			features = list(features)
			featureset.append([features,classification])

	return featureset


#Function to create the feature sets. Each review is converted to the word vector format with the label. In TensorFlow, we 
#generally use a one hot vector to represesnt each classification. [1,0] indicates positive sentiment and [0,1] for negative
def create_feature_set(pos,neg,test_size):
	word_vector = create_word_vector(pos,neg)
	features = []
	features += build_features(pos, word_vector,[1,0])
	print 'done with + words'
	features += build_features(neg, word_vector,[0,1])
	print 'done with - words'
	random.shuffle(features)
	print 'done shuffling'
	features = np.array(features)
	print 'created numpy array'
	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])
	print 'exiting create_feature_set, train_x: ',len(train_x),' test_x: ',len(test_x)
	return train_x, train_y, test_x, test_y


#Neural Network with 4 layers
#We initialize each nuuron with their weights and biases
def neural_network_4layers(data):
	layer_1_params = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), neurons_l1])),
					  'biases': tf.Variable(tf.random_normal([neurons_l1]))}

	layer_2_params = {'weights': tf.Variable(tf.random_normal([neurons_l1, neurons_l2])),
					  'biases': tf.Variable(tf.random_normal([neurons_l2]))}

	layer_3_params = {'weights': tf.Variable(tf.random_normal([neurons_l2, neurons_l3])),
					  'biases': tf.Variable(tf.random_normal([neurons_l3]))}

	layer_4_params = {'weights': tf.Variable(tf.random_normal([neurons_l3, neurons_l4])),
	 				  'biases': tf.Variable(tf.random_normal([neurons_l4]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([neurons_l4, output_categories])),
					  'biases': tf.Variable(tf.random_normal([output_categories]))}

	l1 = tf.add(tf.matmul(data,layer_1_params['weights']), layer_1_params['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,layer_2_params['weights']), layer_2_params['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,layer_3_params['weights']), layer_3_params['biases'])
	l3 = tf.nn.relu(l3)

	l4 = tf.add(tf.matmul(l3,layer_4_params['weights']), layer_4_params['biases'])
	l4 = tf.nn.relu(l4)

	output = tf.matmul(l4,output_layer['weights']) + output_layer['biases']

	return output

#Neural network with 3 layers
#We initialize each nuuron with their weights and biases
def neural_network_3layers(data):
	layer_1_params = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), neurons_l1])),
					  'biases': tf.Variable(tf.random_normal([neurons_l1]))}

	layer_2_params = {'weights': tf.Variable(tf.random_normal([neurons_l1, neurons_l2])),
					  'biases': tf.Variable(tf.random_normal([neurons_l2]))}

	layer_3_params = {'weights': tf.Variable(tf.random_normal([neurons_l2, neurons_l3])),
					  'biases': tf.Variable(tf.random_normal([neurons_l3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([neurons_l3, output_categories])),
					  'biases': tf.Variable(tf.random_normal([output_categories]))}

	l1 = tf.add(tf.matmul(data,layer_1_params['weights']), layer_1_params['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,layer_2_params['weights']), layer_2_params['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,layer_3_params['weights']), layer_3_params['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

	return output


#Method to train the neural network
#each epoch is a cylce of feedforward and backpropogation.
#In feedforward, we pass the input vectors into the network and compare the output with the expected output.
#In the backpropogation stage, the optimiser adjusts the weights in each neuron to get the desired output
#The optimiser used here is the AdamOptimizer and to calculate the error we use the softmax cross entrophy measure
def do_deep_learning(X):
	prediction = neural_network_4layers(X)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	
	start_time = time.time()
	print 'Starting NN training...'
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(1,EPOCHS+1):
			epoch_loss = 0
			i = 0
			while i < len(train_x):
				start = i
				end = i + batch_size
				batch_x = train_x[start:end]
				batch_y = train_y[start:end]
				a, loss = sess.run([optimizer,cost],feed_dict = {x:batch_x, y:batch_y})
				epoch_loss += loss
				i += batch_size
			print 'Epoch ',epoch , 'completed, loss: ',epoch_loss
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print 'Accuracy: ',accuracy.eval({x:test_x, y:test_y})
		end_time = time.time()
		print'Took ', end_time - start_time, 'seconds'

#MAIN
if __name__ == '__main__':
	if len(sys.argv) >= 2:	
		MAX_TEXT = sys.argv[1]

	train_x, train_y, test_x, test_y  = create_feature_set(positive_file,negative_file,0.2)	
	x = tf.placeholder('float',[None,len(train_x[0])])
	y = tf.placeholder('float')	
	do_deep_learning(x)
