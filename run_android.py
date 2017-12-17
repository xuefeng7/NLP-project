import pandas as pd
import string
import numpy as np; np.random.seed(7)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import re
import time
from optparse import OptionParser
import pickle as pkl
import time
from meter import AUCMeter, MAP, MRR, precision
import logging


parser = OptionParser()
parser.add_option("--batch_size", dest="batch_size", default="25")
parser.add_option("--hidden_size", dest="hidden_size", default="120")
parser.add_option("--epoch", dest="epoch", default="20")
parser.add_option("--margin", dest="margin", default="0.3")
parser.add_option("--learning_rate", dest="learning_rate", default="5e-4")
parser.add_option("--print_every", dest="print_every", default="1")
parser.add_option("--model_option", dest="model_option", default="lstm")
parser.add_option("--restore", dest="restore", default=None)
parser.add_option("--restore_domain", dest="restore_domain", default=None)
parser.add_option("--is_android", dest="is_android", default=False)
parser.add_option("--eval_only", dest="eval_only", default=False)
parser.add_option("--lamb", dest="lamb", default="1e-6")
parser.add_option("--use_domain_classifier", dest="use_domain_classifier", default=True)
parser.add_option("--offset", dest="offset", default="1")
parser.add_option("--use_840_embedding", dest="use_840_embedding", default=False)
parser.add_option("--use_joint_embedding", dest="use_joint_embedding", default=False)

opts,args = parser.parse_args()

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
epoch = int(opts.epoch)
margin = float(opts.margin)
learning_rate = float(opts.learning_rate)
print_every = int(opts.print_every)
model_option = opts.model_option
restore = opts.restore
restore_domain = opts.restore_domain
is_android = opts.is_android
eval_only = opts.eval_only
lamb = float(opts.lamb)

use_domain_classifier = False if opts.use_domain_classifier == 'False' else True
use_840_embedding = False if opts.use_840_embedding == 'False' or not opts.use_840_embedding else True
use_joint_embedding = False if opts.use_joint_embedding == 'False' or not opts.use_joint_embedding else True


HIDDEN_DIM = hidden_size
LAMDA = lamb
offset = int(opts.offset)

cuda_available = torch.cuda.is_available()

print ('Cuda is available: {}'.format(cuda_available))


w2i_map_path = 'data/glove/w2i_map.pkl'
w2v_matrix_path = 'data/glove/w2v_matrix.pkl'
android_context_repre_path = 'data/glove/android_context_repre.pkl'
ubuntu_context_repre_path = 'data/glove/ubuntu_context_repre.pkl'

def sen2w(sen):
	processed = []
	sen = sen.strip().split()
	if len(sen) > 100:
		sen = sen[:100]
	for w in sen:
		#ignore date
		if re.match(r'\d{1,}-\d{1,}-\d{1,}', w):
			continue
		if re.match(r'\d{1,}:\d{1,}', w):
			continue
		
		if w in w2i_map:
			processed += [w]
		else:
			separated = re.findall(r"[^\W\d_]+|\d+|[=`%$\^\-@;\[&_*>\].<~|+\d+]", w)
			if len(set(separated)) == 1:
				continue
			if separated.count('*') > 3 or separated.count('=') > 3:
				continue
			for separate_w in separated:
				if separate_w in w2i_map:
					processed += [separate_w]
	return processed

def build_context_repre(path):
	context_repre = {}
	with open('data/' + path, 'r') as src:
		src = src.read().strip().split('\n')
		for line in src:
			context = line.strip().split('\t')
			qid = context.pop(0)
			if len(context) == 1:
				context_repre[int(qid)] = {'t': sen2w(context[0]), 'b': None}
			else:
				context_repre[int(qid)] = {'t':sen2w(context[0]), 'b': sen2w(context[1])}
	return context_repre


if use_840_embedding:

	print ('using 840B embedding')
	EMBEDDING_DIM = 300
	print ('loading 840b w2i ...')
	with open('data/glove/840b.w2i.pkl', 'rb') as f:
		w2i_map = pkl.load(f)
	print ('loading 840b w2v matrix ...')
	with open('data/glove/840b.npy', 'rb') as f:
		w2v_matrix = np.load(f)
	print ('done')
	print ('building context repre for ubuntu ...')
	ubuntu_context_repre = build_context_repre('ubuntu/text_tokenized.txt')
	print ('building context repre for android ...')
	android_context_repre = build_context_repre('android/corpus.tsv')

elif use_joint_embedding:

	print ('using joint embedding')
	EMBEDDING_DIM = 300
	print ('loading joint 300 w2i ...')
	with open('data/glove/joint.w2i.300.pkl', 'rb') as f:
		w2i_map = pkl.load(f)
	print ('loading joint 300 w2v matrix ...')
	with open('data/glove/joint.w2v.300.pkl', 'rb') as f:
		w2v_matrix = np.load(f)
	print ('done')
	print ('building context repre for ubuntu ...')
	ubuntu_context_repre = build_context_repre('ubuntu/text_tokenized.txt')
	print ('building context repre for android ...')
	android_context_repre = build_context_repre('android/corpus.tsv')

else:

	EMBEDDING_DIM = 200

	with open(w2i_map_path, 'rb') as f:
		w2i_map = pkl.load(f)

	with open(w2v_matrix_path, 'rb') as f:
		w2v_matrix = pkl.load(f)

	with open(ubuntu_context_repre_path, 'rb') as f:
		ubuntu_context_repre = pkl.load(f)

	with open(android_context_repre_path, 'rb') as f:
		android_context_repre = pkl.load(f)


# logging
log_filename = 'mdl' + '_margin=' + str(margin) + '_lamb=' + str(lamb) + \
	'_emb_dim=' + str(EMBEDDING_DIM) + '_use_dc=' + str(use_domain_classifier) + str(time.time()%100000)[:5]
logging.basicConfig(filename='logs/' + log_filename + '.out', filemode='w', level=logging.DEBUG)

def w2v(w):
	return w2v_matrix[w2i_map[w]]

def build_set_pair_with_idx(df):
	idx_set = {}
	for idx, row in df.iterrows():
		idx_set[row['Q']] = {'pos': np.array(list(map(int, row['Q+'].split(' ')))), \
							 'neg': np.array(list(map(int, row['Q-'].split(' '))))}
	return idx_set

def read_android_set(pos_path, neg_path):
	
	idx_set = {}
	
	pos_file = open('data/' + pos_path, 'r')
	pos_src = pos_file.read().strip().split('\n')
	
	neg_file = open('data/' + neg_path, 'r')
	neg_src = neg_file.read().strip().split('\n')
	
	for pos in pos_src:
		pos = list(map(int, pos.split(' ')))
		if pos[0] in idx_set:
			idx_set[pos[0]]['pos'] += [pos[1]]
		else:
			idx_set[pos[0]] = {}
			idx_set[pos[0]]['pos'] = [pos[1]]
			idx_set[pos[0]]['neg'] = []
		 
	for neg in neg_src:
		neg = list(map(int, neg.split(' ')))
		idx_set[neg[0]]['neg'] += [neg[1]]

	
	pos_file.close()
	neg_file.close()
	
	return idx_set

def contxt2vec(title, body=None):
	
	if body == None:
		body = []
	
	title_v = np.zeros( (len(title), EMBEDDING_DIM) )
	
	for i, t in enumerate(title):
		title_v[i] = w2v(t)
	
	if len(body) > 0:
		body_v = np.zeros( (len(body), EMBEDDING_DIM) )
		for i, b in enumerate(body):
			body_v[i] = w2v(b)
	
		return title_v, body_v
	
	return title_v, None

# create random batch
def sample_contxt_batch(context_repre, sample_size=128, batch_first=False):
	
	sampled_qids = np.random.choice(list(context_repre.keys()), sample_size)
	
	batch_title, batch_body = [], []
	max_title_len, max_body_len = 0, 0
	title_len, body_len = [], []
	
	for qid in sampled_qids:
		
		title, body = context_repre[qid]['t'], context_repre[qid]['b']
		
		title_len += [len(title)]
		batch_title += [ title ]
		max_title_len = max(max_title_len, len(title))
		
		if not body:
			body_len += [len(title)]
			batch_body += [ title ]
		else:
			batch_body += [ body ]
			body_len += [len(body)]
			max_body_len = max(max_body_len, len(body))
		
	if batch_first:
		# for CNN
		padded_batch_title = np.zeros(( len(batch_title), max_title_len, EMBEDDING_DIM)) 
		padded_batch_body = np.zeros(( len(batch_body),  max_body_len, EMBEDDING_DIM))
		for i, (title, body) in enumerate(zip(batch_title, batch_body)):
			title_repre, body_repre = contxt2vec(title, body)
			padded_batch_title[i, :title_len[i]] = title_repre
			padded_batch_body[i, :body_len[i]] = body_repre
	else:
		# for LSTM
		# (max_seq_len, batch_size, feature_len)
		padded_batch_title = np.zeros(( max_title_len, len(batch_title), EMBEDDING_DIM)) 
		padded_batch_body = np.zeros(( max_body_len, len(batch_body),  EMBEDDING_DIM))
		for i, (title, body) in enumerate(zip(batch_title, batch_body)):
			title_repre, body_repre = contxt2vec(title, body)
			padded_batch_title[:title_len[i], i] = title_repre
			padded_batch_body[:body_len[i], i] = body_repre

	return padded_batch_title, padded_batch_body, \
				np.array(title_len).reshape(-1,1), np.array(body_len).reshape(-1,1)

# create batch with order
def process_contxt_batch(qids, idx_set, context_repre, batch_first=False):
	
	batch_title, batch_body = [], []
	max_title_len, max_body_len = 0, 0
	title_len, body_len = [], []
	
	for qid in qids:
		q_title, q_body = context_repre[qid]['t'], context_repre[qid]['b']
		q_pos = idx_set[qid]['pos']
		
		if len(q_pos) > 20:
			q_pos = q_pos[:20]

		for qid_pos in q_pos:
			# query Q
			title_len += [len(q_title)]
			batch_title += [ q_title ]
			max_title_len = max(max_title_len, len(q_title))
			if not q_body:
				body_len += [len(q_title)]
				batch_body += [ q_title ]
			else:
				batch_body += [ q_body ]
				body_len += [len(q_body)]
				max_body_len = max(max_body_len, len(q_body))
				
			# pos Q
			title, body = context_repre[qid_pos]['t'], context_repre[qid_pos]['b']
			title_len += [len(title)]
			batch_title += [ title ]
			max_title_len = max(max_title_len, len(title))
			if not body:
				body_len += [len(title)]
				batch_body += [ title ]
			else:
				batch_body += [ body ]
				body_len += [len(body)]
				max_body_len = max(max_body_len, len(body))
			# neg Q
			q_neg = idx_set[qid]['neg']
			q_neg_sample_indices = np.random.choice(range(100), size=20)
			q_random_neg = q_neg[q_neg_sample_indices]
			
			for qid_neg in q_random_neg:
				title, body = context_repre[qid_neg]['t'], context_repre[qid_neg]['b']
				title_len += [len(title)]
				batch_title += [ title ]
				max_title_len = max(max_title_len, len(title))
				if not body:
					body_len += [len(title)]
					batch_body += [ title ]
				else:
					batch_body += [ body ]
					body_len += [len(body)]
					max_body_len = max(max_body_len, len(body))
	
	if batch_first:
		# for CNN
		padded_batch_title = np.zeros(( len(batch_title), max_title_len, EMBEDDING_DIM)) 
		padded_batch_body = np.zeros(( len(batch_body),  max_body_len, EMBEDDING_DIM))
		for i, (title, body) in enumerate(zip(batch_title, batch_body)):
			title_repre, body_repre = contxt2vec(title, body)
			padded_batch_title[i, :title_len[i]] = title_repre
			padded_batch_body[i, :body_len[i]] = body_repre
	else:
		# for LSTM
		# (max_seq_len, batch_size, feature_len)
		padded_batch_title = np.zeros(( max_title_len, len(batch_title), EMBEDDING_DIM)) 
		padded_batch_body = np.zeros(( max_body_len, len(batch_body),  EMBEDDING_DIM))
		for i, (title, body) in enumerate(zip(batch_title, batch_body)):
			title_repre, body_repre = contxt2vec(title, body)
			padded_batch_title[:title_len[i], i] = title_repre
			padded_batch_body[:body_len[i], i] = body_repre

	return padded_batch_title, padded_batch_body, \
				np.array(title_len).reshape(-1,1), np.array(body_len).reshape(-1,1)

def cos_sim(qv, qv_):
	return torch.sum(qv * qv_, dim=1) / (torch.sqrt(torch.sum(qv ** 2, dim=1)) * torch.sqrt(torch.sum(qv_ ** 2, dim=1)))

def evaluate(embeddings): # (n x 240)
	qs = embeddings[0]
	qs_ = embeddings[1:]
	cos_scores = cos_sim(qs.expand(len(embeddings)-1, qs.size(0)), qs_)
	return cos_scores

def process_eval_batch(qid, data, batch_first=False):
	qid_dict = data[qid]
	qs = qid_dict['q']
	max_title_len, max_body_len = 0, 0
	title_len, body_len = [], []
	batch_title, batch_body = [], []
	for qid_ in [qid] + qs:
		title, body = android_context_repre[qid_]['t'], android_context_repre[qid_]['b']
		title_len += [len(title)]
		batch_title += [ title ]
		max_title_len = max(max_title_len, len(title))
		if not body:
			body_len += [len(title)]
			batch_body += [ title ]
		else:
			batch_body += [ body ]
			body_len += [len(body)]
			max_body_len = max(max_body_len, len(body))
			
	if batch_first:
		padded_batch_title = np.zeros(( len(batch_title), max_title_len, EMBEDDING_DIM)) 
		padded_batch_body = np.zeros(( len(batch_body),  max_body_len, EMBEDDING_DIM))
		for i, (title, body) in enumerate(zip(batch_title, batch_body)):
			title_repre, body_repre = contxt2vec(title, body)
			padded_batch_title[i, :title_len[i]] = title_repre
			padded_batch_body[i, :body_len[i]] = body_repre
	else:
		padded_batch_title = np.zeros(( max_title_len, len(batch_title), EMBEDDING_DIM)) 
		padded_batch_body = np.zeros(( max_body_len, len(batch_body),  EMBEDDING_DIM))
		for i, (title, body) in enumerate(zip(batch_title, batch_body)):
			title_repre, body_repre = contxt2vec(title, body)
			padded_batch_title[:title_len[i], i] = title_repre
			padded_batch_body[:body_len[i], i] = body_repre
	
	return padded_batch_title, padded_batch_body, \
		   np.array(title_len).reshape(-1,1), np.array(body_len).reshape(-1,1) 

# Model

class GradReverse(torch.autograd.Function):
	def forward(self, x):
		return x.view_as(x)

	def backward(self, grad_output):
		return (grad_output * (-LAMDA)) # need tune

def grad_reverse(x):
	return GradReverse()(x)

class DomainClassifer(nn.Module):
	
	def __init__(self, input_size, hidden_size, num_classes):
		
		super(DomainClassifer, self).__init__()
		
		self.domain_classifier = nn.Sequential(
		  nn.Linear(input_size, 300),
		  nn.BatchNorm1d(300),
		  nn.ReLU(),
		  nn.Linear(300, hidden_size),
		  nn.ReLU(),
		  nn.Linear(hidden_size, num_classes),
		  nn.LogSoftmax()
		)


	def forward(self, embedding):

		embedding = grad_reverse(embedding)
		return self.domain_classifier(embedding)


class EmbeddingLayer(nn.Module):
	
	def __init__(self, input_size, hidden_size, layer_type, num_layer=1, kernel_size=3):
		
		super(EmbeddingLayer, self).__init__()
		
		self.num_layer = num_layer
		
		self.layer_type = layer_type
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.kernel_size = kernel_size
		
		self.tanh = nn.Tanh()
		
		if self.layer_type == 'lstm':
			
			self.embedding_layer = nn.LSTM(self.input_size, hidden_size, bidirectional=True)
		
		elif self.layer_type == 'cnn':

			self.embedding_layer = nn.Sequential(
						nn.Conv1d(in_channels = self.input_size,
								  out_channels = self.hidden_size,
								  kernel_size = self.kernel_size),
						nn.Dropout(p=0.2),
						self.tanh)

	def init_hidden(self, batch_size):
		hidden = Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size))
		memory = Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size))
		if cuda_available:
			hidden = hidden.cuda()
			memory = memory.cuda()
		return (hidden, memory)
		# return (Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)), \
		#       Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)))

	def forward(self, title, body, title_len, body_len):
		
			
		if self.layer_type == 'lstm':
			
			title_mask = Variable(torch.FloatTensor(build_mask3d(title_len, np.max(title_len))))
			body_mask = Variable(torch.FloatTensor(build_mask3d(body_len, np.max(body_len))))
			
			
			title_out, self.title_hidden = self.embedding_layer(title, (self.tanh(self.title_hidden[0]), \
																   self.tanh(self.title_hidden[1])))
			body_out, self.body_hidden = self.embedding_layer(body, (self.tanh(self.body_hidden[0]), \
																   self.tanh(self.body_hidden[1])))
		
		if self.layer_type == 'cnn':
			# batch first input
			title_mask = Variable(torch.FloatTensor(build_mask3d(title_len - self.kernel_size + 1,\
																 np.max(title_len) - self.kernel_size + 1)))
			body_mask = Variable(torch.FloatTensor(build_mask3d(body_len - self.kernel_size + 1, \
																np.max(body_len) - self.kernel_size + 1)))
			
			title = torch.transpose(title, 1, 2)
			body = torch.transpose(body, 1, 2)

			title_out =  self.embedding_layer(title)
			body_out =  self.embedding_layer(body)

			title_out = torch.transpose(title_out, 1, 2)
			body_out = torch.transpose(body_out, 1, 2)
		
			title_out = torch.transpose(title_out, 0, 1)
			body_out = torch.transpose(body_out, 0, 1)

		if cuda_available:
			title_mask = title_mask.cuda()
			body_mask = body_mask.cuda()

		title_embeddings = torch.sum(title_out * title_mask, dim=0) / torch.sum(title_mask, dim=0)
		body_embeddings = torch.sum(body_out * body_mask, dim=0) / torch.sum(body_mask, dim=0)
		embeddings = ( title_embeddings + body_embeddings ) / 2

		if cuda_available:
			embeddings = embeddings.cuda()
		return embeddings

def build_mask3d(seq_len, max_len):
	mask = np.zeros((max_len, len(seq_len), 1))
	for i, s in enumerate(seq_len):
		# only one word
		if int(s) <= -1:
			mask[0, i] = 1
		# only two word
		elif int(s) == 0:
			mask[:2, i] = np.ones((2, 1))
		elif int(s) > 0: 
			mask[:int(s), i] = np.ones((int(s), 1))
	return mask

def multi_margin_loss(hidden, margin=0.50):
	
	def loss_func(embeddings):
		# a batch of embeddings
		blocked_embeddings = embeddings.view(-1, 22, hidden)
		q_vecs = blocked_embeddings[:,0,:]
		
		pos_vecs = blocked_embeddings[:,1,:]
		neg_vecs = blocked_embeddings[:,2:,:]

		pos_scores = torch.sum(q_vecs * pos_vecs, dim=1) / (torch.sqrt(torch.sum(q_vecs ** 2, dim=1)) \
												   * torch.sqrt(torch.sum(pos_vecs ** 2, dim=1)))
		neg_scores = torch.sum(torch.unsqueeze(q_vecs, dim=1) * neg_vecs, dim=2) \
		/ (torch.unsqueeze(torch.sqrt(torch.sum(q_vecs ** 2, dim=1)),dim=1) * torch.sqrt(torch.sum( neg_vecs ** 2, dim=2)))
		neg_scores = torch.max(neg_scores, dim=1)[0]
		
		diff = neg_scores - pos_scores + margin
		loss = torch.mean((diff > 0).float() * diff)
		return loss

	return loss_func

def accuracy(output, target):
	output = torch.max(output, 1)[1]
	# torch.squeeze(output > 0.5)#torch.max(output, 1)[1]
	# return torch.sum(output.float().data == target.data) / target.data.shape[0]
	return torch.sum(output.data == target.data) / output.data.shape[0]

def read_annotations_android(pos_path, neg_path, max_neg=20):
	dic = {}
	with open('data/android/' + pos_path) as src:
		src = src.read().strip().split('\n')
		for line in src:
			indices = line.strip().split()
			idx1, idx2 = int(indices[0]), int(indices[1])
			if idx1 not in dic:
				dic[idx1] = {}
				dic[idx1]['q'] = []
				dic[idx1]['label'] = []
			dic[idx1]['q'].append(idx2)
			dic[idx1]['label'].append(1)
	with open('data/android/' + neg_path) as src:
		src = src.read().strip().split('\n')
		for line in src:
			indices = line.strip().split()
			idx1, idx2 = int(indices[0]), int(indices[1])
			if idx1 not in dic:
				dic[idx1] = {}
				dic[idx1]['q'] = []
				dic[idx1]['label'] = []
			if len(dic[idx1]['label']) <= max_neg+1:
				dic[idx1]['q'].append(idx2)
				dic[idx1]['label'].append(0)
	return dic

# DEV SET
dev_android = read_annotations_android('dev.pos.txt', 'dev.neg.txt')

# TEST SET
test_android = read_annotations_android('test.pos.txt', 'test.neg.txt')

def eval_metrics(labels, eval_name):

	map_stdout = eval_name + ' Performance MAP ' +  str(MAP(labels))
	mrr_stdout = eval_name + ' Performance MRR ' +  str(MRR(labels))
	p1_stdout = eval_name + ' Performance P@1 ' + str(precision(1, labels))
	p5_stdout = eval_name + ' Performance P@5 ' + str(precision(5, labels))

	print (map_stdout + '\n' + mrr_stdout + '\n' + p1_stdout + '\n' + p5_stdout)

	logging.debug(map_stdout)
	logging.debug(mrr_stdout)
	logging.debug(p1_stdout)
	logging.debug(p5_stdout)

def do_eval(embedding_layer, eval_name, batch_first=False):
	
	if eval_name == 'Dev':
		eval_data = dev_android   
			
	elif eval_name == 'Test':
		eval_data = test_android
	
	eval_map = {}
	for qid_ in eval_data.keys():
		eval_map[qid_] = process_eval_batch(qid_, eval_data, batch_first=batch_first)

	labels = []
	auc = AUCMeter()

	for qid_ in eval_map.keys():
		eval_title_batch, eval_body_batch, eval_title_len, eval_body_len = eval_map[qid_] # process_eval_batch(qid_, eval_data)
		embedding_layer.title_hidden = embedding_layer.init_hidden(eval_title_batch.shape[1])
		embedding_layer.body_hidden = embedding_layer.init_hidden(eval_body_batch.shape[1])
		eval_title_qs = Variable(torch.FloatTensor(eval_title_batch))
		eval_body_qs = Variable(torch.FloatTensor(eval_body_batch))

		if cuda_available:
			eval_title_qs, eval_body_qs = eval_title_qs.cuda(), eval_body_qs.cuda()
		embeddings = embedding_layer(eval_title_qs, eval_body_qs, eval_title_len, eval_body_len)
		cos_scores = evaluate(embeddings).cpu().data.numpy()
		true_labels = np.array(eval_data[qid_]['label'])
		auc.add(cos_scores, true_labels)
		labels.append(true_labels[np.argsort(cos_scores)][::-1])

	
	auc_stdout = eval_name + ' AUC ' + str(auc.value(0.05))
	print(auc_stdout)
	logging.debug(auc_stdout)
	eval_metrics(labels, eval_name)
	return auc.value(0.05)

def create_target_batch(src, tareget, sample_size=128, batch_first=False):

	sampled_src_qids = np.random.choice(list(src.keys()), sample_size)
	sampled_target_qids = np.random.choice(list(tareget.keys()), sample_size)
	
	batch_title, batch_body = [], []
	max_title_len, max_body_len = 0, 0
	title_len, body_len = [], []

	labels = []
	
	for qid in sampled_src_qids:
		
		title, body = src[qid]['t'], src[qid]['b']
		
		title_len += [len(title)]
		batch_title += [ title ]
		max_title_len = max(max_title_len, len(title))
		
		if not body:
			body_len += [len(title)]
			batch_body += [ title ]
		else:
			batch_body += [ body ]
			body_len += [len(body)]
			max_body_len = max(max_body_len, len(body))

		labels += [0]

	for qid in sampled_target_qids:

		title, body = tareget[qid]['t'], tareget[qid]['b']
		
		title_len += [len(title)]
		batch_title += [ title ]
		max_title_len = max(max_title_len, len(title))
		
		if not body:
			body_len += [len(title)]
			batch_body += [ title ]
		else:
			batch_body += [ body ]
			body_len += [len(body)]
			max_body_len = max(max_body_len, len(body))

		labels += [1]

		
	if batch_first:
		# for CNN
		padded_batch_title = np.zeros(( len(batch_title), max_title_len, EMBEDDING_DIM)) 
		padded_batch_body = np.zeros(( len(batch_body),  max_body_len, EMBEDDING_DIM))
		for i, (title, body) in enumerate(zip(batch_title, batch_body)):
			title_repre, body_repre = contxt2vec(title, body)
			padded_batch_title[i, :title_len[i]] = title_repre
			padded_batch_body[i, :body_len[i]] = body_repre
	else:
		# for LSTM
		# (max_seq_len, batch_size, feature_len)
		padded_batch_title = np.zeros(( max_title_len, len(batch_title), EMBEDDING_DIM)) 
		padded_batch_body = np.zeros(( max_body_len, len(batch_body),  EMBEDDING_DIM))
		for i, (title, body) in enumerate(zip(batch_title, batch_body)):
			title_repre, body_repre = contxt2vec(title, body)
			padded_batch_title[:title_len[i], i] = title_repre
			padded_batch_body[:body_len[i], i] = body_repre

	return padded_batch_title, padded_batch_body, \
				np.array(title_len).reshape(-1,1), np.array(body_len).reshape(-1,1), np.array(labels)


def train( 
	embedding_layer, domain_classifier, 
	emb_batch_size=25, dc_batch_size=100,
	num_epoch=100, lamda=1e-3,
	id_set=None,train_from=None,sample_from=None,
	eval=True,
	margin=0.5
	):
	
	if embedding_layer.layer_type == 'lstm':
		
		margin_criterion = multi_margin_loss(hidden=embedding_layer.hidden_size * 2, margin=margin)
	
	elif embedding_layer.layer_type == 'cnn':
		
		margin_criterion = multi_margin_loss(hidden=embedding_layer.hidden_size, margin=margin)
	
	emb_optimizer = torch.optim.Adam(embedding_layer.parameters(), lr=0.001)
	
	if domain_classifier is not None:
		domain_criterion = torch.nn.NLLLoss()
		params = list(embedding_layer.parameters()) + list(domain_classifier.parameters())
		domain_optimizer = torch.optim.Adam(params, lr=0.0001)
	
	qids = list(id_set.keys())
	num_batch = len(qids) // emb_batch_size
	
	last_loss = np.inf
	for epoch in range(offset, num_epoch + offset):
		cumulative_loss = 0
		for batch_idx in range(1, num_batch + 1):
			
			batch_x_qids = qids[ ( batch_idx - 1 ) * emb_batch_size: batch_idx * emb_batch_size ]
			
			## Minimize margin loss
			if embedding_layer.layer_type == 'lstm':
				batch_title, batch_body, title_len, body_len = process_contxt_batch(batch_x_qids, \
																				id_set, train_from)
				embedding_layer.title_hidden = embedding_layer.init_hidden(batch_title.shape[1])
				embedding_layer.body_hidden = embedding_layer.init_hidden(batch_body.shape[1])
			else:
				batch_title, batch_body, title_len, body_len = process_contxt_batch(batch_x_qids, \
																				id_set, train_from, batch_first=True)
			
			title_qs = Variable(torch.FloatTensor(batch_title))
			body_qs = Variable(torch.FloatTensor(batch_body))
			if cuda_available:
				title_qs = title_qs.cuda()
				body_qs = body_qs.cuda()
			embeddings = embedding_layer(title_qs, body_qs, title_len, body_len) # class label = ubuntu
			
			margin_loss = margin_criterion(embeddings)
			if cuda_available:
				margin_loss = margin_loss.cuda()

			emb_optimizer.zero_grad()
			margin_loss.backward()
			emb_optimizer.step()

			if domain_classifier is None:
				loss = margin_loss
			else:   
				## Domain classification
				# sample title, body from two domains
				if embedding_layer.layer_type == 'lstm':

					## mixed
					batch_title, batch_body, title_len, body_len, labels = create_target_batch(train_from, sample_from, \
							sample_size=dc_batch_size)
		
				else:

					batch_title, batch_body, title_len, body_len, labels = create_target_batch(train_from, sample_from, \
							sample_size=dc_batch_size, batch_first=True)


				# mix
				title_qs = Variable(torch.FloatTensor(batch_title))
				body_qs = Variable(torch.FloatTensor(batch_body))

				if cuda_available:
					title_qs, body_qs = title_qs.cuda(), body_qs.cuda()

				if embedding_layer.layer_type == 'lstm':
					embedding_layer.title_hidden = embedding_layer.init_hidden(batch_title.shape[1])
					embedding_layer.body_hidden = embedding_layer.init_hidden(batch_body.shape[1])

				embedding_X = embedding_layer(title_qs, body_qs, title_len, body_len) 
				embedding_Y = Variable(torch.LongTensor(labels))

				if cuda_available:
					embedding_X, embedding_Y = embedding_X.cuda(), embedding_Y.cuda()
				
				predicted = domain_classifier(embedding_X)
				domain_loss = domain_criterion(predicted, embedding_Y)

			cumulative_loss += loss.cpu().data.numpy()[0]
			if batch_idx % 20 == 1:

				if domain_classifier is None:
					loss_stdout  = 'epoch:{}/{}, batch:{}/{}, loss:{}'.format(epoch, num_epoch+offset-1, \
															  batch_idx, num_batch, loss.cpu().data[0])
				else:

					loss_stdout = 'epoch:{}/{}, batch:{}/{}, margin_loss:{}, domain_loss:{}'.format(epoch, num_epoch+offset-1, \
															  batch_idx, num_batch, margin_loss.cpu().data[0], \
															  domain_loss.cpu().data[0])
					domain_acc = accuracy(predicted, embedding_Y)
					loss_stdout += (', domain_acc:' + str(domain_acc))
					
				
				print (loss_stdout)
				logging.debug(loss_stdout)


			if domain_classifier is not None:
				if cuda_available:
					domain_loss = domain_loss.cuda()
				
				domain_optimizer.zero_grad()
				domain_loss.backward()
				domain_optimizer.step()


			if batch_idx % print_every == 0: 
				print ('evaluating ....')
				if embedding_layer.layer_type == 'lstm':
					do_eval(embedding_layer, 'Dev')
					do_eval(embedding_layer, 'Test')
					# print ('------------------')
					# do_eval(embedding_layer, 'Test')
				elif embedding_layer.layer_type == 'cnn':
					do_eval(embedding_layer, 'Dev', batch_first=True)
					do_eval(embedding_layer, 'Test', batch_first=True)
					# print ('------------------')
					# do_eval(embedding_layer, 'Test', batch_first=True)

		print ('Epoch {} is finished'.format(epoch))
		
		print ('evaluating ....')
		if embedding_layer.layer_type == 'lstm':
			dev_auc = do_eval(embedding_layer, 'Dev')
			test_auc = do_eval(embedding_layer, 'Test')
		elif embedding_layer.layer_type == 'cnn':
			dev_auc = do_eval(embedding_layer, 'Dev', batch_first=True)
			test_auc = do_eval(embedding_layer, 'Test', batch_first=True)

		print ('Saving Model...')
		save_model(embedding_layer, 'models/'+'DC1_' + str(use_domain_classifier) + '_' + str(model_option)+'_bi_epoch=' + str(epoch)+\
				'_margin='+str(margin)+'_hidden='+str(HIDDEN_DIM)+'_devauc=' + str(dev_auc)+'_testauc' + str(test_auc))
		print ('Cumulative loss', cumulative_loss)
		if domain_classifier is not None:
			save_model(domain_classifier, 'models/'+'DC2_' + str(model_option)+'_bi_epoch='+str(epoch)+ \
				'_margin='+str(margin)+'_hidden='+str(HIDDEN_DIM)+ '_devauc=' + str(dev_auc)+'_testauc' + str(test_auc))

		if cumulative_loss < last_loss:
			last_loss = cumulative_loss
		else:
			print('Last loss', last_loss, '/ Current loss', cumulative_loss)
			break

ubuntu_train_df = pd.read_csv('data/train_random.txt', header=None, delimiter='\t', names=['Q','Q+','Q-'])
ubuntu_train_idx_set = build_set_pair_with_idx(ubuntu_train_df)

android_dev_idx_set = read_android_set('android/dev.pos.txt', 'android/dev.neg.txt')
android_test_idx_set = read_android_set('android/test.pos.txt', 'android/test.neg.txt')

def save_model(mdl, path):
	# saving model params
	torch.save(mdl.state_dict(), path)

def restore_model(mdl_skeleton, path):
	# restoring params to the mdl skeleton
	mdl_skeleton.load_state_dict(torch.load('models/' + path))

if __name__ == '__main__':
	print('batch_size={}, epoch={}, margin={}, model={}, hidden_size={}'.format( \
		batch_size, epoch, margin, model_option, HIDDEN_DIM))
	model = EmbeddingLayer(EMBEDDING_DIM, HIDDEN_DIM, model_option)

	if restore is not None:
		restore_model(model, restore)

	if cuda_available:
		model = model.cuda()
	
	if eval_only:
		if model_option == 'lstm':
			do_eval(model, 'Dev')
			print ('------------------')
			do_eval(model, 'Test')
		elif model_option == 'cnn':
			do_eval(model, 'Dev', batch_first=True)
			print ('------------------')
			do_eval(model, 'Test', batch_first=True)

	else:
		print('Start Training...')
		if use_domain_classifier:
			if model_option == 'lstm':
				domain_classifier = DomainClassifer(HIDDEN_DIM * 2, hidden_size=128, num_classes=2)
			else:
				domain_classifier = DomainClassifer(HIDDEN_DIM, hidden_size=128, num_classes=2)
			if restore_domain:
				restore_model(domain_classifier, restore_domain)
			if cuda_available:
				domain_classifier = domain_classifier.cuda()
		else:
			domain_classifier = None
		
		train( 
			model, 
			domain_classifier, 
			lamda = LAMDA,
			id_set=ubuntu_train_idx_set,
			train_from=ubuntu_context_repre,
			sample_from=android_context_repre,
			margin=margin,
			emb_batch_size=batch_size,
			num_epoch=epoch
		)
		print ('Saving Model...')
		save_mdl_path = 'models/'+'DC1_final' + str(use_domain_classifier) + '_' + str(model_option)+'_bi_epoch='+str(epoch)+'_margin='+str(margin)+'_hidden='+\
			str(HIDDEN_DIM)+str(time.time()%100000)[:5]
		save_model(model, save_mdl_path)
		logging.debug('dc mdl is saved to:' + save_mdl_path)
		if use_domain_classifier:
			save_mdl_path =  'models/'+'DC2_final' + str(model_option)+'_bi_epoch='+str(epoch)+'_margin='+str(margin)+'_hidden='+\
				str(HIDDEN_DIM)+str(time.time()%100000)[:5]
			save_model(domain_classifier, save_mdl_path)
			logging.debug('dc mdl is saved to:' + save_mdl_path)
		print ('Done')
