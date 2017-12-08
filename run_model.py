import pandas as pd
import string
import numpy as np; np.random.seed(7)
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import re
import time
from optparse import OptionParser
import pickle as pkl

parser = OptionParser()
parser.add_option("--batch_size", dest="batch_size", default="25")
parser.add_option("--hidden_size", dest="hidden_size", default="120")
parser.add_option("--epoch", dest="epoch", default="20")
parser.add_option("--margin", dest="margin", default="0.3")
parser.add_option("--learning_rate", dest="learning_rate", default="5e-4")
parser.add_option("--print_every", dest="print_every", default="100")
parser.add_option("--model_option", dest="model_option", default="lstm")
opts,args = parser.parse_args()

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
epoch = int(opts.epoch)
margin = float(opts.margin)
learning_rate = float(opts.learning_rate)
print_every = int(opts.print_every)
model_option = opts.model_option

HIDDEN_DIM = hidden_size

cuda_available = torch.cuda.is_available()
var_torch = torch.cuda.FloatTensor(1) if cuda_available else torch.FloatTensor(1)

print ('Cuda is available: {}'.format(cuda_available))

# w2v_map = {}
# with open('data/vectors_pruned.200.txt', 'r') as src:
#     src = src.read().strip().split('\n')
#     for line in src:
#         wv = line.strip().split(' ')
#         word = wv.pop(0)
#         w2v_map[word] = np.array(list(map(float, wv)))

# w2i_map = {}
# w2v_matrix = np.zeros(( len((w2v_map.keys())), 200 ))
# for i, (key, val) in enumerate(w2v_map.items()):
#     w2i_map[key] = i
#     w2v_matrix[i] = val

with open('data/w2i_map.pkl', 'rb') as f:
    w2i_map = pkl.load(f)

with open('data/w2v_matrix.pkl', 'rb') as f:
    w2v_matrix = pkl.load(f)

with open('data/context_repre.pkl', 'rb') as f:
    context_repre = pkl.load(f)

def w2v(w):
    return w2v_matrix[w2i_map[w]]

# def sen2w(sen):
#     processed = []
#     sen = sen.strip().split()
#     if len(sen) > 100:
#         sen = sen[:100]
#     for w in sen:
#         #ignore date
#         if re.match(r'\d{1,}-\d{1,}-\d{1,}', w):
#             continue
#         if re.match(r'\d{1,}:\d{1,}', w):
#             continue
        
#         if w in w2i_map:
#             processed += [w]
#         else:
#             separated = re.findall(r"[^\W\d_]+|\d+|[=`%$\^\-@;\[&_*>\].<~|+\d+]", w)
#             if len(set(separated)) == 1:
#                 continue
#             if separated.count('*') > 3 or separated.count('=') > 3:
#                 continue
#             for separate_w in separated:
#                 if separate_w in w2i_map:
#                     processed += [separate_w]
#     return processed

# context_repre = {}
# with open('data/text_tokenized.txt', 'r', encoding='utf-8') as src:
#     # src = src.read().strip().split('\n')
#     for l in src.read().strip().split('\n'):
#         context = line.strip().split('\t')
#         qid = context.pop(0)
#         if len(context) == 1:
#             context_repre[int(qid)] = {'t': sen2w(context[0]), 'b': None}
#         else:
#             context_repre[int(qid)] = {'t':sen2w(context[0]), 'b': sen2w(context[1])}



# with open('data/w2v.pkl', 'rb') as f:
#     w2v = pkl.load(f)
# w2v_data = open('data/w2v.pkl', 'rb')
# w2v = pkl.load(w2v_data)
# w2v_data.close()
# del w2v_data

def build_set_pair_with_idx(df):
    idx_set = {}
    for idx, row in df.iterrows():
        idx_set[row['Q']] = {'pos': np.array(list(map(int, row['Q+'].split(' ')))), \
                             'neg': np.array(list(map(int, row['Q-'].split(' '))))}
    return idx_set

train_df = pd.read_csv('data/train_random.txt', header=None, delimiter='\t', names=['Q','Q+','Q-'])
train_idx_set = build_set_pair_with_idx(train_df)

def contxt2vec(title, body=None):
    
    if body == None:
        body = []
    
    title_v = np.zeros( (len(title), 200) )
    
    for i, t in enumerate(title):
        title_v[i] = w2v(t)
    
    if len(body) > 0:
        body_v = np.zeros( (len(body), 200) )
        for i, b in enumerate(body):
            body_v[i] = w2v(b)
    
        return title_v, body_v
    
    return title_v, None

def process_contxt_batch(qids, idx_set, batch_first=False):
    
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
        padded_batch_title = np.zeros(( len(batch_title), max_title_len, 200)) 
        padded_batch_body = np.zeros(( len(batch_body),  max_body_len, 200))
        for i, (title, body) in enumerate(zip(batch_title, batch_body)):
            title_repre, body_repre = contxt2vec(title, body)
            padded_batch_title[i, :title_len[i]] = title_repre
            padded_batch_body[i, :body_len[i]] = body_repre
    else:
        # for LSTM
        # (max_seq_len, batch_size, feature_len)
        padded_batch_title = np.zeros(( max_title_len, len(batch_title), 200)) 
        padded_batch_body = np.zeros(( max_body_len, len(batch_body),  200))
        for i, (title, body) in enumerate(zip(batch_title, batch_body)):
            title_repre, body_repre = contxt2vec(title, body)
            padded_batch_title[:title_len[i], i] = title_repre
            padded_batch_body[:body_len[i], i] = body_repre

    return padded_batch_title, padded_batch_body, \
                np.array(title_len).reshape(-1,1), np.array(body_len).reshape(-1,1)

def read_annotations(path, K_neg=20, prune_pos_cnt=20):
    lst = [ ]
    with open(path) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]
            pos = pos.split()
            neg = neg.split()
            if len(pos) == 0 or (len(pos) > prune_pos_cnt and prune_pos_cnt != -1): continue
            if K_neg != -1:
                np.random.shuffle(neg)
                neg = neg[:K_neg]
            s = set()
            qids = [ ]
            qlabels = [ ]
            for q in neg:
                if q not in s:
                    qids.append(q)
                    qlabels.append(0 if q not in pos else 1)
                    s.add(q)
            for q in pos:
                if q not in s:
                    qids.append(q)
                    qlabels.append(1)
                    s.add(q)
            lst.append((pid, qids, qlabels))

    return lst

def cos_sim(qv, qv_):
    return torch.sum(qv * qv_, dim=1) / (torch.sqrt(torch.sum(qv ** 2, dim=1)) * torch.sqrt(torch.sum(qv_ ** 2, dim=1)))

def process_eval_batch(qid, data, batch_first=False):
    qid_dict = data[qid]
    qs = qid_dict['q']
    max_title_len, max_body_len = 0, 0
    title_len, body_len = [], []
    batch_title, batch_body = [], []
    for qid_ in [qid] + qs:
        title, body = context_repre[qid_]['t'], context_repre[qid_]['b']
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
        padded_batch_title = np.zeros(( len(batch_title), max_title_len, 200)) 
        padded_batch_body = np.zeros(( len(batch_body),  max_body_len, 200))
        for i, (title, body) in enumerate(zip(batch_title, batch_body)):
            title_repre, body_repre = contxt2vec(title, body)
            padded_batch_title[i, :title_len[i]] = title_repre
            padded_batch_body[i, :body_len[i]] = body_repre
    else:
        padded_batch_title = np.zeros(( max_title_len, len(batch_title), 200)) 
        padded_batch_body = np.zeros(( max_body_len, len(batch_body),  200))
        for i, (title, body) in enumerate(zip(batch_title, batch_body)):
            title_repre, body_repre = contxt2vec(title, body)
            padded_batch_title[:title_len[i], i] = title_repre
            padded_batch_body[:body_len[i], i] = body_repre
    
    return padded_batch_title, padded_batch_body, \
           np.array(title_len).reshape(-1,1), np.array(body_len).reshape(-1,1) 
    
def evaluate(embeddings): # (n x 240)
    qs = embeddings[0]
    qs_ = embeddings[1:]
    cos_scores = cos_sim(qs.expand(len(embeddings)-1, qs.size(0)), qs_)
    return cos_scores

def precision(at, labels):
    res = []
    for item in labels:
        tmp = item[:at]
        if any(val==1 for val in item):
            res.append(np.sum(tmp) / len(tmp) if len(tmp) != 0 else 0.0)
    return sum(res)/len(res) if len(res) != 0 else 0.0

def MAP(labels):
    scores = []
    missing_MAP = 0
    for item in labels:
        temp = []
        count = 0.0
        for i,val in enumerate(item):
            
            if val == 1:
                count += 1.0
                temp.append(count/(i+1))
            if len(temp) > 0:
                scores.append(sum(temp) / len(temp))
            else:
                missing_MAP += 1
    return sum(scores)/len(scores) if len(scores) > 0 else 0.0
    
def MRR(labels):
    scores = []
    for item in labels:
        for i,val in enumerate(item):
            if val == 1:
                scores.append(1.0/(i+1))
                break
    return sum(scores)/len(scores) if len(scores) > 0 else 0.0
    
def do_eval(embedding_layer, eval_name, batch_first=False):
    
    if eval_name == 'Dev':
        eval_data = dev_data
        eval_map = {}
        for qid_ in dev_data.keys():
            eval_map[qid_] = process_eval_batch(qid_, dev_data, batch_first=batch_first)
            
    elif eval_name == 'Test':
        eval_data = test_data
        eval_map = {}
        for qid_ in test_data.keys():
            eval_map[qid_] = process_eval_batch(qid_, test_data, batch_first=batch_first)
    
    labels = []
    
    for qid_ in eval_map.keys():
        
        eval_title_batch, eval_body_batch, eval_title_len, eval_body_len = eval_map[qid_] # process_eval_batch(qid_, eval_data)
        embedding_layer.title_hidden = embedding_layer.init_hidden(eval_title_batch.shape[1])
        embedding_layer.body_hidden = embedding_layer.init_hidden(eval_body_batch.shape[1])
        eval_title_qs = Variable(torch.FloatTensor(eval_title_batch))
        eval_body_qs = Variable(torch.FloatTensor(eval_body_batch))

        if cuda_available:
            eval_title_qs, eval_body_qs = eval_title_qs.cuda(), eval_body_qs.cuda()
        embeddings = embedding_layer(eval_title_qs, eval_body_qs, eval_title_len, eval_body_len)
        cos_scores = evaluate(embeddings)
        if cuda_available:
            cos_scores = cos_scores.cpu()
        labels.append(np.array(eval_data[qid_]['label'])[np.argsort(cos_scores.data.numpy())][::-1])
    
    print (eval_name + ' Performance MAP', MAP(labels))
    print (eval_name + ' Performance MRR', MRR(labels))
    print (eval_name + ' Performance P@1', precision(1, labels))
    print (eval_name + ' Performance P@5', precision(5, labels))


# DEV SET
dev = read_annotations('data/dev.txt')
dev_data = {}
for item in dev:
    qid = int(item[0])
    dev_data[qid] = {}
    dev_data[qid]['q'] = list(map(int, item[1]))
    dev_data[qid]['label'] = item[2]

# TEST SET
test = read_annotations('data/test.txt')
test_data = {}
for item in test:
    qid = int(item[0])
    test_data[qid] = {}
    test_data[qid]['q'] = list(map(int, item[1]))
    test_data[qid]['label'] = item[2]

dev_map = {}
for qid_ in dev_data.keys():
    dev_map[qid_] = process_eval_batch(qid_, dev_data)

test_map = {}
for qid_ in test_data.keys():
    test_map[qid_] = process_eval_batch(qid_, test_data)

def build_mask(seq_len):
    mask = []
    for i, s in enumerate(seq_len):
        s_mask = np.zeros((np.max(seq_len), 1))
        s_mask[:int(s)] = np.ones((int(s), 1))
        mask += [s_mask]
    return mask

def build_mask3d(seq_len, max_len):
    mask = np.zeros((max_len, len(seq_len), 1))
    for i, s in enumerate(seq_len):
        # only one word
        if int(s) == -1:
            mask[0, i] = 1
        # only two word
        elif int(s) == 0:
            mask[:2, i] = np.ones((2, 1))
        else: 
            mask[:int(s), i] = np.ones((int(s), 1))
    return mask

def multi_margin_loss(hidden, margin=0.30):
    
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

class EmbeddingLayer(nn.Module):
    
    def __init__(self, input_size, hidden_size, layer_type, num_layer=1, kernel_size=3):
        
        super(EmbeddingLayer, self).__init__()

        self.num_layer = num_layer
        
        self.layer_type = layer_type
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        
        self.tanh = nn.Tanh()
        
        if layer_type == 'lstm':
            
            self.layer_type = 'lstm'
            self.embedding_layer = nn.LSTM(input_size, hidden_size, bidirectional=True)
        
        elif layer_type == 'cnn':
            self.layer_type = 'cnn'
            self.embedding_layer = nn.Sequential(
                        nn.Conv1d(in_channels = 200,
                                  out_channels = self.hidden_size,
                                  kernel_size = self.kernel_size),
                        self.tanh)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size))
        memory = Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size))
        if cuda_available:
            hidden = hidden.cuda()
            memory = memory.cuda()
        return (hidden, memory)

    def forward(self, title, body, title_len, body_len):
            
        if self.layer_type == 'lstm':
            
            title_mask = Variable(torch.FloatTensor(build_mask3d(title_len, np.max(title_len))))
            body_mask = Variable(torch.FloatTensor(build_mask3d(body_len, np.max(body_len))))
            
            
            title_out, self.title_hidden = self.embedding_layer(title, (self.tanh(self.title_hidden[0]), \
                                                                   self.tanh(self.title_hidden[1])))
            body_out, self.body_hidden = self.embedding_layer(body, (self.tanh(self.body_hidden[0]), \
                                                                   self.tanh(self.body_hidden[1])))    

        if self.layer_type == 'cnn':

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
            title_out, body_out = title_out.cpu(), body_out.cpu()
            # title_mask = title_mask.cuda()
            # body_mask = body_mask.cuda()
        # print(title_out.size(), title_mask.size())
        title_embeddings = torch.sum(title_out * title_mask, dim=0) / torch.sum(title_mask, dim=0)
        body_embeddings = torch.sum(body_out * body_mask, dim=0) / torch.sum(body_mask, dim=0)
        
        embeddings = ( title_embeddings + body_embeddings ) / 2
        if cuda_available:
            embeddings = embeddings.cuda()
        return embeddings

def save_model(mdl, path):
    # saving model params
    torch.save(mdl.state_dict(), path)

def restore_model(mdl_skeleton, path):
    # restoring params to the mdl skeleton
    mdl_skeleton.load_state_dict(torch.load(path))

def train(layer_type, embedding_layer, batch_size=25, margin=0.3,
          num_epoch=100, id_set=train_idx_set, eval=True):
        
    optimizer = torch.optim.Adam(embedding_layer.parameters(), lr=0.001)
    criterion = multi_margin_loss(hidden=embedding_layer.hidden_size, margin=margin)
    
    qids = list(id_set.keys())
    num_batch = len(qids) // batch_size
    last_loss = np.inf
    for epoch in range(1, num_epoch + 1):
        cumulative_loss = 0
        for batch_idx in range(1, num_batch + 1):
            batch_x_qids = qids[ ( batch_idx - 1 ) * batch_size: batch_idx * batch_size ]
            
            if layer_type == 'lstm':
                batch_title, batch_body, title_len, body_len = process_contxt_batch(batch_x_qids, \
                                                                                train_idx_set)
                embedding_layer.title_hidden = embedding_layer.init_hidden(batch_title.shape[1])
                embedding_layer.body_hidden = embedding_layer.init_hidden(batch_body.shape[1])
            else:
                batch_title, batch_body, title_len, body_len = process_contxt_batch(batch_x_qids, \
                                                                                train_idx_set, batch_first=True)

            title_qs = Variable(torch.FloatTensor(batch_title))#, requires_grad=True)
            body_qs = Variable(torch.FloatTensor(batch_body))#, requires_grad=True)
            if cuda_available:
                title_qs = title_qs.cuda()
                body_qs = body_qs.cuda()
            embeddings = embedding_layer(title_qs, body_qs, title_len, body_len)

            loss = criterion(embeddings)
            cumulative_loss += loss.cpu().data.numpy()[0]
            if cuda_available:
                loss = loss.cuda()

            if batch_idx % 20 == 1:
                print ('epoch:{}/{}, batch:{}/{}, loss:{}'.format(epoch+1, num_epoch+1, batch_idx, num_batch, loss.data[0]))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if eval and batch_idx % print_every == 0: # lstm for now
                print ('evaluating ....')
                if layer_type == 'lstm':
                    do_eval(embedding_layer, 'Dev')
                    print ('------------------')
                    do_eval(embedding_layer, 'Test')
                elif layer_type == 'cnn':
                    do_eval(embedding_layer, 'Dev', batch_first=True)
                    print ('------------------')
                    do_eval(embedding_layer, 'Test', batch_first=True)
            
            del loss, embeddings, title_qs, body_qs
            if layer_type == 'lstm':
                del embedding_layer.title_hidden, embedding_layer.body_hidden

        if eval: # lstm for now
            print ('Epoch {} is finished'.format(epoch))
            if layer_type == 'lstm':
                do_eval(embedding_layer, 'Dev')
                print ('------------------')
                do_eval(embedding_layer, 'Test')
            elif layer_type == 'cnn':
                do_eval(embedding_layer, 'Dev', batch_first=True)
                print ('------------------')
                do_eval(embedding_layer, 'Test', batch_first=True)
        if cumulative_loss < last_loss:
            last_loss = cumulative_loss
        else:
            print('Last loss', last_loss, '/ Current loss', cumulative_loss)
            break

if __name__ == '__main__':

    print('batch_size={}, epoch={}, margin={}, model={}, hidden_size={}'.format( \
        batch_size, epoch, margin, model_option, HIDDEN_DIM))
    print('Start Training...')
    
    if model_option == 'lstm':
        model = EmbeddingLayer(200, HIDDEN_DIM*2, 'lstm')
        criterion = multi_margin_loss(hidden=model.hidden_size * 2)
        if cuda_available:
            model = model.cuda()
        train(model_option, model, batch_size=batch_size, num_epoch=epoch, margin=margin)
    
    elif model_option == 'cnn':  
        model = EmbeddingLayer(200, HIDDEN_DIM, 'cnn')
        # restore_model(model, 'models/lstm_bi_epoch=1_margin=0.3_hidden=667')
        criterion = multi_margin_loss(hidden=model.hidden_size)
        if cuda_available:
            model = model.cuda()
        train(model_option, model, batch_size=batch_size, num_epoch=epoch, margin=margin)
    print ('Saving Model...')
    save_model(model, 'models/'+str(model_option)+'_bi_epoch='+str(epoch)+'_margin='+str(margin)+'_hidden='+str(HIDDEN_DIM)+'_0')
    print ('Done')



