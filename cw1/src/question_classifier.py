import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter
import numpy as np
import re
import sys
import random
import configparser

# read and global initialize parameters
def str_to_bool(str):
    return True if str.lower() == 'true' else False

try:
    assert(sys.argv[2] =='-config')
except Exception as e:
    print("The command is wrong: '-config' should be used in command")

parser = configparser.ConfigParser()
parser.read(sys.argv[3])

# path
data_path =  parser.get("File-Path", "data_path")
glove_path = parser.get("File-Path", "glove_path")
test_path = parser.get("File-Path", "test_path")
path_eval_result = parser.get("File-Path", "path_eval_result")

# parameters
validation_size = float(parser.get("Parameters", "validation_size"))
word_appear_times=int(parser.get("Parameters", "word_appear_times"))
WORD_DIM = int(parser.get("Parameters", "WORD_DIM"))
learning_rate = float(parser.get("Parameters", "learning_rate"))
epoches = int(parser.get("Parameters", "epoches"))
is_Pretrain = str_to_bool(parser.get("Parameters", "is_Pretrain"))
is_pre_freeze = str_to_bool(parser.get("Parameters", "is_pre_freeze"))
early_stoping = int(parser.get("Parameters", "early_stoping"))
EMBEDDING_DIM = int(parser.get("Parameters", "EMBEDDING_DIM"))
HIDDEN_DIM = int(parser.get("Parameters", "HIDDEN_DIM"))
model=parser.get("Parameters", "model")

# 1. load data and do preprocessing:
def load_dataset(data_path):
    examples=[]
    reg = "[^0-9A-Za-z]"
    stopword=["?",",","``","''","'",":","."]
    for line in open(data_path,'rb'): # for each line
        label, _, text = line.replace(b'\xf0', b' ').strip().decode().partition(' ')
        sentence = re.sub(reg, ' ', text) 
        final_result = []
        for word in sentence.lower().split():
            if word not in stopword: 
                final_result.append(word) 
        if len(final_result)!= 0:
            examples.append((final_result,label))
    return examples
data_set=load_dataset(data_path)
test_set=load_dataset(test_path)

# create word to index and vocabulary (contain word: #UNK#)
def build_random_vocabulary(word_appear_times):
    allwords=[]
    word_to_ix = {}
    for sent, _ in data_set:
        for word in sent:
            allwords.append(word)
    word_count = dict(Counter(allwords))
    for word, count in word_count.items():
        if count >=word_appear_times:
            word_to_ix[word] = len(word_to_ix)
    if len(word_to_ix)<=0:
        print('The value of words appearing times is too big, change to a smaller one.')
        sys.exit()
    word_to_ix.update({"#UNK#": len(word_to_ix)})
    return word_to_ix

word_to_ix=build_random_vocabulary(word_appear_times)
vocabulary=word_to_ix.copy() 
VOCAB_SIZE = len(vocabulary) # the size of vocabulary

# create label to index and get the size of it, make it be a tensor type.
label_to_ix = {}
for _,label in data_set:
    if label not in label_to_ix:
        label_to_ix[label]=len(label_to_ix)
NUM_LABELS=len(label_to_ix)
def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

def load_glove(glove_path):
    glove_words=[]
    glove_word2idx={}
    glove_weight=[]
    word2index=build_random_vocabulary(1)
    for l in open(glove_path, 'rb'):
        line = l.decode().split()
        word=line[0]
        if  word in word2index.keys():
            glove_words.append(word)
            glove_word2idx[word] = len(glove_word2idx)
            vector = np.array(line[1:]).astype(np.float)
            glove_weight.append(vector)
    return glove_weight, glove_word2idx
glove_wordEd,glove_word2idx=load_glove(glove_path)

# 2. divide into train, test, dev subset.
def spliteDataset(validation_size):
    random.seed(1)
    trainset_length= int(len(data_set)*validation_size)
    train_set=random.sample(data_set,trainset_length)
    dev_set = [x for x in data_set if x not in train_set]
    return train_set,dev_set
train_set, dev_set=spliteDataset(validation_size)

# 4. two kinds of embedding - randomEmbedding and pre-train embedding
if is_Pretrain==False: # print what kinds of embedding you set
    print('Using randomly initialize word embeddings')
else:
    print('Using pre-trained word embeddings')
    
def word_embedding(is_Pretrain,is_pre_freeze):
    if is_Pretrain == True: #use glove pretrain word embedding initialization
        weights=torch.FloatTensor(glove_wordEd)
        embedding = nn.Embedding.from_pretrained(weights, freeze=is_pre_freeze)
    else: # use random word embedding initialization
        torch.manual_seed(10)
        embedding=nn.Embedding(VOCAB_SIZE, WORD_DIM)
    return embedding
embedding= word_embedding(is_Pretrain, is_pre_freeze)

# make sentence to vector by using bow method
def make_bow_vector(sentence, vocabulary):
    sentence_vec=torch.zeros([1,WORD_DIM],dtype=torch.float)
    word_vec=torch.zeros([1,WORD_DIM],dtype=torch.float)
    for word in sentence:
        if word in vocabulary.keys():
            ix=vocabulary[word]
            word_vec= embedding(torch.LongTensor([ix]))
        else:
            ix= vocabulary['#UNK#']
            word_vec= embedding(torch.LongTensor([ix]))
        sentence_vec=torch.add(sentence_vec,word_vec)
    sentence_vec=sentence_vec/len(sentence)
    return sentence_vec

# 5. Three models: Bow, bilstm, bilstm ensemble
class BoWClassifier(nn.Module):  # inheriting from nn.Module!
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)

class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,bidirectional=True)
        self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        tag_scores=torch.mean(tag_scores,dim=0)
        return tag_scores

class BiLSTMEnsembleClassifier(nn.Module):
    def __init__(self, num, embedding_dim, hidden_dim, vocab_size, tagset_size):
        self.model_num=int(num)
        self.models=[]
        for i in range(self.model_num):
            model=BiLSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(label_to_ix))
            self.models.append(model)
    def bagging(self, data):
        num = len(data)
        temp=[]
        for i in range(num):
            temp.append(data[random.randint(0,num-1)])
        return temp
# Training the models
def bow_train(bow_model):
    print("Training the bow model....")
    bow_loss_function = nn.NLLLoss()
    bow_optimizer = optim.SGD(bow_model.parameters(), lr =learning_rate)
    #bow_model.train()
    stopping_step = 0
    best_loss = 9999
    loss_temp = 9999
    for epoch in range(epoches):
        for instance, label in train_set:
            bow_model.zero_grad()
            if is_Pretrain==False:
                bow_vec = make_bow_vector(instance,vocabulary)
            else:
                bow_vec = make_bow_vector(instance,glove_word2idx)
            target = make_target(label, label_to_ix)
            log_probs = bow_model(bow_vec)
            loss = bow_loss_function(log_probs, target)
            loss.backward()
            bow_optimizer.step()
        print('Bow model iteration-',epoch, 'ends')
        error_sum = 0
        for sentence, label in dev_set:
            loss_function=nn.NLLLoss()
            if is_Pretrain == False:
                bow_vec = make_bow_vector(sentence, vocabulary)
            else:
                bow_vec = make_bow_vector(sentence, glove_word2idx)
            target = make_target(label, label_to_ix)
            log_probs_dev = bow_model(bow_vec)
            loss = loss_function(log_probs_dev, target)
            error_sum += loss

        if (float(error_sum) < float(loss_temp)):
            stopping_step = 0
            best_loss=float(error_sum)
            loss_temp=float(error_sum)
        else:
            stopping_step += 1
            loss_temp = float(error_sum)
        if stopping_step >= early_stoping:  # 预设的 这个我们随意
            print("Early stopping is trigger at step: {} loss:{} ".format(epoch-stopping_step, float(best_loss)))
            break

    torch.save(bow_model, '../data/bow_model.pt')

def prepare_sequence(seq, to_ix): #for bilstm
    idxs=[]
    for word in seq:
        if word in to_ix.keys():
            idxs.append(to_ix[word])
        else:
            idxs.append(to_ix['#UNK#'])
    return torch.tensor(idxs, dtype=torch.long)

def bilstm_train(bilstm_model):
    print('Bilstm model is training....')
    bilstm_loss_function = nn.NLLLoss()
    bilstm_optimizer = optim.SGD(bilstm_model.parameters(), lr=learning_rate)
    stopping_step = 0
    best_loss = 9999
    loss_temp = 9999
    for epoch in range(epoches):
        for sentence, tags in train_set:
            label=[]
            label.append(tags)
            bilstm_model.zero_grad() # clear them out before each instance
            if is_Pretrain==False:
                sentence_in = prepare_sequence(sentence, word_to_ix)
            else:
                sentence_in = prepare_sequence(sentence,glove_word2idx)    
            targets = prepare_sequence(label, label_to_ix)
            tag_scores = bilstm_model(sentence_in)
            # Compute the loss, gradients, and update the parameters 
            loss = bilstm_loss_function(tag_scores.view(1,-1), targets)
            loss.backward()
            bilstm_optimizer.step()
        print('Bilstm model iteration-',epoch, 'ends')
        error_sum = 0
        for sentence, tags in dev_set:
            loss_function = nn.NLLLoss()
            label = []
            label.append(tags)
            bilstm_model.zero_grad()  # clear them out before each instance
            if is_Pretrain == False:
                sentence_in = prepare_sequence(sentence, word_to_ix)
            else:
                sentence_in = prepare_sequence(sentence, glove_word2idx)
            targets = prepare_sequence(label, label_to_ix)
            tag_scores = bilstm_model(sentence_in)
            # Compute the loss, gradients, and update the parameters
            loss = loss_function(tag_scores.view(1, -1), targets)
            error_sum += loss
        if (float(error_sum) < float(loss_temp)):
            stopping_step = 0
            best_loss = float(error_sum)
            loss_temp = float(error_sum)
        else:
            stopping_step += 1
            loss_temp = float(error_sum)
        if stopping_step >= early_stoping:  # 预设的 这个我们随意
            print("Early stoping is trigger at step: {} loss:{} ".format(epoch - stopping_step, float(best_loss)))
            break
    torch.save(bilstm_model, '../data/bilstm_model.pt')

def bilstm_ensemble_train(models, trn, model_num):
    print('Bilstm ensemble model is training (it will cost a lot of time)....')
    stopping_step = 0
    best_loss = 9999
    loss_temp = 9999
    for i in range(model_num):
        model=models.models[i]
        bilstm_loss_function = nn.NLLLoss()
        bilstm_optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        # again, normally you would NOT do 300 epochs, it is toy data
        model.zero_grad()  # clear them out before each instance
        for epoch in range(epoches):
            for sentence, tags in models.bagging(train_set):
                label = []
                label.append(tags)
                model.zero_grad()
                if is_Pretrain == False:
                    sentence_in = prepare_sequence(sentence, word_to_ix)
                else:
                    sentence_in = prepare_sequence(sentence, glove_word2idx)
                targets = prepare_sequence(label, label_to_ix)
                tag_scores = model(sentence_in)
                loss = bilstm_loss_function(tag_scores.view(1, -1), targets)
                loss.backward()
                bilstm_optimizer.step()

            error_sum = 0
            for sentence, tags in dev_set:
                loss_function = nn.NLLLoss()
                label = []
                label.append(tags)
                model.zero_grad()  # clear them out before each instance
                if is_Pretrain == False:
                    sentence_in = prepare_sequence(sentence, word_to_ix)
                else:
                    sentence_in = prepare_sequence(sentence, glove_word2idx)
                targets = prepare_sequence(label, label_to_ix)
                tag_scores = model(sentence_in)
                # Compute the loss, gradients, and update the parameters
                loss = loss_function(tag_scores.view(1, -1), targets)
                error_sum += loss

            if (float(error_sum) < float(loss_temp)):
                stopping_step = 0
                best_loss = float(error_sum)
                loss_temp = float(error_sum)
            else:
                stopping_step += 1
                loss_temp = float(error_sum)
            if stopping_step >= early_stoping:
                print("Early stopping is trigger at step: {} loss:{} ".format(epoch - stopping_step, float(best_loss)))
                break
        torch.save(model, '../data/bilstm_ensemble_trained_model' + str(i) + '.pt')
        print('Training Bilstm ensemble-', i, 'ends')
        pass

# 6. testing and evaluate model
precision_list=[]
output=[]
def save_output(path_eval_result,label_to_ix, precision_list,correct_rate,F_score,output):
    with open(path_eval_result,'w') as f:
        f.write('Predicted results for each question in the test set are shown below:')
        f.write('\n')
        for i in range(len(test_set)):
            f.write(output[i])
            f.write('\n')
        f.write('The accuracy is:'+str(correct_rate)+'\t\t'+'An F score is:'+str(F_score))
        f.write('\n')
        f.write('--------------------------------------------------------')
        f.write('\n')
        f.write('The accuracy of each clss of question answer is shown below:')
        f.write('\n')
        for i in range(len(label_to_ix)):
            f.write(str(list(label_to_ix.keys())[i])+'\t\t'+str(precision_list[i]))
            f.write('\n')
        f.close()

def bow_test():
    model = torch.load('../data/bow_model.pt')
    id_to_label = {v: k for k, v in label_to_ix.items()}
    model.eval()
    with torch.no_grad():
        correct_count = 0
        error_count = 0
        correct_rate = 0.0
        confusion_matrix = np.zeros((NUM_LABELS, NUM_LABELS))
        for sentence, label in test_set:
            if is_Pretrain==False:
                bow_vec = make_bow_vector(sentence,vocabulary)
            else:
                 bow_vec = make_bow_vector(sentence,glove_word2idx)
            target = make_target(label, label_to_ix)
            log_prob = model(bow_vec)
            confusion_matrix[label_to_ix[label]][log_prob.argmax().item()]+=1
            output.append(id_to_label[log_prob.argmax().item()])
            if label == id_to_label[log_prob.argmax().item()]:
                correct_count += 1
            else:
                error_count += 1
        correct_rate = correct_count / len(test_set)
        print(confusion_matrix)
        recall=0
        recall_mean=0
        pre=0
        precision_mean=0
        [rows, cols] = confusion_matrix.shape
        for i in range(rows):
            total_row=0
            for j in range(cols):
                  total_row= total_row+ confusion_matrix[i][j]
            if total_row==0:
                recall=0
            else:
             recall=float(confusion_matrix[i][i]/total_row)
            recall_mean+=recall
        recall_mean = float(recall_mean / NUM_LABELS)
        for i in range(cols):
            total_cols=0
            for j in range(rows):
                  total_cols= total_cols+ confusion_matrix[i][j]
            if total_cols==0:
                pre=0
            else:
                pre=float(confusion_matrix[i][i]/total_cols)
            precision_list.append(pre)
            precision_mean+=pre
        precision_mean = float(precision_mean / NUM_LABELS)
        print("correct_rate:", correct_rate)
        F_score=2*recall_mean*precision_mean/(recall_mean+precision_mean)
        print("F score:",F_score)
        save_output(path_eval_result, label_to_ix, precision_list, correct_rate, F_score, output)

def bilstm_test():
    model = torch.load('../data/bilstm_model.pt')
    id_to_label = {v: k for k, v in label_to_ix.items()}
    model.eval()
    with torch.no_grad():
        correct_count = 0
        error_count = 0
        correct_rate = 0.0
        confusion_matrix = np.zeros((NUM_LABELS, NUM_LABELS))
        for sentence, label in test_set:
            label_temp=[]
            label_temp.append(label)
            model.zero_grad() # clear them out before each instance
            if is_Pretrain==False:
                sentence_in = prepare_sequence(sentence, word_to_ix)
            else:
                sentence_in = prepare_sequence(sentence,glove_word2idx)
            log_prob = model(sentence_in)
            output.append(id_to_label[log_prob.argmax().item()])
            confusion_matrix[label_to_ix[label]][log_prob.argmax().item()]+=1

            if label == id_to_label[log_prob.argmax().item()]:
                correct_count += 1
            else:
                error_count += 1
        correct_rate = correct_count / len(test_set)
        print(confusion_matrix)
        recall=0
        recall_mean=0
        pre=0
        precision_mean=0
        [rows, cols] = confusion_matrix.shape
        for i in range(rows):
            total_row=0
            for j in range(cols):
                  total_row= total_row+ confusion_matrix[i][j]
            if total_row==0:
                recall=0
            else:
             recall=float(confusion_matrix[i][i]/total_row)
            recall_mean+=recall
        recall_mean = float(recall_mean /NUM_LABELS)

        for i in range(cols):
            total_cols=0
            for j in range(rows):
                  total_cols= total_cols+ confusion_matrix[i][j]
            if total_cols==0:
                pre=0
            else:
                pre=float(confusion_matrix[i][i]/total_cols)
            precision_list.append(pre)
            precision_mean+=pre
        precision_mean = float(precision_mean / NUM_LABELS)
        print("correct_rate:", correct_rate)
        F_score = 2 * recall_mean * precision_mean / (recall_mean + precision_mean)
        print("F score:", F_score)
        save_output(path_eval_result, label_to_ix, precision_list, correct_rate, F_score, output)

def bilstm_ensemble_test(size):
    confusion_matrix = np.zeros((NUM_LABELS, NUM_LABELS))
    id_to_label = {v: k for k, v in label_to_ix.items()}
    results_dic = {}
    final_pre=[]
    for i in range(size):
        results_dic[i]=[]
        filename='../data/bilstm_ensemble_trained_model'+str(i)+'.pt'
        train_model=torch.load(filename)
        with torch.no_grad():
            correct_count = 0
            correct_rate = 0.0
            for sentence, label in test_set:
                train_model.zero_grad()  # clear them out before each instance
                if is_Pretrain == False:
                    sentence_in = prepare_sequence(sentence, word_to_ix)
                else:
                    sentence_in = prepare_sequence(sentence, glove_word2idx)
                log_prob = train_model(sentence_in)
                results_dic[i].append(log_prob.argmax().item())
    for i in range(len(results_dic[0])):
        array = []
        for j in range(size):
            array.append(results_dic[j][i])
        final_pre.append(np.argmax(np.bincount(array)))
    for i in range(len(final_pre)):
        confusion_matrix[label_to_ix[test_set[i][1]]][final_pre[i]] += 1
        if test_set[i][1] == id_to_label[final_pre[i]]:
            correct_count += 1
        output.append(id_to_label[final_pre[i]])
    correct_rate = correct_count / len(test_set)
    print(confusion_matrix)
    recall = 0
    recall_mean = 0
    pre = 0
    precision_mean = 0
    [rows, cols] = confusion_matrix.shape
    for i in range(rows):
        total_row = 0
        for j in range(cols):
            total_row = total_row + confusion_matrix[i][j]
        if total_row == 0:
            recall = 0
        else:
            recall = float(confusion_matrix[i][i] / total_row)
        recall_mean += recall
    recall_mean = float(recall_mean / NUM_LABELS)

    for i in range(cols):
        total_cols = 0
        for j in range(rows):
            total_cols = total_cols + confusion_matrix[i][j]
        if total_cols == 0:
            pre = 0
        else:
            pre = float(confusion_matrix[i][i] / total_cols)
        precision_list.append(pre)
        precision_mean += pre
    precision_mean = float(precision_mean / NUM_LABELS)
    print("correct_rate:", correct_rate)
    print("correct_num:",correct_count)
    F_score = 2 * recall_mean * precision_mean / (recall_mean + precision_mean)
    print("F score:", F_score)
    save_output(path_eval_result, label_to_ix, precision_list, correct_rate, F_score, output)


# Determine input command: train and test which model
def select_operation():
    if sys.argv[1] ==  'train':
        if model=='bow':
            bow_model = BoWClassifier(NUM_LABELS, WORD_DIM)
            bow_train(bow_model)
        elif model=='bilstm':
            bilstm_model = BiLSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), NUM_LABELS)
            bilstm_train(bilstm_model)
        elif model=='bilstm_ensemble':
            ensemble_size = int(parser.get("Parameters", "ensemble_size"))
            bilstm_ensemble_model = BiLSTMEnsembleClassifier(ensemble_size,EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), NUM_LABELS)
            bilstm_ensemble_train(bilstm_ensemble_model,train_set,ensemble_size)
        else:
            print('The parameter of model should be bow or bilstm')
    elif sys.argv[1] ==  'test':
        if model=='bow':
            bow_test()
        elif model=='bilstm':
            bilstm_test()
        elif model =='bilstm_ensemble':
            ensemble_size = int(parser.get("Parameters", "ensemble_size"))
            bilstm_ensemble_test(ensemble_size)          
        else:
            print('The parameter of model should be bow or bilstm')
    else:       
        print('The command is wrong. For training, pleae run: For testing, please run:')
select_operation()
