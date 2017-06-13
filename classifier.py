import argparse
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_selection.from_model import SelectFromModel
import cPickle
import json
import sys
import kenlm

parser = argparse.ArgumentParser()
parser.add_argument('--lm', help='trained klm language model')
parser.add_argument('--model', default=False, help='pickled featureset')
parser.add_argument('--test_data', default='test', help='data to test')
parser.add_argument('--rand_forest', default=False, help='use random forest for feature selection') 
args = parser.parse_args()
dets = ['a', 'an', 'the', None]
vowels = ['a', 'e', 'i', 'o', 'u']
null = None

class Model:
  def __init__(self):
    self.model = LogisticRegression()
    self.vec = DictVectorizer() 
    self.featSelect = None
        
  #fits a maximum-entropy model based on features
  def fitModel(self, data):
    allFeatures = [x[0] for x in data]
    allLabels = [x[1] for x in data]
    #transforms the dict of features to a vector
    X = self.vec.fit_transform(allFeatures).toarray().astype(np.float) 
    y = np.array(allLabels).astype(np.float)
    if args.rand_forest:
      self.featSelect = SelectFromModel(RandomForestClassifier()).fit(X,y)
      X = self.featSelect.transform(X)
    self.model.fit(X,y)

  #predicts correction based on features
  def predict(self, features):
    x = self.vec.transform(features).toarray().astype(np.float)
    if args.rand_forest:
      x = self.featSelect.transform(x)
    probs = self.model.predict_proba(x)[0]
    pred_label = int(self.model.predict(x)[0])
    return [dets[pred_label], probs[pred_label]]    
 
#extracts the training data and the features
def get_features(data_type): 
  all_data = []
  all_labels = [] 
  data_filename = 'data/sentence_'+data_type+'.txt'
  label_filename = 'data/corrections_'+data_type+'.txt'
  uncount_nouns = 'data/uncountable_nouns.txt'
  plural_nouns = 'data/plural_nouns.txt'
  singular_nouns = 'data/singular_nouns.txt'
  pos_filename = 'data/pos_tags_'+data_type+'.txt'
  deps_filename = 'data/dependencies_'+data_type+'.txt'
  ngrams_filename = 'data/ngrams.txt'
 
  with open(data_filename, 'r') as train, \
  open(uncount_nouns, 'r') as uncounts, \
  open(plural_nouns, 'r') as plurals, \
  open(singular_nouns, 'r') as sings, \
  open(pos_filename, 'r') as postags, \
  open(deps_filename, 'r') as deps, \
  open(ngrams_filename, 'r') as ngrams: 
    if data_type == 'train':
      with open(label_filename, 'r') as labels:
        all_labels = json.load(labels)
    all_lines = json.load(train) 
    uncounts = json.load(uncounts)
    plurals = json.load(plurals) 
    sings = json.load(sings)
    postags = json.load(postags)
    deps = json.load(deps)
    ngrams = json.load(ngrams)

    if args.lm:
     lmodel = kenlm.LanguageModel(args.lm) 

    for sent_idx in range(0, len(all_lines)):
      for word_idx in range(0, len(all_lines[sent_idx])):
          featureset = {} 
          word = all_lines[sent_idx][word_idx].lower() 
          if word in dets:
            featureset['word_idx'] = word_idx
            featureset['word'] =  word 

            #next 4 pos tags
            for i in range(0, 5):
              try:
                featureset['pos_next'+str(i)] = postags[sent_idx][word_idx+i]
              except:    
                featureset['pos_next'+str(i)] = 0
  
            #previous 3 pos tags
            for i in range(0, 4):
              try:
                featureset['pos_prev'+str(i)] = postags[sent_idx][word_idx-i]
              except:
                featureset['pos_prev'+str(i)] = 0

            #next 2 words
            for i in range(0, 3):
              try:
                featureset['next_word'+str(i)] = all_lines[sent_idx][word_idx+i].lower()
              except:
                featureset['next_word'+str(i)] = 0
  
            #checks if next word starts with a vowel, if next word is capitalized
            if word_idx < len(all_lines[sent_idx])-1:
              featureset['vowel'] = all_lines[sent_idx][word_idx+1][0].lower() in vowels and word == 'an'
              featureset['cap'] = all_lines[sent_idx][word_idx+1][0].isupper()
            else:
              featureset['vowel'] = False 
              featureset['cap'] = 0
 
            #previous word in the sentence
            if word_idx >= 1:
              featureset['prev_word'] = all_lines[sent_idx][word_idx-1].lower()
            else:
              featureset['prev_word'] = 0

            #finds head word that determiner is attached to
            for dep in deps[sent_idx]:
              if int(dep['childIndex']) == word_idx:
                head_idx = int(dep['headIndex'])
                head_word = all_lines[sent_idx][head_idx].lower()
            featureset['head'] = head_word
            featureset['singular'] = head_word in sings
            featureset['plural'] = head_word in plurals
            featureset['uncountable'] = head_word in uncounts 

            #checks if NP is in ngrams
            try:
              ngrams[" ".join(all_lines[sent_idx][word_idx:head_idx+1]).lower()]
              featureset['np-gram'] = True
            except:
              featureset['np-gram'] = False
            
            #checks if head noun is in the previous sentence or in the next sentence
            if sent_idx == 0:
              featureset['in-prev'] = False
            else:
              featureset['in-prev'] = head_word in all_lines[sent_idx-1]
            try:
              featureset['in-next'] = head_word in all_lines[sent_idx+1]
            except:
              featureset['in-next'] = False
            
            #language model score for entire sentence
            if args.lm:
              featureset['lm-score'] = lmodel.score(" ".join(all_lines[sent_idx]))
            else:
              featureset['lm-score'] = None     

            #add label if train
            if all_labels:
              label = dets.index(all_labels[sent_idx][word_idx])
              all_data.append([featureset, label])
            else:
              all_data.append([featureset])

          #if current word is not a determiner, no change
          else:
            all_data.append(null) 
          #marks end of the sentence to be recovered later
          if word_idx == len(all_lines[sent_idx])-1:
            all_data.append("****")
  #pickle the featureset for use on other test sets
  outfile = open('features.pickle', 'wb')
  pickler = cPickle.Pickler(outfile, cPickle.HIGHEST_PROTOCOL)
  pickler.fast = 1
  pickler.dump(all_data)
  outfile.close()
  return all_data
      
    

if __name__ == '__main__':
  model = Model()
  print >> sys.stderr, "getting data..."
  #open a trained featureset
  if args.model:
    FILE = open("features.pickle", 'r')
    f = FILE.read()
    train_data = cPickle.loads(f)
    FILE.close()
  else:
    train_data = get_features('train')
  train_data = [x for x in train_data if x and x != "****"]
  print >> sys.stderr, "fitting model..."
  model.fitModel(train_data)
  print >> sys.stderr, 'predicting on new data...'
  dev_data = get_features(args.test_data)
  all_sents = []
  sent = []
  for word_idx in range(0, len(dev_data)): 
    word = dev_data[word_idx]
    #if None, append None
    if not word:
      sent.append(word)
    #recover sentence boundary
    elif word == "****":
      all_sents.append(sent)
      sent = []
    #else, predict correction
    else:
      prediction, probability = model.predict(word[0])
      if prediction != word and prediction != None:
        sent.append([prediction, probability])
      else:
        sent.append(None)
  print >> sys.stderr, 'dumping results...'
  with open('submission_'+args.test_data+'.txt', 'wb') as output:
    json.dump(all_sents, output)



