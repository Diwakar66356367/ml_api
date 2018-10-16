
from flask import Flask, request
import pandas as pd
import numpy as np
import json
import pickle
import os
from old.fastai.text import *
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import stop_words

app = Flask(__name__)


def get_texts(df,lang='en'):
    BOS = 'xbos'  # beginning-of-sentence tag
    FLD = 'xfld'  # data field tag
    texts = f'\n{BOS} {FLD} 1 ' + df
    tok = Tokenizer(lang=lang).proc_all(df,'en')
    return tok

def lemmatize_text(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    word_list=nltk.wordpunct_tokenize(str(text))
    if(len(word_list)==0):
        return ''
    lemmatizetext=' '.join(wordnet_lemmatizer.lemmatize(word) for word in word_list) 
    return lemmatizetext

def remove_stopwords(text):
    word_list=nltk.wordpunct_tokenize(text)
    if len(word_list)==0:
        return ''
    text_after_nonenglish_removal=(' '.join(word for word in word_list if word not in stop_words.ENGLISH_STOP_WORDS))
    return text_after_nonenglish_removal

def removeString(data, regex):
    return data.str.replace(regex, ' ')

def getRegexList():
    regexList = []
    regexList += ['From:(.*)\r\n']  # from line
    regexList += ['Sent:(.*)\r\n']  # sent to line
    regexList += ['Received:(.*)\r\n']  # received data line
    regexList += ['To:(.*)\r\n']  # to line
    regexList += ['CC:(.*)\r\n']  # cc line
    regexList += ['\[cid:(.*)]']  # images cid
    regexList += ['https?:[^\]\n\r]+']  # https & http
    regexList += ['Subject:']
    regexList += ['[\w\d\-\_\.]+@[\w\d\-\_\.]+']  # emails
    regexList += ['[0-9][\-0–90-9 ]+']  # phones
    regexList += ['Subject:']
    regexList += ['com']
    regexList += ['vmware']
    regexList += ['team']
    regexList += ['hi']
    regexList += ['hello']
    regexList += ['thanks']
    regexList += ['thank you']
    regexList += ['thank']
    regexList += ['\.']
    regexList += ['[^a-zA-z 0-9]']
    regexList += [' ']
    regexList += ['nbsp;']
    return regexList

def cleanDataset(df):
    for column in df.columns:
        for regex in getRegexList():
            df[column] = removeString(df[column], regex)
    return df

def remove_singlechar_words(text):
    word_sent=nltk.wordpunct_tokenize(str(text))
    text1=' '.join(temp for temp in word_sent if len(''.join(set(temp)))>1)
    return text1

@app.route('/api', methods=['POST'])
def make_prediction():
    model_path = os.path.join(os.path.pardir,os.path.pardir,'models')
    model_file_path = os.path.join(model_path,'svm_model.pkl')
    tfidf_file_path = os.path.join(model_path,'svm_tfidf.pkl')
    clf_sgd_loaded = pickle.load(open(model_file_path, 'rb'))
    tfidf_sgd_loaded = pickle.load(open(tfidf_file_path, 'rb'))
    df=pd.DataFrame(columns=['TicketDescription','Location'])    
    data = request.get_json(force=True)
    df['TicketDescription']=pd.Series(data['TicketDescription'])
    df['Location']=data['Location']
    
    df=cleanDataset(df)
    df['TicketDescription']=df['TicketDescription'].apply(lambda x: remove_singlechar_words(x))
    df['TicketDescription'].str.strip()
    df['Location'].str.strip()
    df['TicketDesc+Loc']=df[['TicketDescription','Location']].apply(lambda x:' '.join(x),axis=1)
    tok_trn= get_texts(df['TicketDesc+Loc'])
    df['TicketDesc+Loc']=' '.join(tok_trn[0])
    df['TicketDesc+Loc']=df['TicketDesc+Loc'].apply(lambda x: lemmatize_text(x))    
    df['TicketDesc+Loc']=df['TicketDesc+Loc'].apply(lambda x: remove_stopwords(x))   
    
       
    tfidf_fit=tfidf_sgd_loaded.transform(df['TicketDesc+Loc'])
    prediction=clf_sgd_loaded.predict(tfidf_fit)
        
    prediction_pd=pd.DataFrame(prediction)
        
    return prediction_pd.to_json()

if __name__ == '__main__':
    
    # host flask app at port 10001
    
    app.run(port=10001, debug=False)   