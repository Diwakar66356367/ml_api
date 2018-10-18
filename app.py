
from flask import Flask, abort, jsonify, request, render_template
import pandas as pd
import numpy as np
import json
import pickle
import os
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import re
import spacy
from spacy import load
import en_core_web_sm

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

class Tokenizer():
    def __init__(self, lang='en'):
        self.re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        self.tok = spacy.load(lang)

    def sub_br(self,x): return self.re_br.sub("\n", x)

    def spacy_tok(self,x):
        return [t.text for t in self.tok.tokenizer(self.sub_br(x))]

    re_rep = re.compile(r'(\S)(\1{3,})')
    re_word_rep = re.compile(r'(\b\w+\W+)(\1{3,})')

    @staticmethod
    def replace_rep(m):
        TK_REP = 'tk_rep'
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '

    @staticmethod
    def replace_wrep(m):
        TK_WREP = 'tk_wrep'
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '

    @staticmethod
    def do_caps(ss):
        TOK_UP,TOK_SENT,TOK_MIX = ' t_up ',' t_st ',' t_mx '
        res = []
        prev='.'
        re_word = re.compile('\w')
        re_nonsp = re.compile('\S')
        for s in re.findall(r'\w+|\W+', ss):
            res += ([TOK_UP,s.lower()] if (s.isupper() and (len(s)>2))
    #                 else [TOK_SENT,s.lower()] if (s.istitle() and re_word.search(prev))
                    else [s.lower()])
    #         if re_nonsp.search(s): prev = s
        return ''.join(res)

    def proc_text(self, s):
        s = self.re_rep.sub(Tokenizer.replace_rep, s)
        s = self.re_word_rep.sub(Tokenizer.replace_wrep, s)
        s = Tokenizer.do_caps(s)
        s = re.sub(r'([/#])', r' \1 ', s)
        s = re.sub(' {2,}', ' ', s)
        return self.spacy_tok(s)

    @staticmethod
    def proc_all(ss, lang):
        tok = Tokenizer(lang)
        return [tok.proc_text(s) for s in ss]



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
    print('Executed App')
    clf_sgd_loaded = pickle.load(open('svm_model.pkl', 'rb'))
    tfidf_sgd_loaded = pickle.load(open('svm_tfidf.pkl', 'rb'))
    df=pd.DataFrame(columns=['TicketDescription','Location']) 
    result=request.form
    df['TicketDescription']=pd.Series(result['ticketdescription'])
    df['Location']=pd.Series(result['location'])
    df=cleanDataset(df)
    df['TicketDescription']=df['TicketDescription'].apply(lambda x: remove_singlechar_words(x))
    df['TicketDescription'].str.strip()
    df['Location'].str.strip()
    df['TicketDesc+Loc']=df[['TicketDescription','Location']].apply(lambda x:' '.join(x),axis=1)
    tok_trn= get_texts(df['TicketDesc+Loc'])
    df['TicketDesc+Loc']=' '.join(tok_trn[0])
    df['TicketDesc+Loc']=df['TicketDesc+Loc'].apply(lambda x: lemmatize_text(x))  
    
       
    tfidf_fit=tfidf_sgd_loaded.transform(df['TicketDesc+Loc'])
    prediction=clf_sgd_loaded.predict(tfidf_fit)
    prediction_pd=pd.DataFrame(prediction)
        
        
    return prediction_pd.to_json()

if __name__ == '__main__':
    
    # host flask app at port 10001
    
    app.run(port=10001, debug=False)   