
import re
import spacy 
nlp = spacy.load("en_core_web_sm")
#Pkgs for Text Cleaning
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
def important_words(text): 
    #Removing special characters and digits
    text = re.sub(r'\[[0-9]*\]', ' ', text) #Remove citations
    text = re.sub(r'\s+', ' ', text) #remove whitespaces
    text=re.sub(r'[^\w]',' ',text) #remove anything that is not a word
    text = nlp(text)
    res_words = []
    part_of_speech_tag = ['NOUN'] 
    for word in text:
        if(word.text in STOP_WORDS or word.text in punctuation):
            continue
        if(word.pos_ in part_of_speech_tag):
            res_words.append(word.text.lower())
    from collections import Counter
    counts = Counter(res_words)
    print(counts)
    count=counts
    keywords=[]
    for new_key, new_val in count.items():
        if new_val>=2:
            keywords.append(new_key)
    return keywords





# import spacy
# from string import punctuation
# import re
# nlp = spacy.load("en_core_web_sm")
# from spacy.lang.en.stop_words import STOP_WORDS
# #from nltk import sent_tokenize, word_tokenize
# #list of stop words
# stopwords = list(STOP_WORDS)
# #list of punctuations
# punctuation_list= list(punctuation)

# def important_words(text): 
#     #Removing special characters and digits
#     text = re.sub(r'\[[0-9]*\]', ' ', text) #Remove citations
#     text = re.sub(r'\s+', ' ', text) #remove whitespaces
#     text=re.sub(r'[^\w]',' ',text) #remove anything that is not a word
#     text = nlp(text)
#     res_words = []
#     keywords=[]
#     part_of_speech_tag = ['NOUN'] 
#     for word in text:
#         if(word.text in stopwords or word.text in punctuation_list):
#             continue
#         if(word.pos_ in part_of_speech_tag):
#             res_words.append(word.text)
#     for w in res_words:
#         w=w.lower()
#         if w not in keywords:
#             keywords.append(w)
#             str(keywords)
#     return keywords




