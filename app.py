# #Flask Pkg
from __future__ import unicode_literals
from flask import Flask,render_template,url_for
from flask import request, redirect
import flask
app = Flask(__name__)
from matplotlib import pyplot as plt
import numpy as np


#Extractive summarization
import os
import math
from TFIDF_extractive_summarization import TF_IDF

#Spacy Pkg
import spacy
import time
from textrank import textrank
from spacy_summarization import spacy_summarizer
nlp = spacy.load("en_core_web_lg")
from keywordgeneration import important_words
from rouge_measure import rouge_scoring

 
# #Gensim
from gensim_summarizer import g_summarizer
from gensim.summarization.summarizer import summarize


#NLTK
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk_summarization import nltk_summarizer

#Abstractive summarization
#HuggingFace Transformers
from transformers import pipeline
from abstractivet5 import abstractivet5

from abstractiveautotokenizer import abstractiveautotokenizer
from abstractivebart import bart
from transformersabs import transformers_abs
from gpt import gpt


# Web Scraping Pkg
from bs4 import BeautifulSoup
import requests


# Reading Time
def readingTime(mytext):
    #nlp( ) makes human language recognisable by computer programs
	total_words = len([ token.text for token in nlp(mytext)])
    #reading 200 words per minute is the average reading speed of a human
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
    req = requests.get(url)
    parsed_text = BeautifulSoup(req.text, 'html.parser') #soup=Beautifulsoup(doc.html,html.parser)
    foundresult = parsed_text.find_all(['p'])
    text = [result.text for result in foundresult]
    article = ' '.join(text)
    return article
	

@app.route('/')
def index():
	return render_template('index.html')


#Extractive summarization using NLTK
@app.route('/analyze',methods=['GET','POST'])
def analyze():
    start = time.time()
    if request.method == 'POST':
         rawtext = request.form['rawtext']
         final_reading_time = readingTime(rawtext)
         final_summary = nltk_summarizer(rawtext)
         summary_reading_time = readingTime(final_summary)
         keywords = important_words(final_summary)
         rouge_s=rouge_scoring(final_summary,rawtext)
         end = time.time()
         final_time = end-start
         return render_template('index.html',ctext=rawtext,keywords=keywords,rouge_s=rouge_s,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

         
@app.route('/analyze_url',methods=['GET','POST'])
def analyze_url():
    start = time.time()
    if request.method == 'POST':
         raw_url = request.form['raw_url']
         rawtext=get_text(raw_url)
         final_reading_time = readingTime(rawtext)
         final_summary = nltk_summarizer(rawtext)
         summary_reading_time = readingTime(final_summary)
         keywords = important_words(final_summary)
         rouge_s=rouge_scoring(final_summary,rawtext)
         end = time.time()
         final_time = end-start
         return render_template('index.html',ctext=rawtext,keywords=keywords,rouge_s=rouge_s,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

         

@app.route('/analyze_gpt',methods=['GET','POST'])
def analyze_gpt():
    start = time.time()
    if request.method == 'POST':
         rawtext = request.form['rawtext']
         final_reading_time = readingTime(rawtext)
         final_summary = gpt(rawtext)
         summary_reading_time = readingTime(final_summary)
         keywords = important_words(final_summary)
         rouge_s=rouge_scoring(final_summary,rawtext)
         end = time.time()
         final_time = end-start
         return render_template('index.html',ctext=rawtext,keywords=keywords,rouge_s=rouge_s,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)


@app.route('/analyze_url_gpt',methods=['GET','POST'])
def analyze_url_gpt():
    start = time.time()
    if request.method == 'POST':
         raw_url = request.form['raw_url']
         rawtext=get_text(raw_url)
         final_reading_time = readingTime(rawtext)
         final_summary = gpt(rawtext)
         summary_reading_time = readingTime(final_summary)
         keywords = important_words(final_summary)
         rouge_s=rouge_scoring(final_summary,rawtext)
         end = time.time()
         final_time = end-start
         return render_template('index.html',ctext=rawtext,keywords=keywords,rouge_s=rouge_s,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)



@app.route('/compare_summary')
def compare_summary():
 	return render_template('compare_summary.html')
 
    
@app.route('/compare_summary_abs')
def compare_summary_abs():
 	return render_template('compare_summary_abs.html')
 
 
@app.route('/aboutapp')
def aboutapp():
	return render_template('aboutapp.html')



@app.route('/comparer',methods=['GET','POST'])
def comparer():
    global rouge_s_spacy,rouge_s_gensim,rouge_s_nltk,rouge_s_tfidf
    start = time.time()
    if request.method == 'POST':
         rawtext = request.form['rawtext']
         final_reading_time = readingTime(rawtext)
         
          #SpaCy Summarizer
         final_summary_spacy = spacy_summarizer(rawtext)
         keywords_spacy=important_words(final_summary_spacy)
         rouge_s_spacy=rouge_scoring(final_summary_spacy,rawtext)
         summary_reading_time = readingTime(final_summary_spacy)
         
          # Gensim Summarizer
         final_summary_gensim = g_summarizer(rawtext)
         keywords_gensim=important_words(final_summary_gensim)
         rouge_s_gensim=rouge_scoring(final_summary_gensim,rawtext)
         summary_reading_time_gensim = readingTime(final_summary_gensim)
         
         # NLTK
         final_summary_nltk = nltk_summarizer(rawtext)
         keywords_nltk=important_words(final_summary_nltk)
         rouge_s_nltk=rouge_scoring(final_summary_nltk,rawtext)
         summary_reading_time_nltk = readingTime(final_summary_nltk)
         
         # TFIDF
         final_summary_tfidf = TF_IDF(rawtext)
         keywords_tfidf=important_words(final_summary_tfidf)
         rouge_s_tfidf=rouge_scoring(final_summary_tfidf,rawtext)
         summary_reading_time_tfidf = readingTime(final_summary_tfidf) 
         
         # # TextRank
         # final_summary_textrank = textrank(rawtext)
         # keywords_textrank=important_words(final_summary_textrank)
         # rouge_s_textrank=rouge_scoring(final_summary_textrank,rawtext)
         # summary_reading_time_textrank = readingTime(final_summary_textrank) 
         
         
         #bar_chart()
         end = time.time()
         final_time = end-start
         #return render_template('compare_summary.html',ctext=rawtext,keywords_spacy=keywords_spacy,keywords_nltk=keywords_nltk,keywords_gensim=keywords_gensim,keywords_tfidf=keywords_tfidf,keywords_textrank=keywords_textrank,rouge_s_spacy=rouge_s_spacy,rouge_s_gensim=rouge_s_gensim,rouge_s_nltk=rouge_s_nltk,rouge_s_tfidf=rouge_s_tfidf,rouge_s_textrank=rouge_s_textrank,final_summary_gensim=final_summary_gensim,final_summary_nltk=final_summary_nltk,final_summary_spacy=final_summary_spacy,final_summary_tfidf=final_summary_tfidf,final_summary_textrank=final_summary_textrank,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time_gensim=summary_reading_time_gensim,summary_reading_time_nltk=summary_reading_time_nltk,summary_reading_time=summary_reading_time,summary_reading_time_tfidf=summary_reading_time_tfidf,summary_reading_time_textrank=summary_reading_time_textrank)
         return render_template('compare_summary.html',ctext=rawtext,keywords_spacy=keywords_spacy,keywords_nltk=keywords_nltk,keywords_gensim=keywords_gensim,keywords_tfidf=keywords_tfidf,rouge_s_spacy=rouge_s_spacy,rouge_s_gensim=rouge_s_gensim,rouge_s_nltk=rouge_s_nltk,rouge_s_tfidf=rouge_s_tfidf,final_summary_gensim=final_summary_gensim,final_summary_nltk=final_summary_nltk,final_summary_spacy=final_summary_spacy,final_summary_tfidf=final_summary_tfidf,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time_gensim=summary_reading_time_gensim,summary_reading_time_nltk=summary_reading_time_nltk,summary_reading_time=summary_reading_time,summary_reading_time_tfidf=summary_reading_time_tfidf)



@app.route('/comparer_url',methods=['GET','POST'])
def comparer_url():
 	start = time.time()
 	if request.method == 'POST':
         raw_url = request.form['raw_url']
         rawtext = get_text(raw_url)
         final_reading_time = readingTime(rawtext)
         
          #SpaCy Summarizer
         final_summary_spacy = spacy_summarizer(rawtext)
         keywords_spacy=important_words(final_summary_spacy)
         rouge_s_spacy=rouge_scoring(final_summary_spacy,rawtext)
         summary_reading_time = readingTime(final_summary_spacy)
         
          # Gensim Summarizer
         final_summary_gensim = g_summarizer(rawtext)
         keywords_gensim=important_words(final_summary_gensim)
         rouge_s_gensim=rouge_scoring(final_summary_gensim,rawtext)
         summary_reading_time_gensim = readingTime(final_summary_gensim)
         
         
         # NLTK
         final_summary_nltk = nltk_summarizer(rawtext)
         keywords_nltk=important_words(final_summary_nltk)
         rouge_s_nltk=rouge_scoring(final_summary_nltk,rawtext)
         summary_reading_time_nltk = readingTime(final_summary_nltk)
        
         
         # TFIDF
         final_summary_tfidf = TF_IDF(rawtext)
         keywords_tfidf=important_words(final_summary_tfidf)
         rouge_s_tfidf=rouge_scoring(final_summary_tfidf,rawtext)
         summary_reading_time_tfidf = readingTime(final_summary_tfidf)
         
         
         # # TextRank
         # final_summary_textrank = textrank(rawtext)
         # keywords_textrank=important_words(final_summary_textrank)
         # rouge_s_textrank=rouge_scoring(final_summary_textrank,rawtext)
         # summary_reading_time_textrank = readingTime(final_summary_textrank)
         
         
         
         end = time.time()
         final_time = end-start
         return render_template('compare_summary.html',ctext=rawtext,keywords_spacy=keywords_spacy,keywords_nltk=keywords_nltk,keywords_gensim=keywords_gensim,keywords_tfidf=keywords_tfidf,rouge_s_spacy=rouge_s_spacy,rouge_s_gensim=rouge_s_gensim,rouge_s_nltk=rouge_s_nltk,rouge_s_tfidf=rouge_s_tfidf,final_summary_gensim=final_summary_gensim,final_summary_nltk=final_summary_nltk,final_summary_spacy=final_summary_spacy,final_summary_tfidf=final_summary_tfidf,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time_gensim=summary_reading_time_gensim,summary_reading_time_nltk=summary_reading_time_nltk,summary_reading_time=summary_reading_time,summary_reading_time_tfidf=summary_reading_time_tfidf)




@app.route('/comparer_abs',methods=['GET','POST'])
def comparer_abs():
 	start = time.time()
 	if request.method == 'POST':
         rawtext = request.form['rawtext']
         final_reading_time = readingTime(rawtext)
         
          #T5Tokenizer Model
         final_summary_abstractivet5 = abstractivet5(rawtext)
         keywords_abstractivet5=important_words(final_summary_abstractivet5)
         rouge_s_t5=rouge_scoring(final_summary_abstractivet5,rawtext)
         summary_reading_time_t5 = readingTime(final_summary_abstractivet5)
         
         
         #GPT Model
         final_summary_gpt = gpt(rawtext)
         keywords_gpt=important_words(final_summary_gpt)
         rouge_s_gpt=rouge_scoring(final_summary_gpt,rawtext)
         summary_reading_time_gpt = readingTime(final_summary_gpt)
         
         
         
          # AutoTokenizer Model
         final_summary_abstractiveautotokenizer = abstractiveautotokenizer(rawtext)
         keywords_autotokenizer=important_words(final_summary_abstractiveautotokenizer)
         rouge_s_auto=rouge_scoring(final_summary_abstractiveautotokenizer,rawtext)
         summary_reading_time_abstractiveautotokenizer = readingTime(final_summary_abstractiveautotokenizer )
         
         
          # BART Model
         final_summary_bart = bart(rawtext)
         keywords_bart=important_words(final_summary_bart)
         rouge_s_bart=rouge_scoring(final_summary_bart,rawtext)
         summary_reading_time_bart = readingTime(final_summary_bart)
         
         
         # # Transformers Model
         # final_summary_trans = transformers_abs(rawtext)
         # keywords_trans=important_words(final_summary_trans)
         # rouge_s_trans=rouge_scoring(final_summary_trans,rawtext)
         # summary_reading_time_trans = readingTime(final_summary_trans)
         

         end = time.time()
         final_time = end-start
         #return render_template('compare_summary_abs.html',ctext=rawtext,rouge_s_t5=rouge_s_t5,rouge_s_gpt=rouge_s_gpt,rouge_s_auto=rouge_s_auto,rouge_s_bart=rouge_s_bart,rouge_s_trans=rouge_s_trans,keywords_gpt=keywords_gpt,keywords_bart=keywords_bart,keywords_abstractivet5=keywords_abstractivet5,keywords_autotokenizer=keywords_autotokenizer,keywords_trans=keywords_trans,final_summary_abstractiveautotokenizer=final_summary_abstractiveautotokenizer,final_summary_abstractivet5=final_summary_abstractivet5,final_summary_gpt=final_summary_gpt, final_summary_bart=final_summary_bart,final_summary_trans=final_summary_trans,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time_abstractiveautotokenizer=summary_reading_time_abstractiveautotokenizer,summary_reading_time_gpt=summary_reading_time_gpt,summary_reading_time_t5=summary_reading_time_t5,summary_reading_time_bart=summary_reading_time_bart,summary_reading_time_trans=summary_reading_time_trans)
         return render_template('compare_summary_abs.html',ctext=rawtext,rouge_s_t5=rouge_s_t5,rouge_s_gpt=rouge_s_gpt,rouge_s_auto=rouge_s_auto,rouge_s_bart=rouge_s_bart,keywords_gpt=keywords_gpt,keywords_bart=keywords_bart,keywords_abstractivet5=keywords_abstractivet5,keywords_autotokenizer=keywords_autotokenizer,final_summary_abstractiveautotokenizer=final_summary_abstractiveautotokenizer,final_summary_abstractivet5=final_summary_abstractivet5,final_summary_gpt=final_summary_gpt, final_summary_bart=final_summary_bart,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time_abstractiveautotokenizer=summary_reading_time_abstractiveautotokenizer,summary_reading_time_gpt=summary_reading_time_gpt,summary_reading_time_t5=summary_reading_time_t5,summary_reading_time_bart=summary_reading_time_bart)




@app.route('/comparer_url_abs',methods=['GET','POST'])
def comparer_url_abs():
 	start = time.time()
 	if request.method == 'POST':
         raw_url = request.form['raw_url']
         rawtext = get_text(raw_url)
         final_reading_time = readingTime(rawtext)
         
           #T5Tokenizer Model
         final_summary_abstractivet5 = abstractivet5(rawtext)
         keywords_abstractivet5=important_words(final_summary_abstractivet5)
         rouge_s_t5=rouge_scoring(final_summary_abstractivet5,rawtext)
         summary_reading_time_t5 = readingTime(final_summary_abstractivet5)
         
         
          #GPT Model
         final_summary_gpt = gpt(rawtext)
         keywords_gpt=important_words(final_summary_gpt)
         rouge_s_gpt=rouge_scoring(final_summary_gpt,rawtext)
         summary_reading_time_gpt = readingTime(final_summary_gpt)
         
         
          # T5AutoTokenizer Model
         final_summary_abstractiveautotokenizer = abstractiveautotokenizer(rawtext)
         keywords_autotokenizer=important_words(final_summary_abstractiveautotokenizer)
         rouge_s_auto=rouge_scoring(final_summary_abstractiveautotokenizer,rawtext)
         summary_reading_time_abstractiveautotokenizer = readingTime(final_summary_abstractiveautotokenizer )
         
         
         # BART Model
         final_summary_bart = bart(rawtext)
         keywords_bart=important_words(final_summary_bart)
         rouge_s_bart=rouge_scoring(final_summary_bart,rawtext)
         summary_reading_time_bart = readingTime(final_summary_bart)
         
        
         
         # # Transformers Model
         # final_summary_trans = transformers_abs(rawtext)
         # keywords_trans=important_words(final_summary_trans)
         # rouge_s_trans=rouge_scoring(final_summary_trans,rawtext)
         # summary_reading_time_trans = readingTime(final_summary_trans)
         
         end = time.time()
         final_time = end-start
         #return render_template('compare_summary_abs.html',ctext=rawtext,rouge_s_t5=rouge_s_t5,rouge_s_gpt=rouge_s_gpt,rouge_s_auto=rouge_s_auto,rouge_s_bart=rouge_s_bart,rouge_s_trans=rouge_s_trans,keywords_gpt=keywords_gpt,keywords_bart=keywords_bart,keywords_abstractivet5=keywords_abstractivet5,keywords_autotokenizer=keywords_autotokenizer,keywords_trans=keywords_trans,final_summary_abstractiveautotokenizer=final_summary_abstractiveautotokenizer,final_summary_abstractivet5=final_summary_abstractivet5,final_summary_gpt=final_summary_gpt, final_summary_bart=final_summary_bart,final_summary_trans=final_summary_trans,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time_abstractiveautotokenizer=summary_reading_time_abstractiveautotokenizer,summary_reading_time_gpt=summary_reading_time_gpt,summary_reading_time_t5=summary_reading_time_t5,summary_reading_time_bart=summary_reading_time_bart,summary_reading_time_trans=summary_reading_time_trans)
         return render_template('compare_summary_abs.html',ctext=rawtext,rouge_s_t5=rouge_s_t5,rouge_s_gpt=rouge_s_gpt,rouge_s_auto=rouge_s_auto,rouge_s_bart=rouge_s_bart,keywords_gpt=keywords_gpt,keywords_bart=keywords_bart,keywords_abstractivet5=keywords_abstractivet5,keywords_autotokenizer=keywords_autotokenizer,final_summary_abstractiveautotokenizer=final_summary_abstractiveautotokenizer,final_summary_abstractivet5=final_summary_abstractivet5,final_summary_gpt=final_summary_gpt, final_summary_bart=final_summary_bart,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time_abstractiveautotokenizer=summary_reading_time_abstractiveautotokenizer,summary_reading_time_gpt=summary_reading_time_gpt,summary_reading_time_t5=summary_reading_time_t5,summary_reading_time_bart=summary_reading_time_bart)


         
         
         
         
@app.route('/text_generator')  
def text_generator():  
    return render_template("text_generator.html")  

@app.route('/audio_text_generator')
def audio_text_generator():
 	return render_template('audio_text_generator.html')

 
@app.route('/generation', methods = ['POST'])  
def generation():  
    start = time.time()
    global fff
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        
        import moviepy.editor
        video_file = moviepy.editor.VideoFileClip(f.filename)
        audio = video_file.audio
        audio.write_audiofile("audio.wav" )
        print("Audio Generation Completed!")
       
        import speech_recognition as sr 
        import os 
        from pydub import AudioSegment
        from pydub.silence import split_on_silence
        
        # create a speech recognition object
        r = sr.Recognizer()
        

        # a function that splits the audio file into chunks
        # and applies speech recognition
        def get_large_audio_transcription(path):
            """
            Splitting the large audio file into chunks
            and apply speech recognition on each of these chunks
            """
            # open the audio file using pydub
            sound = AudioSegment.from_wav(path)  
            # split audio sound where silence is 500 miliseconds to get chunks
            chunks = split_on_silence(sound,
            # experiment with this value for your target audio file
            min_silence_len = 500,
            # adjusting as per requirement
            silence_thresh = sound.dBFS-16,
            # keep the silence for 500 milisecond, keep silence at the beginning and end
            #so that it does not sound as the audio was abruptly cut off
            keep_silence=500,)
            folder_name = "audio-chunks"
            # create a directory to store the audio chunks
            if not os.path.isdir(folder_name):
                os.mkdir(folder_name)
            whole_text = ""
            # process each chunk 
            for i, audio_chunk in enumerate(chunks, start=1):
                # export audio chunk and save it in the 'folder_name' directory.
                chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
                audio_chunk.export(chunk_filename, format="wav")
                # recognize the chunk
                with sr.AudioFile(chunk_filename) as source:
                    audio_listened = r.record(source) #to make a dummy object for sound
                    # try converting it to text
                    try:
                        text = r.recognize_google(audio_listened) # toretrievetextfromaudio
                    except sr.UnknownValueError as e:
                        print("Music:", str(e))
                    else:
                        text = f"{text.capitalize()}. "
                        print(chunk_filename, ":", text)
                        whole_text += text
                    
            # return the text for all chunks detected
            
            global fff, final_reading_time,final_time,summary_reading_time
            fff=whole_text
            
            final_reading_time = readingTime(fff)
            summary_reading_time = readingTime(fff)
            end = time.time()
            final_time = end-start
            return whole_text
        
        
        path = "audio.wav"
        final_text= get_large_audio_transcription(path)
        print("\nFull text:",final_text)
        return render_template("text_generator.html",rawtext=final_text,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

    
@app.route('/generation_audio', methods = ['POST'])  
def generation_audio():
    
    start = time.time()
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        import speech_recognition as sr 
        import os 
        from pydub import AudioSegment
        from pydub.silence import split_on_silence
        
        # create a speech recognition object
        r = sr.Recognizer()
        
        src=f.filename
        dst = "audio.wav"
        # convert any file to wav
        sound = AudioSegment.from_file(src) #open sound with pydub from any source file       
        sound.export(dst, format="wav") #export the sound in the form of dst at location
        path=dst
        
        

        # a function that splits the audio file into chunks
        # and applies speech recognition
        def get_large_audio_transcription(path):
            """
            Splitting the large audio file into chunks
            and apply speech recognition on each of these chunks
            """
            
            # open the audio file using pydub
            sound = AudioSegment.from_wav(path)  
            # split audio sound where silence is 500 miliseconds or more and get chunks
            chunks = split_on_silence(sound,
            # experiment with this value for your target audio file
            min_silence_len = 500,
            # adjust this per requirement
            silence_thresh = sound.dBFS-14,
            # keep the silence for 500 milisecond, adjustable as well
            keep_silence=500,)
            folder_name = "audio-chunks"
            # create a directory to store the audio chunks
            if not os.path.isdir(folder_name):
                os.mkdir(folder_name)
            whole_text = ""
            # process each chunk 
            for i, audio_chunk in enumerate(chunks, start=1):
                # export audio chunk and save it in
                # the `folder_name` directory.
                chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
                audio_chunk.export(chunk_filename, format="wav")
                # recognize the chunk
                with sr.AudioFile(chunk_filename) as source:
                    audio_listened = r.record(source)
                    # try converting it to text
                    try:
                        text = r.recognize_google(audio_listened)
                    except sr.UnknownValueError as e:
                        print("Music:", str(e))
                    else:
                        text = f"{text.capitalize()}. "
                        print(chunk_filename, ":", text)
                        whole_text += text
                        
            # return the text for all chunks detected
            global fff, final_reading_time,final_time,summary_reading_time
            fff=whole_text
            
            final_reading_time = readingTime(fff)
            summary_reading_time = readingTime(fff)
            end = time.time()
            final_time = end-start
            return whole_text
        
        path="audio.wav"
        final_text= get_large_audio_transcription(path)
        print("\nFull text:",final_text)
        return render_template("audio_text_generator.html",rawtext=final_text,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)


@app.route('/youtube_url',methods=['GET','POST'])
def youtube_url():
    global fff
    start = time.time()
    from youtube_transcript_api import YouTubeTranscriptApi
    if request.method == 'POST':
        raw_url = request.form['raw_url']
        youtube_video = raw_url
        video_id = youtube_video.split("=")[1]
        print(video_id)
        #from IPython.display import YouTubeVideo
        #print(YouTubeVideo(video_id))
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        #print(transcript[0:5])
        final_result = " "
        for i in transcript:
            final_result += '. ' + i['text']
            final_result = '. '.join(i.capitalize() for i in final_result.split('. '))
          
        fff=final_result
        final_reading_time = readingTime(fff)
        summary_reading_time = readingTime(fff)
        end = time.time()
        final_time = end-start
    return render_template("text_generator.html",rawtext=final_result,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)



@app.route('/summarizedata',methods=['GET','POST'])
def summarizedata ():
    start = time.time()
    if request.method == 'POST': 
        final_reading_time = readingTime(fff)
        final_summary = TF_IDF(fff)
        keywords = important_words(final_summary)
        rouge_s_tfidf=rouge_scoring(final_summary,fff)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end-start 
    return render_template('index.html',ctext=fff,keywords=keywords,rouge_s=rouge_s_tfidf,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)


@app.route('/summarizedataspacy',methods=['GET','POST'])
def summarizedataspacy():
    start = time.time()
    if request.method == 'POST': 
        final_reading_time = readingTime(fff)
        final_summary = spacy_summarizer(fff)
        keywords = important_words(final_summary)
        rouge_s_spacy=rouge_scoring(final_summary,fff)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end-start 
    return render_template('index.html',ctext=fff,keywords=keywords,rouge_s=rouge_s_spacy,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)


@app.route('/summarizedatanltk',methods=['GET','POST'])
def summarizedatanltk():
    start = time.time()
    if request.method == 'POST': 
        final_reading_time = readingTime(fff)
        final_summary = nltk_summarizer(fff)
        keywords = important_words(final_summary)
        rouge_s_nltk=rouge_scoring(final_summary,fff)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end-start 
    return render_template('index.html',ctext=fff,keywords=keywords,rouge_s=rouge_s_nltk,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)



@app.route('/summarizedatagensim',methods=['GET','POST'])
def summarizedatagensim():
    start = time.time()
    if request.method == 'POST':
        final_reading_time = readingTime(fff)
        final_summary = g_summarizer(fff)
        keywords = important_words(final_summary)
        rouge_s_gensim=rouge_scoring(final_summary,fff)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end-start 
    return render_template('index.html',ctext=fff,keywords=keywords,rouge_s=rouge_s_gensim,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)


@app.route('/summarizedatatextrank',methods=['GET','POST'])
def summarizedatatextrank():
    start = time.time()
    if request.method == 'POST': 
        final_reading_time = readingTime(fff)
        final_summary = textrank(fff)
        keywords = important_words(final_summary)
        rouge_s_textrank=rouge_scoring(final_summary,fff)
        summary_reading_time= readingTime(final_summary)
        end = time.time()
        final_time = end-start 
    return render_template('index.html',ctext=fff,keywords=keywords,rouge_s=rouge_s_textrank,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)



#Transformer Pipeline with DistalBart
@app.route('/summarizedata_abs',methods=['GET','POST'])
def summarizedata_abs():
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    summarizer = pipeline("summarization")
    
    if request.method == 'POST':
        final_reading_time = readingTime(fff)
        ARTICLE=fff
        max_chunk = 500
        ARTICLE = ARTICLE.replace('.', '.<eos>')
        ARTICLE = ARTICLE.replace('?', '?<eos>')
        ARTICLE = ARTICLE.replace('!', '!<eos>')
        sentences = ARTICLE.split('<eos>')
        current_chunk = 0  
        chunks = []
        for sentence in sentences:
            if len(chunks) == current_chunk + 1:
                if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                    chunks[current_chunk].extend(sentence.split(' '))
                else:
                    current_chunk += 1
                    chunks.append(sentence.split(' '))
            else:
                print(current_chunk)
                chunks.append(sentence.split(' '))
        for chunk_id in range(len(chunks)):
            chunks[chunk_id] = ' '.join(chunks[chunk_id])
            
        res = summarizer(chunks, max_length=150, min_length=30, do_sample=False)
        final_summary=' '.join([summ['summary_text'] for summ in res])
        keywords = important_words(final_summary)
        rouge_s_trans=rouge_scoring(final_summary,fff)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end-start
    return render_template('index.html',ctext=fff,keywords=keywords,rouge_s=rouge_s_trans,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)



#T5
@app.route('/summarizedatat5',methods=['GET','POST'])
def summarizedatat5():
    start = time.time()
    if request.method == 'POST': 
        final_reading_time = readingTime(fff)
        final_summary = abstractivet5(fff)
        keywords = important_words(final_summary)
        rouge_s_t5=rouge_scoring(final_summary,fff)
        summary_reading_time= readingTime(final_summary)
        end = time.time()
        final_time = end-start 
    return render_template('index.html',ctext=fff,keywords=keywords,rouge_s=rouge_s_t5,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)


#DistalBart
@app.route('/summarizedataauto',methods=['GET','POST'])
def summarizedataauto():
    start = time.time()
    if request.method == 'POST': 
        final_reading_time = readingTime(fff)
        final_summary = abstractiveautotokenizer(fff)
        keywords = important_words(final_summary)
        rouge_s_auto=rouge_scoring(final_summary,fff)
        summary_reading_time= readingTime(final_summary)
        end = time.time()
        final_time = end-start 
    return render_template('index.html',ctext=fff,keywords=keywords,rouge_s=rouge_s_auto,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)


#BART
@app.route('/summarizedatabart',methods=['GET','POST'])
def summarizedatabart():
    start = time.time()
    if request.method == 'POST': 
        final_reading_time = readingTime(fff)
        final_summary = bart(fff)
        keywords = important_words(final_summary)
        rouge_s_bart=rouge_scoring(final_summary,fff)
        summary_reading_time= readingTime(final_summary)
        end = time.time()
        final_time = end-start 
    return render_template('index.html',ctext=fff,keywords=keywords,rouge_s=rouge_s_bart,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)


#GPT
@app.route('/summarizedatagpt',methods=['GET','POST'])
def summarizedatagpt():
    start = time.time()
    if request.method == 'POST': 
        final_reading_time = readingTime(fff)
        final_summary = gpt(fff)
        keywords = important_words(final_summary)
        rouge_s_gpt=rouge_scoring(final_summary,fff)
        summary_reading_time= readingTime(final_summary)
        end = time.time()
        final_time = end-start 
    return render_template('index.html',ctext=fff,keywords=keywords,rouge_s=rouge_s_gpt,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

        

if __name__ == '__main__':
 	app.run()  
