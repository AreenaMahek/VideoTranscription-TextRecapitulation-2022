# VideoTranscription-TextRecapitulation

This study is based on Natural Language Processing where Media in the form of audio and video is converted into a textual format using Audio Extraction using MoviePy python library along with Google Speech Recognition Application Programming Interface (API). The users can summarize the generated text in both Extractive approaches using SpaCy, Natural Language ToolKit (NLTK), Gensim, Term Frequency and Inverse Document Frequency (TF-IDF), TextRank, and Abstractive approaches with Hugging Face Transformer models that include Text-to-Text Transfer Transformer (T5), Generative Pre-trained Transformer-2 (GPT-2), Bidirectional Auto-Regressive Transformers (BART), DistalBART, and Transformer Pipeline Model. The comparison of the different summaries at once helps to reduce reading time and highlights key information along with generating the keywords of the summary using Part of Speech (POS) tagging along with keeping the meaning of the original text intact. The comparison is useful to judge the most accurate summary of them all. The accuracy of the computer-generated summary is compared with the human-generated summary from the dataset is deduced using the Recall-Oriented Understudy for Gisting Evaluation (ROUGE) metric which determines the relevancy and accuracy of the summary, provided through the visualization of the changes in the accuracy values considering different dataset elements. This entire video transcription and the summarization provide a user-friendly interface to reduce the dependency on the manual summarization process and retrieve the most accurate summary to be utilized in our day-to-day lives.


Steps to execute:

Install Flask:
```
pip install flask
```

Install Virtual Environment:
```
pip install virtualenv
```

Create a virtual environment:
```
python -m venv <environment_name>
```

To activate the flask environment:
```
For windows:
venv\Scripts\activate

For remote machine:
source .venv\bin\activate
```

Run the webpage:
```
python app.py
```

