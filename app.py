import os
import string
import requests
import pandas as pd
import streamlit as st
from pytube import YouTube
from zipfile import ZipFile
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

st.markdown('# 📝 **Transcriber and Text Mining App**')
bar = st.progress(0)


# global varible
with open("static/corpus/stopwords pandemic.txt", "r") as f:
     data = f.read()
     must_exist_stopwords = data.split("\n")
     f.close()

porter = PorterStemmer()
stopwords = set(STOPWORDS)
vectorizer = TfidfVectorizer()
api_key = "53dde355832942d78dace4de938bd86f"

# Custom functions 

# 2. Retrieving audio file from YouTube video
def get_yt(URL):
    video = YouTube(URL)
    yt = video.streams.get_audio_only()
    yt.download(output_path = "static/video")

    #st.info('2. Audio file has been retrieved from YouTube video')
    bar.progress(10)

    return yt.get_file_path().split("\\")[-1]

# 3. Upload YouTube audio file to AssemblyAI
def transcribe_yt(file_target: str):

    current_dir = "static/video"
    for file in os.listdir(current_dir)[::-1]:
        if file.endswith(".mp4") and file == file_target:
            mp4_file = os.path.join(current_dir, file)
            # print(mp4_file)

    filename = mp4_file
    bar.progress(20)

    def read_file(filename, chunk_size=5242880):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data
    headers = {'authorization': api_key}
    response = requests.post('https://api.assemblyai.com/v2/upload',
                            headers=headers,
                            data=read_file(filename))
    audio_url = response.json()['upload_url']
    #st.info('3. YouTube audio file has been uploaded to AssemblyAI')
    bar.progress(30)

    # 4. Transcribe uploaded audio file
    endpoint = "https://api.assemblyai.com/v2/transcript"

    json = {
    "audio_url": audio_url
    }

    headers = {
        "authorization": api_key,
        "content-type": "application/json"
    }

    transcript_input_response = requests.post(endpoint, json=json, headers=headers)

    #st.info('4. Transcribing uploaded file')
    bar.progress(40)

    # 5. Extract transcript ID
    transcript_id = transcript_input_response.json()["id"]
    #st.info('5. Extract transcript ID')
    bar.progress(50)

    # 6. Retrieve transcription results
    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    headers = {
        "authorization": api_key,
    }
    transcript_output_response = requests.get(endpoint, headers=headers)
    #st.info('6. Retrieve transcription results')
    bar.progress(60)

    # Check if transcription is complete
    from time import sleep

    while transcript_output_response.json()['status'] != 'completed':
        sleep(5)
        st.warning('Transcription is processing ...')
        transcript_output_response = requests.get(endpoint, headers=headers)
    
    bar.progress(100)

    # 7. Print transcribed text
    st.header('Output')
    st.success(transcript_output_response.json()["text"])

    # 8. Save transcribed text to file
    file = filename.split("\\")[-1].split(".")[0]
    # file = "transcription"

    # Save as TXT file
    # yt_txt = open(f'static/text/{file}.txt', 'w')
    # yt_txt.write(transcript_output_response.json()["text"])
    # yt_txt.close()
    f = open(f'static/text/{file}.txt', 'w')
    f.write(transcript_output_response.json()["text"])
    f.close()

    filename = f'static/{file}.zip'
    zip_file = ZipFile(filename, 'w')
    zip_file.write(f'static/text/{file}.txt')
    zip_file.close()

    return filename

def preprocess_text(text):
    
    # Casefolding
    casefolding = text.casefold()
    casefolding = casefolding.split()
    # casefolding = casefolding.split()
    # print(Casefolding)

    # Tokenization
    tokenization = " ".join(casefolding).split()
    # print(Tokenization)

    # Filtering
    ## Punctuation
    filter_punct = str.maketrans("", "", string.punctuation)
    punctuatization = [word.translate(filter_punct) for word in tokenization]
    # print(Filtering)

    # Stemming
    stemming = [porter.stem(word) for word in punctuatization]
    # print(Stemming)

    # stopwords filtering
    final_text = [i for i in stemming if i in must_exist_stopwords]

    result = {
        "Casefold" : casefolding,
        "Tokenize" : tokenization,
        "Filtering" : punctuatization,
        "Stemming" : stemming
    }
    
    result = pd.DataFrame(result)

    return result, final_text

def generate_wordcloud(word_token):
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(" ".join(word_token))
    fig = plt.figure(figsize = (10, 10))
    plt.imshow(wordcloud)
    plt.axis("off")

    return fig

def readme():
    description = st.empty()
    description.markdown("""
    # About Me :
     Denny Chrystian Baso (17210056)
    ---------------
    Text Mining and Transcriber YouTube Videos App
    ===============
    ---------------
    Informatics Engineering
    ---------------
    MANADO STATE UNIVERSITY
    ---------------
    """)

#####

def run_inference():
    # The App

    #st.info('1. API is read ...')
    st.warning('Awaiting URL input in the sidebar.')


    # Sidebar
    st.sidebar.header('Input Mode')
    selection_mode = st.sidebar.selectbox("",["About", "Transcriber App", "Text Mining App"])

    if selection_mode == "About":
        readme()
        st.empty()

    if selection_mode == "Transcriber App":
        st.sidebar.header('Input URL')

        with st.sidebar.form(key='my_form'):
            URL = st.text_input('Enter URL of YouTube video:')
            submit_button = st.form_submit_button(label='Go')

        # Run custom functions if URL is entered 
        if submit_button:
            file_path = get_yt(URL)
            filename = transcribe_yt(file_path)

            with open(filename, "rb") as zip_download:
                btn = st.download_button(
                    label="Download ZIP",
                    data=zip_download,
                    file_name=filename,
                    mime="application/zip"
                )

    if selection_mode == "Text Mining App":
        dir_text = "static/text"
        st.sidebar.header('Input text')
        with st.sidebar.form(key='my_form'):
            URL = st.selectbox(
                'Select Text Data:', 
                [" "] + [i for i in os.listdir(dir_text) if "txt" in i and i not in ["requirements.txt", "api.txt"]]
                )
            submit_button = st.form_submit_button(label='Go')

        # Run custom function if URL is available and entered
        if submit_button:
            URL = os.path.join(dir_text, URL)
            with open(URL, 'r') as f:
                data = f.read()
                f.close() 

            try:
                # filename as subheader
                st.info(URL)
                try:
                    fname = URL.split(".")[0].split("\\")[1]
                    st.subheader(fname)

                    # text preprocessing
                    st.subheader("Text Preprocessing")
                    data_text, cleaned_text = preprocess_text(data)
                    st.dataframe(data_text)
                
                except Exception as E:
                    fname = URL.split(".")[0].split("/")[-1]
                    st.subheader(fname)

                    # text preprocessing
                    st.subheader("Text Preprocessing")
                    data_text, cleaned_text = preprocess_text(data)
                    st.dataframe(data_text)
            
            except Exception as E:
                st.error(E)
                pass

            try:
                # text separation
                final_text = []
                iteration  = 0
                temp_list  = []

                for word in cleaned_text:
                    if iteration > 5:
                        final_text.append(" ".join(temp_list))
                        iteration = 0
                        temp_list = []
                    elif iteration < 5:
                        temp_list.append(word)
                        iteration += 1
                    else:
                        final_text.append(" ".join(temp_list))
                        iteration = 0
                        temp_list = []

                # TF-IDF algorithm
                st.subheader("TF-IDF Weighting")
                result_weight = vectorizer.fit_transform(final_text)
                text_names, weights = vectorizer.get_feature_names(), result_weight.toarray()
                data_weights = pd.DataFrame.from_records(weights)
                # data_weights = data_weights.set_axis(text_names, axis = 1, inplace = False)
                data_weights = data_weights.set_axis(text_names, axis = 1)
                st.dataframe(data_weights)

                # WordCloud
                st.subheader("Wordcloud Visualisation")
                wordcloud_visual = generate_wordcloud(cleaned_text)
                st.pyplot(wordcloud_visual)

                # Download Stopwords
                with open("static/corpus/stopwords pandemic.txt", "rb") as file_stopword:

                    stopwords_button = st.download_button(
                        label = "Download Stopwords", 
                        data = file_stopword, 
                        file_name = "stopwords.txt"
                    )
                    
            except Exception as E:
                st.error(E)

if __name__ == "__main__":

    current_dir = "static/video"

    # for (root,dirs,files) in os.walk('.', topdown=True):
    #     st.success(root)
    #     st.success(dirs)
    #     st.success(files)
    #     st.success('--------------------------------')

    run_inference()