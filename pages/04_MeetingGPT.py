import streamlit as st
import subprocess
import math
import glob
import openai
from pydub import AudioSegment
from utils import embed_file
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser

llm = ChatOpenAI(
    temperature=0.1
)

has_transcript = os.path.exists("./.cache/podcast.txt")

@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    
    audio_path = video_path.replace("mp4", "mp3")
    command = [
		"ffmpeg", 
        "-y", 
		"-i", # input
		video_path, # 비디오 파일 경로
		"-vn", # 비디오는 무시
		audio_path # 오디오 파일이 저장될 경로
	]
    subprocess.run(command) # command를 파이썬에서 실행하도록 해주는 subprocess
      
@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_minute, chunks_folder):
    if has_transcript:
        return
    
    track = AudioSegment.from_mp3(audio_path) # 오디오 파일을 리스트처럼 조작할 수 있도록 해주는 pydub
    chunk_len = chunk_minute * 60 * 1000 # pydub은 밀리초 단위
    chunks = math.ceil(len(track) / chunk_len) # 몇개로 분할할지

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = start_time + chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")

@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()

    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            # "a"는 append 모드로 파일의 끝에 붙여줌
            transcript = openai.Audio.transcribe(
                "whisper-1", 
                audio_file
            )
            text_file.write(transcript["text"])


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="🗣️"
)

st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])

if video:
    chunk_folder = "./.cache/chunks"

    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)

        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)

        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunk_folder)

        status.update(label="Transcribing audio...")
        transcribe_chunks(chunk_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript", 
            "Summary", 
            "Q&A"
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())
        
    with summary_tab:
        start = st.button("Generate Summary")

        if start:
            # 두 체인이 필요함. 하나는 문서를 요약하기 위한 것
            # 하나는 이전의 요약과 새 context로 새로운 요약을 위한 것
            loader = TextLoader(transcript_path)
            spliiter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size = 800,
                chunk_overlap = 100,
            )

            docs = loader.load_and_split(text_splitter = spliiter)
            # st.write(docs)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                다음을 핵심 위주로 요약해줘:
                "{text}"
                요약 결과:
                """
            )
            
            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            summary = first_summary_chain.invoke({
                "text": docs[0].page_content
            })
            
            refine_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """
                    당신은 요약 전문가입니다. 아래 지침을 엄격히 따르세요:
                    1. 기존 요약본({existing_summary})과 새로운 내용({context})을 통합하여 '최종 요약본'을 만드세요.
                    2. 절대로 내용을 그대로 번역하거나 같은 말을 반복하지 마세요.
                    3. 중복되는 내용은 삭제하고, 전체 흐름이 자연스러운 한국어로 요약하세요.
                    """
                )
            ])   

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i + 1}/{len(docs) - 1}")
                    summary = refine_chain.invoke({
                        "existing_summary": summary,
                        "context": doc.page_content
                    })
                    st.write(summary)
            st.write(summary)