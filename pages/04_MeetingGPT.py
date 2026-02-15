import streamlit as st
import subprocess
import math
import glob
import openai
from pydub import AudioSegment
from utils import embed_file
import os

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

    with st.status("Loading video..."):
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)

    with st.status("Extracting audio..."):
        extract_audio_from_video(video_path)

    with st.status("Cutting audio segments..."):
        cut_audio_in_chunks(audio_path, 10, chunk_folder)

    with st.status("Transcribing audio..."):
        transcribe_chunks(chunk_folder, transcript_path)
