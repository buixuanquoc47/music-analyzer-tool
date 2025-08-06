import streamlit as st
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
import joblib
import tempfile
import os
import pyloudnorm as pyln

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

st.title("🎵 Advanced music analysis tool with integrated Machine Learning")

uploaded_file = st.file_uploader("Tải file nhạc (MP3, WAV)", type=["mp3", "wav"])

def convert_to_wav(input_path, output_path="temp.wav"):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(22050)
    audio.export(output_path, format="wav")
    return output_path

def extract_librosa_features(wav_path):
    y, sr = librosa.load(wav_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (np.ndarray, list)):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)
    return mfcc_mean, chroma_mean, tempo, y, sr

def extract_pyaudio_features(wav_path):
    Fs, x = audioBasicIO.read_audio_file(wav_path)
    F, _ = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    F_mean = np.mean(F, axis=1)
    return F_mean

def estimate_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F',
            'F#', 'G', 'G#', 'A', 'A#', 'B']

    chroma_norm = (chroma_mean - np.min(chroma_mean)) / (np.max(chroma_mean) - np.min(chroma_mean) + 1e-6)

    best_correlation = -np.inf
    best_key = None
    best_mode = None

    for i in range(12):
        rotated_major = np.roll(major_profile, i)
        rotated_minor = np.roll(minor_profile, i)

        corr_major = np.corrcoef(chroma_norm, rotated_major)[0, 1]
        corr_minor = np.corrcoef(chroma_norm, rotated_minor)[0, 1]

        if corr_major > best_correlation:
            best_correlation = corr_major
            best_key = keys[i]
            best_mode = 'Major'

        if corr_minor > best_correlation:
            best_correlation = corr_minor
            best_key = keys[i]
            best_mode = 'Minor'

    return best_key, best_mode

def measure_loudness_lufs(y, sr):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    return loudness

def combine_features(mfcc, chroma, tempo, pyaudio_feats):
    combined = np.hstack((mfcc, chroma, tempo, pyaudio_feats))
    return combined

MODEL_PATH = "multioutput_rf_model_extended.pkl"

if not os.path.exists(MODEL_PATH):
    st.warning("Chưa có model huấn luyện. Vui lòng train model trước khi sử dụng phần ML.")
    model = None
else:
    model = joblib.load(MODEL_PATH)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_wav_path = convert_to_wav(tmp_file.name)

    mfcc, chroma, tempo, y, sr = extract_librosa_features(temp_wav_path)
    pyaudio_feats = extract_pyaudio_features(temp_wav_path)
    combined_features = combine_features(mfcc, chroma, tempo, pyaudio_feats)

    st.audio(uploaded_file, format='audio/mp3')
    st.write(f"🎵 Tempo (BPM): {tempo:.2f}")

    key, mode_music = estimate_key(y, sr)
    st.write(f"🎼 Dự đoán key: {key} ({mode_music})")

    lufs = measure_loudness_lufs(y, sr)
    st.write(f"🔊 Loudness chuẩn LUFS: {lufs:.2f} LUFS")

    # Cảnh báo loudness ngoài khoảng -20 đến -12 LUFS
    if lufs > -12:
        st.warning("⚠️ Loudness quá cao! Nên giảm âm lượng để phù hợp với thể loại Jazz, R&B cổ điển.")
    elif lufs < -20:
        st.warning("⚠️ Loudness quá thấp! Nên tăng âm lượng để phù hợp với thể loại Jazz, R&B cổ điển.")
    else:
        st.success("🎉 Loudness chuẩn LUFS nằm trong khoảng phù hợp.")

    if model is not None:
        predicted_scores = model.predict(combined_features.reshape(1, -1))[0]
        predicted_scores_float = [float(x) for x in predicted_scores]
        columns = ['Technique', 'Expression', 'Creativity', 'Structure',
                   'Sound Quality', 'Tradition & Innovation', 'Engagement', 'Total Score',
                   'Genre Structure', 'Melody Flow', 'Lead Clarity', 'Instrument Coordination',
                   'Background Quality', 'Background Richness', 'Emotion Guidance',
                   'Song Structure', 'Tempo Mood', 'Genre Fit', 'Intro Quality',
                   'Hook Quality', 'Lead Role', 'Verse Quality', 'Outro Quality']

        st.subheader("💡 Dự đoán điểm chi tiết bài nhạc (ML MultiOutput):")
        for col, score in zip(columns, predicted_scores_float):
            st.write(f"{col}: {score:.2f}")

        total_score_index = columns.index('Total Score')
        total_score_value = predicted_scores_float[total_score_index]
        st.markdown(f"### ⭐ Điểm tổng trung bình (Total Score): **{total_score_value:.2f}**")

    st.subheader("📊 Dạng sóng âm thanh")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    st.subheader("📈 Phổ tần số (Spectrogram)")
    Xdb = librosa.amplitude_to_db(np.abs(librosa.stft(y)))
    fig2, ax2 = plt.subplots()
    img = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.f dB")
    st.pyplot(fig2)
