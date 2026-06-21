import io
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sns.set_style('whitegrid')

@st.cache_data
def create_stopword_remover():
    factory = StopWordRemoverFactory()
    return factory.create_stop_word_remover()

@st.cache_data
def create_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stopword_remover = create_stopword_remover()
stemmer = create_stemmer()

@st.cache_data
def to_sentiment(rating):
    try:
        rating = int(rating)
    except (ValueError, TypeError):
        return np.nan
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

@st.cache_data
def remove_emoji(text):
    if not isinstance(text, str):
        return ''
    emoji_pattern = re.compile(
        '['
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U00002600-\U000026FF"
        u"\u200d"
        ']+',
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)

@st.cache_data
def remove_kaomoji(text):
    if not isinstance(text, str):
        return ''
    kaomoji_pattern = re.compile(
        r'(?:[°oO\-_=^][>.<]?[°oO\-_=^])|(?:[<>]?[()Oo|][<>]?)|(?:[><]?[_.-]?[><])|\((?:[^\]\[\)\(\/\\]*?[°oO\-_=^][^\]\[\)\(\/\\]*?)\)'
    )
    return kaomoji_pattern.sub(r'', text)

@st.cache_data
def remove_power(text):
    if not isinstance(text, str):
        return ''
    power_pattern = re.compile(r'[\u00B2\u00B3\u00B9\u2070-\u207F\u2080-\u208F]')
    return power_pattern.sub(r'', text)

@st.cache_data
def bersihkan_teks(teks):
    if not isinstance(teks, str):
        return ''
    teks = teks.lower()
    teks = re.sub(r'[^\n\w\s]', '', teks)
    teks = re.sub(r'\d+', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

@st.cache_data
def remove_stopwords_indonesian(text):
    if not isinstance(text, str):
        return ''
    return stopword_remover.remove(text)

@st.cache_data
def stem_indonesian_text(text):
    if not isinstance(text, str):
        return ''
    return stemmer.stem(text)

@st.cache_data
def preprocess_review(text):
    text = remove_emoji(text)
    text = remove_kaomoji(text)
    text = remove_power(text)
    text = bersihkan_teks(text)
    text = remove_stopwords_indonesian(text)
    return stem_indonesian_text(text)

@st.cache_data
def prepare_dataset(df):
    df = df.copy()
    if 'content' not in df.columns:
        raise ValueError('Kolom content tidak ditemukan di dataset.')

    df = df.dropna(subset=['content'])
    df = df.drop_duplicates().reset_index(drop=True)
    df['content'] = df['content'].astype(str)
    df['cleaned_review'] = df['content'].apply(remove_emoji)
    df['cleaned_review'] = df['cleaned_review'].apply(remove_kaomoji)
    df['cleaned_review'] = df['cleaned_review'].apply(remove_power)
    df['preprocessed_review'] = df['cleaned_review'].apply(bersihkan_teks)
    df['final_cleaned_review'] = df['preprocessed_review'].apply(remove_stopwords_indonesian)
    df['stemmed_review'] = df['final_cleaned_review'].apply(stem_indonesian_text)
    if 'score' in df.columns:
        df['sentiment'] = df['score'].apply(to_sentiment)
    else:
        df['sentiment'] = np.nan

    if 'at' in df.columns:
        try:
            df['at'] = pd.to_datetime(df['at'], errors='coerce')
        except Exception:
            pass

    return df

@st.cache_data
def train_model(df):
    x = df['stemmed_review'].astype(str)
    y = df['sentiment']
    if y.isna().any():
        raise ValueError('Kolom sentiment harus berisi nilai yang valid.')

    vectorizer = TfidfVectorizer(max_features=3500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(x)
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X, y)
    return model, vectorizer

@st.cache_data
def generate_wordcloud(text):
    if not text:
        text = 'data kosong'
    wc = WordCloud(width=400, height=400, background_color='white', collocations=False)
    return wc.generate(text)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

st.set_page_config(page_title='Sentiment Review App', layout='wide')

st.title('Aplikasi Analisis Sentimen Review Aplikasi')
st.write('Unggah dataset review Google Play, lakukan preprocessing teks Indonesia, dan prediksi sentimen secara interaktif.')

uploaded_file = st.sidebar.file_uploader('Unggah file CSV review', type=['csv'])
show_full_dataset = st.sidebar.checkbox('Tampilkan dataset lengkap', value=False)

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f'Gagal membaca file CSV: {exc}')
        st.stop()

    try:
        df = prepare_dataset(df_raw)
    except Exception as exc:
        st.error(f'Gagal memproses dataset: {exc}')
        st.stop()

    st.subheader('Preview Data')
    st.write('Dataset setelah preprocessing dan pembersihan.')
    st.dataframe(df.head(10))

    if show_full_dataset:
        st.subheader('Dataset Lengkap')
        st.dataframe(df)

    st.markdown('---')
    st.subheader('Ringkasan Statistik')
    col1, col2, col3 = st.columns(3)
    col1.metric('Jumlah Baris', df.shape[0])
    col2.metric('Jumlah Kolom', df.shape[1])
    duplicate_count = int(df.duplicated().sum())
    col3.metric('Baris Duplikat', duplicate_count)

    st.markdown('### Filter Interaktif')
    sentiment_options = {
        'Semua': None,
        'Negatif': 0,
        'Netral': 1,
        'Positif': 2,
    }
    selected_sentiment = st.sidebar.selectbox('Pilih Sentiment', list(sentiment_options.keys()), index=0)
    selected_app_version = None
    if 'appVersion' in df.columns:
        unique_versions = df['appVersion'].dropna().astype(str).unique().tolist()
        unique_versions = sorted(unique_versions)[:50]
        selected_app_version = st.sidebar.selectbox('Pilih App Version', ['Semua'] + unique_versions, index=0)

    start_date = None
    end_date = None
    if 'at' in df.columns and pd.api.types.is_datetime64_any_dtype(df['at']):
        min_date = df['at'].min().date()
        max_date = df['at'].max().date()
        start_date, end_date = st.sidebar.date_input('Rentang tanggal', [min_date, max_date])

    df_filtered = df.copy()
    if sentiment_options[selected_sentiment] is not None:
        df_filtered = df_filtered[df_filtered['sentiment'] == sentiment_options[selected_sentiment]]
    if selected_app_version and selected_app_version != 'Semua':
        df_filtered = df_filtered[df_filtered['appVersion'].astype(str) == selected_app_version]
    if start_date is not None and end_date is not None and 'at' in df_filtered.columns:
        df_filtered = df_filtered[(df_filtered['at'].dt.date >= start_date) & (df_filtered['at'].dt.date <= end_date)]

    st.write(f'Jumlah data setelah filter: {df_filtered.shape[0]}')
    st.dataframe(df_filtered.head(10))

    st.markdown('---')
    st.subheader('Visualisasi Distribusi')
    fig_score, ax_score = plt.subplots(figsize=(6, 4))
    if 'score' in df.columns:
        sns.countplot(x='score', data=df_filtered, palette='viridis', ax=ax_score)
        ax_score.set_title('Distribusi Score')
        ax_score.set_xlabel('Score')
        ax_score.set_ylabel('Jumlah Review')
        st.pyplot(fig_score)
    else:
        st.info('Kolom score tidak tersedia di dataset sehingga grafik distribusi score tidak dapat ditampilkan.')

    fig_sentiment, ax_sentiment = plt.subplots(figsize=(6, 4))
    if df_filtered['sentiment'].notna().any():
        sns.countplot(x='sentiment', data=df_filtered, palette='viridis', ax=ax_sentiment)
        ax_sentiment.set_title('Distribusi Sentiment')
        ax_sentiment.set_xlabel('Sentiment (0=Negatif, 1=Netral, 2=Positif)')
        ax_sentiment.set_ylabel('Jumlah Review')
        ax_sentiment.set_xticks([0, 1, 2])
        ax_sentiment.set_xticklabels(['Negatif', 'Netral', 'Positif'])
        st.pyplot(fig_sentiment)
    else:
        st.info('Sentiment belum tersedia untuk visualisasi.')

    if 'appVersion' in df.columns:
        st.markdown('### Top 10 App Version')
        version_counts = df['appVersion'].value_counts().head(10)
        fig_version, ax_version = plt.subplots(figsize=(8, 5))
        sns.barplot(x=version_counts.values, y=version_counts.index, palette='viridis', ax=ax_version)
        ax_version.set_title('Top 10 App Version')
        ax_version.set_xlabel('Jumlah Review')
        ax_version.set_ylabel('App Version')
        st.pyplot(fig_version)

    st.markdown('---')
    st.subheader('Word Cloud per Sentiment')
    wordcloud_cols = st.columns(3)
    texts = {
        'Positif': df[df['sentiment'] == 2]['stemmed_review'].str.cat(sep=' '),
        'Netral': df[df['sentiment'] == 1]['stemmed_review'].str.cat(sep=' '),
        'Negatif': df[df['sentiment'] == 0]['stemmed_review'].str.cat(sep=' '),
    }
    for idx, (label, text) in enumerate(texts.items()):
        with wordcloud_cols[idx]:
            st.write(f'**{label}**')
            wc = generate_wordcloud(text)
            fig_wc, ax_wc = plt.subplots(figsize=(4, 4))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

    st.markdown('---')
    st.subheader('Model Sentiment')
    try:
        model, vectorizer = train_model(df)
        st.success('Model dilatih menggunakan data saat ini.')
    except Exception as exc:
        st.error(f'Gagal melatih model: {exc}')
        model = None
        vectorizer = None

    with st.expander('Coba prediksi review baru'):
        user_text = st.text_area('Masukkan review aplikasi di sini', height=120)
        if st.button('Prediksi Sentiment'):
            if not user_text:
                st.warning('Masukkan teks review terlebih dahulu.')
            elif model is None or vectorizer is None:
                st.error('Model belum tersedia untuk prediksi.')
            else:
                processed = preprocess_review(user_text)
                x_vec = vectorizer.transform([processed])
                pred = model.predict(x_vec)[0]
                proba = model.predict_proba(x_vec)[0]
                label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
                st.write('**Review asli:**', user_text)
                st.write('**Teks setelah preprocessing:**', processed)
                st.write('**Prediksi sentiment:**', label_map.get(pred, str(pred)))
                st.write('**Probabilitas:**')
                st.write(
                    pd.DataFrame(
                        {
                            'Kelas': ['Negatif', 'Netral', 'Positif'],
                            'Probabilitas': np.round(proba, 4),
                        }
                    )
                )

    csv_data = convert_df_to_csv(df)
    st.download_button(
        label='Unduh dataset hasil preprocessing',
        data=csv_data,
        file_name='preprocessed_review_dataset.csv',
        mime='text/csv',
    )
else:
    st.info('Unggah file CSV review terlebih dahulu. Pastikan dataset memiliki kolom `content` dan (opsional) `score`, `appVersion`, `at`.')
    st.write('Contoh baris pada dataset:')
    st.write('content, score, appVersion, at')
