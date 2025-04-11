import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Streamlit page configuration
st.set_page_config(page_title='Netflix Data Analysis', layout='wide', initial_sidebar_state='expanded')
st.title('📺 Netflix Data Analysis and Recommendation App')
st.markdown('Upload your **netflix_titles.csv** to begin the analysis.')

# Add custom dark theme toggle (CSS hack)
theme = st.sidebar.selectbox("Select Theme", ("Light", "Dark"))

# Apply custom CSS based on the selected theme
if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #121212; color: white; }
        .stApp { background-color: #121212; color: white; }
        .css-18e3th9 { background-color: #121212; color: white; }
        .css-1d391kg { background-color: #1e1e1e; }
        .css-1cpxqw2 { color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body { background-color: #FFFFFF; color: black; }
        .stApp { background-color: #FFFFFF; color: black; }
        .css-18e3th9 { background-color: #FFFFFF; color: black; }
        .css-1d391kg { background-color: #f0f2f6; }
        .css-1cpxqw2 { color: black; }
        </style>
        """,
        unsafe_allow_html=True
    )

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    netflix_df = pd.read_csv(uploaded_file)
    netflix_df['date_added'] = netflix_df['date_added'].str.strip()
    netflix_df['date_added'] = pd.to_datetime(netflix_df['date_added'])
    netflix_df['director'].dropna(inplace=True)
    netflix_df['cast'].dropna(inplace=True)
    netflix_df['description'].dropna(inplace=True)
    netflix_df['country'].dropna(inplace=True)
    netflix_df['rating'].dropna(inplace=True)
    netflix_df['listed_in'].dropna(inplace=True)
    netflix_df['duration'].dropna(inplace=True)

    # Clean the data
    # netflix_df['director'].fillna('No Director', inplace=True)
    # netflix_df['cast'].fillna('No Cast', inplace=True)
    # netflix_df['description'].fillna('No Description', inplace=True)
    # netflix_df['country'].fillna('Unknown', inplace=True)
    # netflix_df['date_added'] = pd.to_datetime(netflix_df['date_added'])

    # Sidebar selection
    analysis_option = st.sidebar.selectbox(
        'Choose Analysis',
        ('Top Directors', 'Word Cloud', 'Content Ratings', 'Frequent Actors', 'Duration Comparison',
         'Monthly Additions', 'Top Countries', 'Correlation Heatmap', 'Top Genres', 'Yearly Trend', 'Content Type Distribution', 'Recommendation System')
    )

    if analysis_option == 'Top Directors':
        top_directors = netflix_df['director'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x=top_directors.values, y=top_directors.index, palette='viridis', ax=ax)
        ax.set_title('Top 10 Directors on Netflix')
        st.pyplot(fig)

    elif analysis_option == 'Word Cloud':
        text = ' '.join(netflix_df['description'])
        wordcloud = WordCloud(background_color='black', width=800, height=400).generate(text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud of Netflix Descriptions')
        st.pyplot(fig)

    elif analysis_option == 'Content Ratings':
        fig, ax = plt.subplots(figsize=(12,6))
        sns.countplot(data=netflix_df, x='rating', order=netflix_df['rating'].value_counts().index, palette='magma', ax=ax)
        ax.set_title('Distribution of Content Ratings')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif analysis_option == 'Frequent Actors':
        actors = netflix_df['cast'].str.split(', ').explode()
        top_actors = actors.value_counts().head(10)
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x=top_actors.values, y=top_actors.index, palette='coolwarm', ax=ax)
        ax.set_title('Top 10 Most Frequent Actors')
        st.pyplot(fig)

    elif analysis_option == 'Duration Comparison':
        netflix_df['duration_num'] = netflix_df['duration'].str.extract('(d+)').astype(float)
        fig, ax = plt.subplots(figsize=(12,6))
        sns.boxplot(data=netflix_df, x='type', y='duration_num', palette='Set2', ax=ax)
        ax.set_title('Duration Comparison: Movies vs TV Shows')
        st.pyplot(fig)

    elif analysis_option == 'Monthly Additions':
        monthly_additions = netflix_df['date_added'].dt.to_period('M').value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(14,6))
        monthly_additions.plot(kind='line', color='skyblue', ax=ax)
        ax.set_title('Netflix Content Additions Over Time')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif analysis_option == 'Top Countries':
        country_counts = netflix_df['country'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x=country_counts.values, y=country_counts.index, palette='rocket', ax=ax)
        ax.set_title('Top 10 Countries with Most Content on Netflix')
        st.pyplot(fig)

    elif analysis_option == 'Correlation Heatmap':
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(netflix_df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)

    elif analysis_option == 'Top Genres':
        top_genres = netflix_df['listed_in'].str.split(', ').explode().value_counts().head(10)
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x=top_genres.values, y=top_genres.index, palette='pastel', ax=ax)
        ax.set_title('Top 10 Genres on Netflix')
        st.pyplot(fig)

    elif analysis_option == 'Yearly Trend':
        netflix_df['release_year'] = pd.to_datetime(netflix_df['date_added']).dt.year
        release_trend = netflix_df['release_year'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(release_trend.index, release_trend.values, marker='o', linestyle='-', color='coral')
        ax.set_title('Content Released Over the Years')
        st.pyplot(fig)

    elif analysis_option == 'Content Type Distribution':
        type_counts = netflix_df['type'].value_counts()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x=type_counts.index, y=type_counts.values, palette='Set1', ax=ax)
        ax.set_title('Content Type Distribution')
        st.pyplot(fig)

    # Recommendation System
    elif analysis_option == 'Recommendation System':
        st.subheader('Netflix Content Recommendation System')
        title_input = st.text_input('Enter a title to get recommendations:', 'Breaking Bad')

        # Combine features for better recommendations
        netflix_df['combined_features'] = netflix_df['description'] + ' ' + netflix_df['cast'].fillna('') + ' ' + netflix_df['director'].fillna('') + ' ' + netflix_df['listed_in'].fillna('')

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(netflix_df['combined_features'])

        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(netflix_df.index, index=netflix_df['title']).drop_duplicates()

        def get_recommendations(title, cosine_sim=cosine_sim):
            idx = indices.get(title)
            if idx is None:
                return "Title not found in the dataset. Please try another."
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6]  # Get top 5 recommendations
            movie_indices = [i[0] for i in sim_scores]
            recommended_titles = netflix_df['title'].iloc[movie_indices]
            recommended_genres = netflix_df['listed_in'].iloc[movie_indices]
            recommended_descriptions = netflix_df['description'].iloc[movie_indices]
            
            recommendations = pd.DataFrame({
                'Title': recommended_titles,
                'Genre': recommended_genres,
                'Description': recommended_descriptions
            })
            return recommendations

        if st.button('Get Recommendations'):
            recommendations = get_recommendations(title_input)
            if isinstance(recommendations, str):
                st.write(recommendations)
            else:
                st.write(recommendations)

    # Add download option
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Download Data')
    csv = netflix_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download cleaned data as CSV",
        data=csv,
        file_name='cleaned_netflix_data.csv',
        mime='text/csv',
    )

else:
    st.warning('Please upload a CSV file to proceed.')
