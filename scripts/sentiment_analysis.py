import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis.lda_model
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class ArticleDataAnalyzer:
    def __init__(self, df):
        # Initialize with the dataset, stop words, and lemmatizer
        self.df = df
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    def ensure_nltk_resources(self):
        # Ensure necessary NLTK resources are available
        resources = [
            'corpora/stopwords.zip',
            'corpora/wordnet.zip',
            'sentiment/vader_lexicon.zip'
        ]
        
        for resource in resources:
            try:
                nltk.data.find(resource)
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource.split('/')[1].split('.')[0])


    def format_datetime(self):
        # Convert 'date' to datetime and extract date components
        self.df['date'] = pd.to_datetime(self.df['date'], format = "ISO8601")
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['dayofweek'] = self.df['date'].dt.dayofweek 
        self.df['hour'] = self.df['date'].dt.hour

    def set_datetime_index(self):
        # Set 'date' as the DataFrame index if not already
        if self.df.index.name != 'date':
            self.df.set_index('date', inplace=True)

    def analyzing_headline(self):
        # Calculate the length of each headline
        self.df['headline_length'] = self.df['headline'].apply(len)
    
    def publisher_analysis(self):
        # Analyze and visualize the number of articles per publisher
        publisher_counts = self.df['publisher'].value_counts()
        top_publishers = publisher_counts.head(30)
        plt.figure(figsize = (10, 6))
        sns.barplot(x= top_publishers.index, y = top_publishers.values, palette =  'viridis')
        plt.title("Number of Articles Per Publisher")
        plt.xlabel('Publisher')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation = 90)
        plt.show()

    def sentiment_analysis(self):
        # Perform sentiment analysis on headlines
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        self.df['sentiment'] = self.df['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

        # Define thresholds for sentiment classification
        def classify_sentiment(score):
            if score >= 0.7:
                return 'highly positive'
            elif score > 0:
                return 'positive'
            elif score == 0:
                return 'neutral'
            elif score <= -0.7:
                return 'highly negative'
            else:
                return 'negative'

        # Apply the classification to the sentiment scores
        self.df['sentiment_class'] = self.df['sentiment'].apply(classify_sentiment)

        # Plot sentiment distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x='sentiment_class', data=self.df, palette='coolwarm')
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()

        # Print sentiment counts
        sentiment_counts = self.df['sentiment_class'].value_counts()
        print("Sentiment Counts:")
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment.capitalize()}: {count}")

        # Analyze sentiment by publisher    
        sentiment_by_publisher = self.df.groupby('publisher')['sentiment'].mean().sort_values(ascending=False).head(30)

        # Plot average sentiment score by top publishers
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sentiment_by_publisher.values, y=sentiment_by_publisher.index, palette='coolwarm')
        plt.title('Average Sentiment Score by Top Publishers')
        plt.xlabel('Average Sentiment')
        plt.ylabel('Publisher')
        plt.show()

        # Combined plot for article count and average sentiment by publisher
        publisher_sentiment = self.df.groupby('publisher').agg(
        article_count=('headline', 'count'),
        avg_sentiment=('sentiment', 'mean')).reset_index()

        # Sort by the number of articles
        publisher_sentiment = publisher_sentiment.sort_values(by='article_count', ascending=False).head(20)

        # Plot article count and sentiment score
        fig, ax1 = plt.subplots(figsize=(14, 8))
        sns.barplot(x='publisher', y='article_count', data=publisher_sentiment, ax=ax1, palette='Blues_d')
        ax2 = ax1.twinx()
        sns.lineplot(x='publisher', y='avg_sentiment', data=publisher_sentiment, ax=ax2, color='red', marker='o')
        ax1.set_title('Number of Articles and Average Sentiment Score per Publisher')
        ax1.set_xlabel('Publisher')
        ax1.set_ylabel('Number of Articles')
        ax2.set_ylabel('Average Sentiment Score')
        ax1.set_xticklabels(publisher_sentiment['publisher'], rotation=90)
        plt.show()
 
    def headline_analysis(self):
        # Generate and visualize a word cloud of headlines
        all_headlines = ' '.join(self.df['headline'].tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_headlines)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Headlines')
        plt.show()

    def articles_published(self):
        # Visualize articles published per year
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='year', palette='viridis')
        plt.title('Number of Articles Published Per Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Articles')
        plt.show()

        # Visualize articles published per month
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='month', palette='viridis')
        plt.title('Number of Articles Published Per Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')
        plt.show()

        # Visualize articles published per week
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='dayofweek', palette='viridis')
        plt.title('Number of Articles Published Per Week')
        plt.xlabel('Days (0 -6)')
        plt.ylabel('Number of Articles')
        plt.show()

        # Visualize articles published per hour
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='hour', palette='viridis')
        plt.title('Number of Articles Published Per Hour of the Day')
        plt.xlabel('Hour of the Day (0-23)')
        plt.ylabel('Number of Articles')
        plt.show()

    def daily_article_trend(self):
        # Analyze the daily trend of article publications
        self.set_datetime_index()
        daily_article_count = self.df.resample('D').size()
        plt.figure(figsize=(14, 7))
        daily_article_count.plot()
        plt.title('Daily Article Publication Trend')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.show()

    def specific_market_event(self, start_period, end_period):
        # Analyze article publication spikes around specific market events
        self.set_datetime_index()
        daily_article_count = self.df.resample('D').size()
        spike_period = daily_article_count[start_period: end_period]  # Example period

        if not spike_period.empty:
            plt.figure(figsize=(14, 7))
            spike_period.plot()
            plt.title(f'Article Publication Spike Around Specific Market Event from {start_period} to {end_period}')
            plt.xlabel('Date')
            plt.ylabel('Number of Articles')
            plt.show()
        else:
            print("No data available for the specified period.")

    def extract_domain(self):
        # Extract and analyze domain from publisher email addresses
        self.df['domain'] = self.df['publisher'].apply(lambda x: x.split('@')[-1] if '@' in x else 'Other')

        # Count and visualize the number of articles per domain
        domain_counts = self.df['domain'].value_counts()
        print(domain_counts)

        # Plot the distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=domain_counts.index, y=domain_counts.values, palette='magma')
        plt.title('Number of Articles per Domain')
        plt.xlabel('Domain')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.show()

    def preprocess_text(self, text):
        # Clean and preprocess text by tokenizing, removing stop words, and lemmatizing
        text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
        text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Remove everything except letters and spaces
        tokens = text.lower().split()

        # Remove stop words and lemmatize tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)

    def topic_modeling(self, num_topics=5, num_words=10, sample_size=None):
        # Sample a portion of the data if sample_size is specified
        if sample_size:
            df_sample = self.df.sample(n=sample_size, random_state=100)
        else:
            df_sample = self.df

        # Create a CountVectorizer and LDA model
        vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=100, n_jobs=-1)  # Use all CPUs

        # Define a custom preprocessor function
        def preprocess_pipeline(texts):
            return [self.preprocess_text(text) for text in texts]

        # Use a pipeline to process text and apply LDA
        pipeline = Pipeline([
            ('preprocessor', FunctionTransformer(lambda x: preprocess_pipeline(x), validate=False)),
            ('vectorizer', vectorizer),
            ('lda', lda_model)
        ])

        # Fit the pipeline
        pipeline.fit(df_sample['headline'])

        # Extract the LDA model and feature names
        feature_names = vectorizer.get_feature_names_out()
        lda_model = pipeline.named_steps['lda']

        # Print the topics
        for topic_idx, topic in enumerate(lda_model.components_):
            print(f'Topic {topic_idx}:')
            print(' '.join([feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]))

        return lda_model, vectorizer

    def visualize_topics(self, lda_model, vectorizer):
        # Ensure the data is transformed outside of any multiprocessing context
        transformed_data = vectorizer.transform(self.df['headline'].apply(self.preprocess_text))

        # Prepare the LDA visualization using pyLDAvis
        lda_display = pyLDAvis.prepare(
            topic_term_dists=lda_model.components_,
            doc_topic_dists=lda_model.transform(transformed_data),
            doc_lengths=[len(doc.split()) for doc in self.df['headline']],
            vocab=vectorizer.get_feature_names_out(),
            term_frequency=transformed_data.sum(axis=0).A1
        )
        pyLDAvis.display(lda_display)

    def plot_topic_word_distribution(self, lda_model, vectorizer, num_topics=5, num_words=10):
        # Visualize the topic word distribution
        feature_names = vectorizer.get_feature_names_out()
        for i in range(num_topics):
            plt.figure(figsize=(10, 6))
            top_words = [feature_names[j] for j in lda_model.components_[i].argsort()[:-num_words - 1:-1]]
            top_weights = lda_model.components_[i][lda_model.components_[i].argsort()[:-num_words - 1:-1]]
            plt.barh(top_words, top_weights)
            plt.xlabel('Word Weight')
            plt.title(f'Topic {i+1} Word Distribution')
            plt.gca().invert_yaxis()
            plt.show()