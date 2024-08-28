import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import re
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

class ArticleDataAnalyzer:
    def __init__(self, df):
        self.df = df
    
    def format_datetime(self):
        self.df['date'] = pd.to_datetime(self.df['date'], format = "ISO8601", utc = True)
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['dayofweek'] = self.df['date'].dt.dayofweek 
        self.df['hour'] = self.df['date'].dt.hour

    def set_datetime_index(self):
        if self.df.index.name != 'date':
            self.df.set_index('date', inplace=True)

    def analyzing_headline(self):
        self.df['headline_length'] = self.df['headline'].apply(len)
    
    def publisher_analysis(self):
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
        nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        self.df['sentiment'] = self.df['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])
        self.df['sentiment_class'] = self.df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')
        plt.figure(figsize=(8, 6))
        sns.countplot(x='sentiment_class', data=self.df, palette='coolwarm')
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()

        sentiment_by_publisher = self.df.groupby('publisher')['sentiment'].mean().sort_values(ascending=False).head(30)

        # Plot sentiment by top publishers
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sentiment_by_publisher.values, y=sentiment_by_publisher.index, palette='coolwarm')
        plt.title('Average Sentiment Score by Top Publishers')
        plt.xlabel('Average Sentiment')
        plt.ylabel('Publisher')
        plt.show()

        publisher_sentiment = self.df.groupby('publisher').agg(
        article_count=('headline', 'count'),
        avg_sentiment=('sentiment', 'mean')).reset_index()

        # Sort by the number of articles
        publisher_sentiment = publisher_sentiment.sort_values(by='article_count', ascending=False).head(20)

        # Plot
        fig, ax1 = plt.subplots(figsize=(14, 8))
        # Bar plot for the number of articles
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
        all_headlines = ' '.join(self.df['headline'].tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_headlines)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Headlines')
        plt.show()


    def articles_published(self):
        # Articles published per year
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='year', palette='viridis')
        plt.title('Number of Articles Published Per Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Articles')
        plt.show()

        # Articles published per month
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='month', palette='viridis')
        plt.title('Number of Articles Published Per Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')
        plt.show()

        # Articles published per week
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='dayofweek', palette='viridis')
        plt.title('Number of Articles Published Per Week')
        plt.xlabel('Days (0 -6)')
        plt.ylabel('Number of Articles')
        plt.show()

        # Articles published per hour
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='hour', palette='viridis')
        plt.title('Number of Articles Published Per Hour of the Day')
        plt.xlabel('Hour of the Day (0-23)')
        plt.ylabel('Number of Articles')
        plt.show()

    def daily_article_trend(self):
        self.set_datetime_index()
        daily_article_count = self.df.resample('D').size()
        plt.figure(figsize=(14, 7))
        daily_article_count.plot()
        plt.title('Daily Article Publication Trend')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.show()

    def specific_market_event(self, start_period, end_period):
        # Zoom into a specific period (e.g., around a known market event)
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


    def publication_frequency(self):
        # Resample data by day to count the number of articles per day
        self.set_datetime_index()
        daily_articles = self.df.resample('D').size()

        # Plot the publication frequency
        plt.figure(figsize=(12, 6))
        daily_articles.plot()
        plt.title('Number of Articles Published Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.show()

    def extract_domain(self):
        # Extract domain from publisher email (if applicable)
        self.df['domain'] = self.df['publisher'].apply(lambda x: x.split('@')[-1] if '@' in x else 'Other')

        # Count the number of articles per domain
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
        nltk.download('stopwords')
        nltk.download('wordnet')
        # Tokenize and clean text
        text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
        text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove everything except letters
        tokens = text.lower().split()
        
        # Remove stop words and lemmatize
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)

    def topic_modeling(self, num_topics=5, num_words=10):
        # Preprocess the headlines
        self.df['processed_headlines'] = self.df['headline'].apply(self.preprocess_text)
        
        # Create a CountVectorizer and LDA model
        vectorizer = CountVectorizer()
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=100)

        # Pipeline to process text and apply LDA
        pipeline = make_pipeline(vectorizer, lda_model)
        pipeline.fit(self.df['processed_headlines'])
        
        # Print the topics
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda_model.components_):
            print(f'Topic {topic_idx}:')
            print(' '.join([feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]))
        
        return lda_model, vectorizer

    def visualize_topics(self, lda_model, vectorizer):
        # Prepare the LDA visualization using pyLDAvis
        lda_display = pyLDAvis.sklearn.prepare(lda_model, vectorizer.transform(self.df['processed_headlines']), vectorizer)
        pyLDAvis.display(lda_display)
        # If running in a Jupyter notebook, you can use this instead:
        # pyLDAvis.enable_notebook()
        # pyLDAvis.display(lda_display)

    def plot_topic_word_distribution(self, lda_model, vectorizer, num_topics=5, num_words=10):
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