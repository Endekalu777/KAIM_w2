import pandas as pd
from textblob   import TextBlob
import os

def load_stock_data(data_folder, news_file, file_keyword='historical'):
    # List all files in the folder
    all_files = os.listdir(data_folder)

    # Filter files that are related to stock prices based on the keyword
    stock_files = [file for file in all_files if file_keyword.lower() in file.lower()]

    # Initialize an empty list to store DataFrames
    all_data = []

    # Read only stock-related files and combine them
    for file in stock_files:
        file_path = os.path.join(data_folder, file)  # Construct full file path
        df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
        all_data.append(df)  # Append to the list

    # Combine all DataFrames into one
    stock_price = pd.concat(all_data, ignore_index=True)

    # Read the news article CSV file
    news_article = pd.read_csv(news_file)

    return stock_price, news_article


def custom_aggregation(news_article, stock_price):
    # Define aggregation functions
    aggregation_functions = {
        'headline': lambda x: ' | '.join(x),  # Concatenate all headlines with ' | ' separator
        'publisher': lambda x: ', '.join(x.unique()),  # List unique publishers separated by commas
        'url': lambda x: ', '.join(x.unique()),  # List unique URLs separated by commas
        # Add other relevant columns with appropriate aggregation functions
        'stock': lambda x: ', '.join(x.unique()),  # Example for a 'stock' column, listing unique values
        # Add more columns if needed with relevant aggregation strategies
    }

    # Aggregate the news_article DataFrame by 'date' using the defined aggregation functions
    news_article_agg = news_article.groupby('date').agg(aggregation_functions).reset_index()

    # Merge stock_price with the aggregated news_article data
    merged_df = pd.merge(stock_price, news_article_agg, left_on='Date', right_on='date', how='inner')

    # Drop the redundant 'date' column
    merged_df.drop(columns=['date'], inplace=True)

    return merged_df


def get_sentiment_score(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity