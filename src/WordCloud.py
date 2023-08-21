# Geneate a word cloud based on review texts called from main.py

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

def generate_wordcloud(text_dataset, display_wc, output_file, mask_path=None, stopwords=None, max_words=1000, background_color="white", width=800, height=400):
    """
    Generate a word cloud from a text dataset.

    Parameters:
        text_dataset (pd.Series or str): The text dataset as a pandas Series or a path to a CSV file.
        display_wc (int, optional): Option to display the wordcloud (plt.show())
        output_file: The file to print the wordcloud to. Default is None (./output/wordcloud.png)
        mask_path (str, optional): Path to the image file used as a mask for the word cloud. Default is None. eg. "../mask.png"
        stopwords (set, optional): Set of stopwords to be removed from the word cloud. Default is None.
        max_words (int, optional): Maximum number of words to be displayed in the word cloud. Default is 2000.
        background_color (str, optional): Background color of the word cloud. Default is "white".
        width (int, optional): Width of the word cloud image in pixels. Default is 800.
        height (int, optional): Height of the word cloud image in pixels. Default is 400.

    Returns:
        WordCloud: The generated word cloud object.
    """
    if isinstance(text_dataset, str):
        # If the input is a file path, read the CSV file
        data_file = pd.read_csv(text_dataset, encoding="utf-8")
        text_data = data_file["text"].to_string()
    elif isinstance(text_dataset, pd.Series):
        # If the input is a pandas Series, convert it to string
        text_data = text_dataset.to_string()
    else:
        raise ValueError("Invalid input. The 'text_dataset' must be either a pandas Series or a CSV file path.")

    if stopwords is None:
        # Use default stopwords from the wordcloud library
        stopwords = set(STOPWORDS)

    if mask_path is not None:
        # Load the mask image if provided
        mask = np.array(Image.open(mask_path))
    else:
        mask = None

    # Create the WordCloud object
    wordcloud = WordCloud(stopwords=stopwords, background_color=background_color, max_words=max_words, mask=mask, width=width, height=height)

    # Generate the word cloud
    wordcloud.generate(text_data)

    # Display the word cloud
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    
    # option to display the wordcloud or not
    if display_wc == 0:
        plt.show()
    
    if output_file is None:
        wordcloud.to_file("./output/wordcloud.png")
    else:
        wordcloud.to_file(output_file)
    return wordcloud

## sample usage
# #   with a pandas Series
# data_file = pd.read_csv("../processed_data/ny_data.csv", encoding="utf-8")
# ## leave only the text column
# data_file = data_file["text"]

# ## with a CSV file path
# data_file_path = "../processed_data/ny_data.csv"

# generate_wordcloud(data_file_path, mask_path=None, stopwords=None, max_words=1000, background_color="white",  width=800, height=800)

