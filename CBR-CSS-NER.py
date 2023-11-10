#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip uninstall fitz
get_ipython().system('pip install pymupdf')
get_ipython().system('pip install frontend')
get_ipython().system('pip install unidecode')
get_ipython().system('pip install nltk')
get_ipython().system('pip install geonamescache')
get_ipython().system('pip install wikipedia')
get_ipython().system('pip install pycountry')
get_ipython().system('pip install pandas')
get_ipython().system('pip install spacy')
get_ipython().system('pip install tqdm')
get_ipython().system('pip install scikit-learn')
import os
import re
import spacy
import pandas as pd
from unidecode import unidecode
from multiprocessing import Pool
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from collections import Counter
import pycountry


# In[2]:


# Download comprehensive list of countries and country subdivisions from pycountry
import json
import re
import ast

results = []

for country in pycountry.countries:
    # Check if the country has an official_name attribute
        results.append(
            country.name
        )

#print(results)

# Send it to our json file
with open ("C:\\Users\\Nick\\Downloads\\Articles (1)\\data\\country_entities.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)


# In[3]:


# From NAMED ENTITY RECOGNITION SERIES, Lesson 02 Rules-Based NER by Dr. W.J.B. Mattingly
# Custom NER model functions
import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import json
import os

def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_multiple_json_files(folder_path):
    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            entity_type = os.path.splitext(filename)[0]  # Extract entity type from file name
            data[entity_type] = load_data(file_path)
    return data

def training_data(folder_path):
    data = load_multiple_json_files(folder_path)
    patterns = []
    for entity_type, entity_patterns in data.items():
        for pattern in entity_patterns:
            patterns.append({
                "label": entity_type,
                "pattern": pattern
            })
    return patterns

def generate_rules(patterns):
    nlp = English()
    ruler = EntityRuler(nlp)
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    nlp.to_disk("C:\\Users\\Nick\\Downloads\\Articles (1)\\cbr_css_ner")

def test_model(model, text):
    doc = nlp(text)
    results = []
    for ent in doc.ents:
        results.append(ent.text)
    return results

# Load training data from JSON files in the "data" folder, only need to run once
#patterns = training_data("C:\\Users\\Nick\\Downloads\\Articles (1)\\data")
#generate_rules(patterns)

def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# In[61]:


import os
import pandas as pd
import fitz  # PyMuPDF
import spacy
import re
from unidecode import unidecode

pattern = [r'\d+:\d+', r'\d-\d', r'\d\(\d\)', r"\(\d{4}\)", r"\d{1,4}–\d{1,4}", r"\d+\(\d+–\d+\):"]
# Removes any sentence with digits seoarated by a colon or dash 499-501 or 499:501

# Load your custom NLP model only once
nlp = spacy.load("C:\\Users\\Nick\\Downloads\\Articles (1)\\cbr_css_ner")

# Add the sentencizer component to the pipeline if not already added
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Folder where articles are held
folder_path = r"C:\\Users\\Nick\\Downloads\\Articles (1)\\Articles"

# List all PDF files in the folder
pdf_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".pdf")]

def should_ignore_paragraph(paragraph, ignore_patterns=None):
    # Check if the paragraph has fewer than 60 characters
    if len(paragraph) < 60:
        return None  # Indicate to ignore

    # Define keywords or entities that indicate sections to ignore
    ignore_keywords = ["abstract", "a b s t r a c t", "bibliography", "[mesh]," "objectives:", "ethics:", "declarations",
                       "acknowledgements", "references", "works cited"]

    # Check if any of the ignore keywords are present in the paragraph
    if any(keyword in paragraph.lower() for keyword in ignore_keywords):
        return None  # Indicate to ignore

    # Check if any of the ignore patterns match the paragraph using regular expressions
    if ignore_patterns and any(re.search(pattern, paragraph) for pattern in ignore_patterns):
        return None  # Indicate to ignore

    # If none of the above conditions are met, do not ignore the paragraph
    return paragraph  # Indicate not to ignore

def should_ignore_sentence_portion(sentence_portion, ignore_patterns=None):
    # Apply your sentence-level ignore logic here
    # For example, you can check for specific keywords, patterns, or conditions
    
    # Check if the sentence portion has fewer than 60 characters
    if len(sentence_portion) < 60:
        return None  # Indicate to ignore

    # Process the paragraph using spaCy
    doc = nlp(sentence_portion)

    # Define keywords or entities that indicate sections to ignore
    ignore_keywords = ["doi", "journal", "https:", "xxx", "xxxx", "©", "conference of", "jordan et al", "& jordan", "jordan," "conference:", "conference.", "society:"
                       "(ed.)", "sage publications", "http:", "reference number:", "phd", "PO Box", "email:", "ph.d", "Elsevier", "pp."]

    # Check if any of the ignore keywords are present in the paragraph
    if any(keyword in sentence_portion.lower() for keyword in ignore_keywords):
        return None  # Indicate to ignore

    # Check if the paragraph contains author name entities
    if any(ent.label_ == "author_entities" for ent in doc.ents):
        return None  # Indicate to ignore

    # Check if any of the ignore patterns match the sentence portion using regular expressions
    if ignore_patterns and any(re.search(pattern, sentence_portion) for pattern in ignore_patterns):
        return None  # Indicate to ignore

    # Replace the logic above with your specific conditions
    return sentence_portion  # Default to not ignoring

def extract_information(text, entity_category):
    if text is None:
        return []  # Return an empty list if the input text is None

    doc = nlp(text)
    return [(ent.text, ent.sent.text) for ent in doc.ents if ent.label_ == entity_category]
    
def process_pdf(file_path):
    text_data = []

    try:
        doc = fitz.open(file_path)
    except fitz.EmptyFileError:
        # Handle empty document
        text_data.append((file_path, 0, "Document is empty"))  # Include page_num as 0 for empty documents
        return None

    # Get the total number of pages in the PDF
    total_pages = len(doc)

    for page_num, page in enumerate(doc):
        # Skip processing the last page
        if page_num == total_pages - 1:
            continue

        page_text = page.get_text()

        if not page_text.strip():
            # Handle empty page
            text_data.append((file_path, page_num, "Page is empty", "", "", ""))
            continue

        # Split the page's text into paragraphs
        paragraphs = page_text.split('\n\n')  # You can adjust the separator as needed

        concatenated_paragraph = ""
        concatenated_sentence = ""

        for paragraph in paragraphs:
            # Apply your paragraph-level ignore logic here
            ignored_paragraph = should_ignore_paragraph(paragraph, ignore_patterns=None)

            if ignored_paragraph is None:
                continue

            # Split the ignored paragraph into sentences using spaCy
            sentences = [sent.text for sent in nlp(ignored_paragraph).sents]
            for sentence in sentences:
                # Apply your sentence-level ignore logic here
                ignored_sentence = should_ignore_sentence_portion(sentence, ignore_patterns=pattern)

                if ignored_sentence is None:
                    continue

                text_data.append((file_path, page_num, paragraph, ignored_paragraph, sentence, ignored_sentence))

    # Create a DataFrame for the PDF
    # file_name = name of file
    # page_num = page number
    # original_paragraph = Extracted paragraph before filtering
    # kept_paragraph = If kept, return original paragraph, if not kept, return blank.
    # original_sentence = Extracted sentence before filtering
    # text = If kept, return sentence, if not kept, return blank.
    text_df = pd.DataFrame(text_data, columns=['file_name', 'page_num', 'original_paragraph', 'kept_paragraph', 'original_sentence', 'text'])

    # Process NER using spaCy for entity categories
    entity_categories = ["discipline_entities", "country_entities", "techniques_entities"]
    processed_dfs = []

    for category in entity_categories:
        entity_info = text_df.apply(lambda row: extract_information(row['text'], category), axis=1)

        # Flatten the list of tuples using .values and .tolist()
        entity_info = [item for sublist in entity_info.values.tolist() for item in sublist]

        entity_df = pd.DataFrame(entity_info, columns=['entity_word', 'triggering_sentence'])
        entity_df['entity_category'] = category
        entity_df['file_name'] = text_df['file_name']  # Add 'file_name' column
        entity_df['page_num'] = text_df['page_num']  # Add 'file_name' column

        processed_dfs.append(entity_df)

    # If no entities were found, create a "no_entities" category with a placeholder text
    found_entities = any(len(df) > 0 for df in processed_dfs)
    if not found_entities:
        no_entities_df = pd.DataFrame({'entity_word': None, 'triggering_sentence': 'No text available', 'entity_category': 'no_entities', 'file_name': text_df['file_name']})
        processed_dfs.append(no_entities_df)

    # Concatenate the processed DataFrames for all categories
    result_df = pd.concat(processed_dfs, ignore_index=True)

    return result_df

# Initialize an empty list to store the results
result_dfs = []

# Process PDF files one at a time
for pdf_file in pdf_files:
    pdf_result = process_pdf(pdf_file)
    if pdf_result is not None:
        result_dfs.append(pdf_result)

# Concatenate the results for all PDF files
final_result_df = pd.concat(result_dfs, ignore_index=True)


# In[63]:


final_result_df1 = final_result_df
# Replace NA with page_number = 0
final_result_df1['page_num'].fillna(0, inplace=True)

# Calculate the median page number for each file_name
median_page_numbers = final_result_df1.groupby('file_name')['page_num'].median().reset_index()
median_page_numbers.rename(columns={'page_num': 'median_page_number'}, inplace=True)

final_result_df1 = final_result_df1.merge(median_page_numbers, on='file_name', how='left')

# Add a new column that subtracts page_num from median_page_number
final_result_df1['page_num_difference'] = final_result_df1['median_page_number'] - final_result_df1['page_num']
final_result_df1['page_num_difference_perc'] = 1/(final_result_df1['page_num_difference'].abs()+1)

final_result_df1


# In[59]:


# Check entity words

mgmt_df = final_result_df[final_result_df['entity_word'] == 'Jordan']

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    pd.options.display.max_colwidth = 500
    display(mgmt_df)


# In[64]:


# Raw frequency of appearnace

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(final_result_df)

# Replace the incorrect year "1902" with "2013" in the "file_name" column
df['file_name'] = df['file_name'].str.replace(r'-(1902)\.pdf', '-2013.pdf')

# Extract the year from the "file_name" column
df['year'] = df['file_name'].str.extract(r'-(\d{4})\.pdf').astype(int)

# Convert the year to a decade
def year_to_decade(year):
    return f"{year // 10 * 10}-{(year // 10 * 10) + 9}"

df['decade'] = df['year'].apply(year_to_decade)

# Filter out decades from 2020 to 2029
df = df[~df['decade'].between('2020-2029', '2020-2029')]

# Create a frequency table of "entity_word" grouped by "entity_category" and "decade"
frequency_table = df.groupby(['entity_category', 'decade', 'entity_word']).size().reset_index(name='frequency')

frequency_table.to_csv('C:\\Users\\Nick\\Downloads\\Articles (1)\\raw_freq_by_decade.csv', index=False)
frequency_table

#plt.hist(df.decade)


# In[65]:


import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(final_result_df1)

# Group by entity_category, file_name, and entity_word, and count occurrences
top3_entities = df.groupby(['entity_category', 'file_name', 'page_num', 'entity_word'])['entity_word'].count().reset_index(name='entity_count')

# Merge with the absolute page_num_difference
top3_entities = top3_entities.merge(df[['file_name', 'entity_word', 'page_num', 'page_num_difference_perc']], on=['file_name', 'page_num','entity_word'], how='left')

# Calculate the weighted entity_count
top3_entities['weighted_entity_count'] = top3_entities['entity_count'] * top3_entities['page_num_difference_perc']

# Remove duplicate rows
top3_entities = top3_entities.drop_duplicates()

# Group by entity_category, file_name, and sum the entity_count
top3_entities_sum = top3_entities.groupby(['entity_category', 'file_name', 'entity_word'])['entity_count'].sum().reset_index(name='sum_entity_count')

# Group by entity_category, file_name, entity_word, and sum the weighted_entity_count
top3_entities_wsum = top3_entities.groupby(['entity_category', 'file_name', 'entity_word'])['weighted_entity_count'].sum().reset_index(name='sum_weighted_entity_count')

# Group by entity_category, file_name, and average the page number difference
top3_entities_pgnum = top3_entities.groupby(['entity_category', 'file_name', 'entity_word'])['page_num_difference_perc'].mean().reset_index(name='avg_distance_from_median_page')

# Merge the sum_entity_count back to top3_entities
top3_entities = top3_entities_wsum.merge(top3_entities_sum, on=['entity_category', 'file_name', 'entity_word'], how='left')

# Replace the incorrect year "1902" with "2013" in the "file_name" column
top3_entities['file_name'] = top3_entities['file_name'].str.replace(r'-(1902)\.pdf', '-2013.pdf')

# Extract year from file_name
top3_entities['year'] = top3_entities['file_name'].str.extract(r'-(\d{4})\.pdf', expand=False).astype(int)

# Formula to convert the year to a decade
def year_to_decade(year):
    return f"{year // 10 * 10}-{(year // 10 * 10) + 9}"

top3_entities['decade'] = top3_entities['year'].apply(year_to_decade)

# Filter out decades from 2020 to 2029
frequency_table = frequency_table[~frequency_table['decade'].between('2020-2029', '2020-2029')]

# Merge the sum_entity_count back to top3_entities
top3_entities = top3_entities.merge(top3_entities_pgnum, on=['entity_category', 'file_name', 'entity_word'], how='left')

# Sort by entity_category, file_name, and entity_count in descending order
top3_entities = top3_entities.sort_values(by=['entity_category', 'file_name', 'sum_weighted_entity_count'], ascending=[True, True, False])

# Add a rank column within each group
top3_entities['rank'] = top3_entities.groupby(['entity_category', 'file_name'])['sum_weighted_entity_count'].rank(method='min', ascending=False)

# Filter for the top 5 entities within each group
top3_entities = top3_entities[top3_entities['rank'] <= 3]

# Filter out entities with sum_entity_count equal to 1 and rank greater than
top3_entities = top3_entities.drop(top3_entities[(top3_entities['sum_entity_count'] == 1) & (top3_entities['rank'] > 1)].index)

top3_entities
top3_entities.to_csv('C:\\Users\\Nick\\Downloads\\Articles (1)\\top3_entities_per_article_page_num.csv', index=False)


# In[66]:


# Create a frequency table grouped by entity_category, decade, and entity_word
frequency_table = top3_entities.groupby(['entity_category', 'decade', 'entity_word'])['sum_entity_count'].sum().reset_index()

# Display the frequency table
frequency_table.to_csv('C:\\Users\\Nick\\Downloads\\Articles (1)\\by_decade.csv', index=False)


# In[ ]:


# Term Frequency - Inverse Document Frequency Analysis


# In[23]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Function to extract information using TF-IDF
def extract_information(result_dfs):
    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b')

    # Create an empty list to store the tidy data
    tidy_data = []

    # Iterate through the list of nested DataFrames
    for result_df in result_dfs:
        # Get unique file names and entity categories
        unique_file_names = result_df['file_name'].unique()
        unique_entity_categories = result_df['entity_category'].unique()

        # Process data row by row
        for entity_category in tqdm(unique_entity_categories, desc='Entity Categories'):
            for file_name in tqdm(unique_file_names, desc='File Name'):
                # Filter the DataFrame for the current entity_category and file_num
                subset_df = result_df[(result_df['entity_category'] == entity_category) & (result_df['file_name'] == file_name)]

                if not subset_df.empty:
                    # Combine all unique entity words into a single string for TF-IDF
                    unique_entity_words = " ".join(subset_df['entity_word'])

                    # Calculate TF-IDF for the combined entity words
                    tfidf_scores = tfidf_vectorizer.fit_transform([unique_entity_words])

                    # Extract feature names and scores
                    feature_names = tfidf_vectorizer.get_feature_names()
                    tfidf_scores = tfidf_scores.toarray()[0]

                    # Sort and select the top 3 keywords by TF-IDF score (excluding repeating words)
                    top_indices = tfidf_scores.argsort()[-5:][::-1]  # Get top 5 to account for potential repeated words
                    top_keywords = [feature_names[i] for i in top_indices]
                    top_scores = [tfidf_scores[i] for i in top_indices]

                    # Remove repeating words and short keywords from the top keywords
                    top_keywords = [kw for kw in top_keywords if top_keywords.count(kw) == 1 and len(kw) > 2][:5]
                    top_scores = [top_scores[i] for i, kw in enumerate(top_keywords)]

                    # Append the top keywords to the tidy_data list
                    for keyword, score in zip(top_keywords, top_scores):
                        tidy_data.append({
                            'file_name': file_name,
                            'entity_category': entity_category,
                            'entity_word': keyword,
                            'tf_idf_score': score
                        })

    # Create the tidy DataFrame
    tidy_df = pd.DataFrame(tidy_data)

    return tidy_df

# Extract top 3 non-repeating keywords using TF-IDF within each entity category for each file number
tidy_df = extract_information(result_dfs)
tidy_df


# In[72]:


len(pd.unique(tidy_df['file_name']))


# In[24]:


# Filter rows where tf_idf_score > 0
filtered_df = tidy_df[tidy_df['tf_idf_score'] > 0]
df = pd.DataFrame(filtered_df)

# Create a frequency table for entity words
ew_frequency_table = df.groupby(['entity_category', 'entity_word']).size().reset_index(name='frequency')

# Sort in descending order within each category
ew_frequency_table['rank'] = ew_frequency_table.groupby('entity_word')['frequency'].rank(method='max', ascending=False)
ew_frequency_table = ew_frequency_table.sort_values(['entity_word', 'rank'], ascending=[True, True])

# Save the frequency table to a CSV file
ew_frequency_table.to_csv('C:\\Users\\Nick\\Downloads\\Articles (1)\\EW_TF_IDF_frequency_table.csv', index=False)

# Display the frequency table
ew_frequency_table


# In[26]:


## Run TF-IDF on the extracted sentence
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import re

# Function to extract information using TF-IDF for a batch of articles
def extract_information_batch(result_df):
    # Initialize TfidfVectorizer with stop words and token pattern
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', stop_words='english')

    # Create an empty list to store the tidy data for this batch
    tidy_data = []

    unique_file_names = result_df['file_name'].unique()
    unique_entity_categories = result_df['entity_category'].unique()

    # Process data row by row
    for entity_category in tqdm(unique_entity_categories, desc='Entity Categories'):
        for file_name in tqdm(unique_file_names, desc='File Names'):
            # Filter the DataFrame for the current entity_category and file_name
            subset_df = result_df[(result_df['entity_category'] == entity_category) & (result_df['file_name'] == file_name)]
    
            if not subset_df.empty:
                unique_entity_words = " ".join(subset_df['triggering_sentence'])
                unique_entity_words = re.sub(r'\b\d+\b', '', unique_entity_words)
    
                tfidf_scores = tfidf_vectorizer.fit_transform([unique_entity_words])
                feature_names = tfidf_vectorizer.get_feature_names()
                tfidf_scores = tfidf_scores.toarray()[0]
    
                top_keywords = []
                top_scores = []
    
                for i in tfidf_scores.argsort()[::-1]:
                    keyword = feature_names[i]
                    if len(keyword) > 2 and keyword not in top_keywords:
                        top_keywords.append(keyword)
                        top_scores.append(tfidf_scores[i])
                        if len(top_keywords) == 5:
                            break
    
                for keyword, score in zip(top_keywords, top_scores):
                    tidy_data.append({
                        'file_name': file_name,
                        'entity_category': entity_category,
                        'entity_word': keyword,
                        'tf_idf_score': score
                    })

    # Create the tidy DataFrame
    tidy_df = pd.DataFrame(tidy_data)

    return tidy_df
    
final_result_df = pd.concat(result_dfs, ignore_index=True)
# Extract top 3 non-repeating keywords using TF-IDF within each entity category for each file number
tw_df = extract_information_batch(final_result_df)
tw_df


# In[69]:


len(pd.unique(final_result_df['file_name']))


# In[27]:


# Filter rows where tf_idf_score > 0
filtered_df = tw_df[tw_df['tf_idf_score'] > 0]
df = pd.DataFrame(filtered_df)

# Create a frequency table for entity words
tw_frequency_table = df.groupby(['entity_category', 'entity_word']).size().reset_index(name='frequency')

# Sort in descending order within each category
tw_frequency_table['rank'] = tw_frequency_table.groupby('entity_word')['frequency'].rank(method='max', ascending=False)
tw_frequency_table = tw_frequency_table.sort_values(['entity_word', 'rank'], ascending=[True, True])

# Filter out terms with fewer than 5 occurrences
tw_frequency_table = tw_frequency_table[tw_frequency_table['frequency'] >= 2]

# Save the frequency table to a CSV file
tw_frequency_table.to_csv('C:\\Users\\Nick\\Downloads\\Articles (1)\\TW_TF_IDF_frequency_table.csv', index=False)

# Display the frequency table
tw_frequency_table

