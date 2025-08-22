import pandas as pd
from google.cloud import translate_v2 as translate
import os

# NOTE: You need to setup Google Translation API before running the script

# TODO: Input your own Credential File
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "F:/Research_Shivani/neon-circle-431814-i0-fa12f6567ed2.json"

# Initialize Google Translate client
translate_client = translate.Client()

# Load the CSV file
file_path = 'cm_sample_combined.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Ensure there is a column named 'text' to translate; adjust this if the column name is different
text_column = 'input'  # Replace with the actual column name containing text for translation

# Define target languages
target_languages = {
    'Chinese': 'zh',
    'Urdu': 'ur',
    'Hindi': 'hi',
    'Spanish': 'es',
    'German': 'de'
}

# Add columns for each target language
for language in target_languages:
    data[language] = None  # Create a new column for each target language

# Function to call Google Translate API for translation
def translate_text(text, target_lang):
    result = translate_client.translate(text, target_language=target_lang)
    return result['translatedText']

# Translate each row in the text column for each target language
for index, row in data.iterrows():
    original_text = row[text_column]
    for language, lang_code in target_languages.items():
        translated_text = translate_text(original_text, lang_code)
        data.at[index, language] = translated_text

# Generate output file name based on input file name
base_name = os.path.splitext(file_path)[0]
output_file_path = f"{base_name}_result.csv"

# Save the translated dataset to a new CSV file
data.to_csv(output_file_path, index=False)

print(f"Translation complete. Saved to {output_file_path}")
