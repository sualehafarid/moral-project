import pandas as pd
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import os

# TODO: Input your own cache directory
cache_dir = "F:/huggingface_cache"

# Initialize the SeamlessM4T model and processor
model_name = "facebook/seamless-m4t-v2-large"
processor = AutoProcessor.from_pretrained(model_name, use_fast=False, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)

# Load the CSV file
file_path = 'cm_sample_combined.csv'
data = pd.read_csv(file_path)

# Define target languages with SeamlessM4T-supported `tgt_lang` codes
target_languages = {
    'Chinese': 'cmn',      # Simplified Chinese
    'Urdu': 'urd',
    'Hindi': 'hin',
    'Spanish': 'spa',
    'German': 'deu'
}

# Add columns for each target language
for language in target_languages:
    data[language] = None

# Function to translate text using the `tgt_lang` argument
def translate_text(text, tgt_lang, verbose=False):
    if verbose:
        print(f"Translating to {tgt_lang}...")

    inputs = processor(text, return_tensors="pt")
    generated_tokens = model.generate(**inputs, tgt_lang=tgt_lang)
    translated_text = processor.decode(generated_tokens[0], skip_special_tokens=True)
    
    if verbose:
        print(f"Translation to {tgt_lang} complete.")
    
    return translated_text

# Translate each row in the 'combined_input' column for each target language
verbose = True  # Set verbose mode to True to see progress
for index, row in data.iterrows():
    original_text = row['input']
    if verbose:
        print(f"\nTranslating row {index + 1}/{len(data)}: '{original_text}'")

    for language, lang_code in target_languages.items():
        translated_text = translate_text(original_text, lang_code, verbose=verbose)
        data.at[index, language] = translated_text

# Save the translated dataset to a new CSV file
output_file_path = 'cm_sample_combined_seamlessM4T_result.csv'
data.to_csv(output_file_path, index=False)

print(f"\nTranslation complete. Results saved to {output_file_path}")
