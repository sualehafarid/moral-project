# llama_3.1_translation.py

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# NOTE: You need to get access from meta to access llama on huggingface

# TODO: Input your own cache directory
cache_dir = "F:/huggingface_cache"

# Initialize LLama 3.1 (6B) model and tokenizer
model_name = "meta-llama/Llama-3.1-8B"
# model_name = "meta-llama/Llama-3.1-70B"  # Replace with the actual model path if available
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

# Load the CSV file
file_path = 'cm_sample_combined.csv'  # Replace with your file path
data = pd.read_csv(file_path)

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

# Function to translate text using LLama 3.1 (6B) with verbose option
def translate_text(text, target_lang, verbose=False):
    if verbose:
        print(f"Translating to {target_lang}...")

    inputs = tokenizer(f"Translate to {target_lang}: {text}", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if verbose:
        print(f"Translation to {target_lang} complete.")
    
    return translated_text

# Translate each row in the text column for each target language
verbose = True  # Set verbose mode to True to see progress
for index, row in data.iterrows():
    original_text = row['input']
    if verbose:
        print(f"\nTranslating row {index + 1}/{len(data)}: '{original_text}'")

    for language, lang_code in target_languages.items():
        translated_text = translate_text(original_text, lang_code, verbose=verbose)
        data.at[index, language] = translated_text

# Save to CSV
# output_file_path = 'cm_sample_combined_llama3.1_result.csv'
output_file_path = 'cm_sample_combined_llama3.1_8B_result.csv'
data.to_csv(output_file_path, index=False)

print(f"\nTranslation complete. Results saved to {output_file_path}")
