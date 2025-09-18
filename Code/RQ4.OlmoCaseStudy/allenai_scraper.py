import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import os
from bs4 import BeautifulSoup
import csv
from selenium.webdriver.common.action_chains import ActionChains
import random


def setup_driver():
    """Set up and return a configured Firefox WebDriver with proxy rotation."""
    firefox_options = Options()
    # Uncomment the line below if you want to run in headless mode
    firefox_options.add_argument('--headless')

    # Add additional options for better stability
    firefox_options.add_argument('--disable-gpu')
    firefox_options.add_argument('--no-sandbox')
    firefox_options.add_argument('--disable-dev-shm-usage') # randomly select a proxy
    
    firefox_options.set_preference('network.proxy.type', 0)

    service = Service(executable_path="/shared/3/resources/webbrowser-drivers/geckodriver")
    driver = webdriver.Firefox(service=service, options=firefox_options)
    driver.set_window_size(1920, 1080)  # Set a larger window size
    
    return driver

def wait_for_element(driver, by, value, timeout=20):
    """Wait for an element to be present and visible."""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except TimeoutException:
        print(f"Timeout waiting for element: {value}")
        return None

def parse_response_html(html_file):
    """Parse the saved HTML file to extract all response spans and all sources (corpus name, type, and URL)."""
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')

    # Extract all response spans (blockquotes)
    responses = [b.get_text(strip=True) for b in soup.find_all('blockquote')]

    # Extract all sources
    sources = []
    for card in soup.find_all('li', class_='MuiCard-root'):
        # Find the corpus name and HuggingFace dataset URL
        corpus_link = card.find('a', href=True)
        corpus_name = corpus_link.get_text(strip=True) if corpus_link else None
        corpus_url = corpus_link['href'] if corpus_link else None

        # Find the corpus type (e.g., web corpus (DCLM))
        corpus_type = None
        corpus_type_span = card.find('span', class_='MuiTypography-root MuiTypography-body2 css-1je4czj')
        if corpus_type_span:
            corpus_type = corpus_type_span.get_text(strip=True)

        # Try to find the real document URL (not huggingface)
        document_url = None
        for a in card.find_all('a', href=True):
            href = a['href']
            if href.startswith('http') and 'huggingface.co' not in href:
                document_url = href
                break

        # Get the document content from the blockquote
        doc_content = card.find('blockquote')
        doc_text = doc_content.get_text(strip=True) if doc_content else None

        # Only add the source if we have either a real document URL or document content
        if document_url or doc_text:
            sources.append({
                'corpus_name': corpus_name,
                'corpus_type': corpus_type,
                'corpus_url': corpus_url,
                'document_url': document_url,
                'document_content': doc_text
            })

    return {
        'responses': responses,
        'sources': sources
    }

def extract_sources_with_selenium(driver):
    """Extract sources by clicking 'View Document' and scraping the modal, no debug prints."""
    sources = []
    cards = driver.find_elements(By.CSS_SELECTOR, 'li.MuiCard-root')
    for idx, card in enumerate(cards):
        try:
            corpus_link = card.find_element(By.CSS_SELECTOR, 'a[href*="huggingface.co"]')
            corpus_name = corpus_link.text.strip()
            corpus_url = corpus_link.get_attribute('href')
        except Exception:
            corpus_name = None
            corpus_url = None
        try:
            view_doc_btn = card.find_element(By.XPATH, ".//button[contains(., 'View Document')]")
            driver.execute_script("arguments[0].scrollIntoView(true);", view_doc_btn)
            ActionChains(driver).move_to_element(view_doc_btn).perform()
            view_doc_btn.click()
            time.sleep(3)  # Wait for modal
            modals = driver.find_elements(By.XPATH, "//div[contains(@role, 'dialog') or contains(@class, 'MuiDialogContent-root')]")
            document_url = None
            document_content = None
            for m_idx, modal in enumerate(modals):
                try:
                    a_tags = modal.find_elements(By.TAG_NAME, 'a')
                    for a in a_tags:
                        href = a.get_attribute('href')
                        if href and href.startswith('http') and 'huggingface.co' not in href:
                            document_url = href
                    if not document_content:
                        document_content = modal.text
                except Exception as e:
                    pass
            try:
                close_btn = driver.find_element(By.XPATH, "//button[contains(@class, 'MuiButtonBase-root') and @aria-label='Close']")
                close_btn.click()
            except Exception:
                ActionChains(driver).send_keys_to_element(card, '\ue00c').perform()
            time.sleep(1)
        except Exception as e:
            document_url = None
            document_content = None
        sources.append({
            'corpus_name': corpus_name,
            'corpus_url': corpus_url,
            'document_url': document_url,
            'document_content': document_content
        })
    return sources

def process_prompt(driver, prompt_id, prompt_text):
    """Process a single prompt and save the results."""
    try:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        # Navigate to the AllenAI playground
        driver.get("https://playground.allenai.org/")
        print(f"Navigated to AllenAI playground for prompt {prompt_id}")

        # Wait for the page to load completely
        time.sleep(10)

        # Wait for and find the first textarea on the page
        try:
            message_input = wait_for_element(driver, By.TAG_NAME, "textarea")
            if not message_input:
                # Save screenshot and HTML for debugging
                driver.save_screenshot(f"outputs/debug_prompt_{prompt_id}.png")
                raise Exception("Could not find message input field")
            print("Found message input field (first <textarea> on the page)")
        except Exception as e:
            print(f"Error finding message input: {e}")
            raise

        # Clear and enter the prompt
        message_input.clear()
        message_input.send_keys(prompt_text)
        print(f"Entered prompt text for ID {prompt_id}")

        # Try different selectors for the send button
        send_button = None
        button_selectors = [
            "//button[@type='submit']",
            "//button[contains(text(), 'Send')]",
            "//button[contains(@class, 'send')]",
            "//button[contains(@class, 'submit')]",
            "//button[@aria-label='Send message']"
        ]

        for selector in button_selectors:
            try:
                send_button = wait_for_element(driver, By.XPATH, selector)
                if send_button:
                    print(f"Found send button with selector: {selector}")
                    # If we found a span, get its parent button
                    if send_button.tag_name == 'span':
                        send_button = send_button.find_element(By.XPATH, "./..")
                    break
            except Exception as e:
                print(f"Failed with button selector {selector}: {str(e)}")
                continue

        if not send_button:
            # Try to find any button that might be the send button
            try:
                all_buttons = driver.find_elements(By.TAG_NAME, "button")
                print(f"Found {len(all_buttons)} buttons on the page")
                for button in all_buttons:
                    print(f"Button text: {button.text}, classes: {button.get_attribute('class')}")
                    if "send" in button.text.lower() or "submit" in button.text.lower():
                        send_button = button
                        print(f"Found potential send button with text: {button.text}")
                        break
            except Exception as e:
                print(f"Failed to find any buttons: {str(e)}")

        if not send_button:
            raise Exception("Could not find send button")

        # Click the send button
        send_button.click()
        print(f"Clicked send button for prompt {prompt_id}")

        # Wait for the response to be generated
        time.sleep(25)

        # (add screenshot here)
        driver.save_screenshot(f"outputs/after_submit_prompt_{prompt_id}.png")

        answer_divs = driver.find_elements(By.CSS_SELECTOR, 'div.MuiBox-root.css-0')
        prompt_answer = ''
        if len(answer_divs) > 2:
            ps = answer_divs[2].find_elements(By.CSS_SELECTOR, 'p.css-5tk1pa')
            if ps:
                prompt_answer = '\n'.join([p.text for p in ps])
        elif len(answer_divs) > 0:
            # Fallback: if not enough divs, use the first one (shouldn't happen in your case)
            ps = answer_divs[0].find_elements(By.CSS_SELECTOR, 'p.css-5tk1pa')
            if ps:
                prompt_answer = '\n'.join([p.text for p in ps])
        print(f"[Before trace] Extracted prompt_answer: {prompt_answer}")

        # Now proceed to open the trace panel and extract sources as before
        trace_button = None
        trace_selectors = [
            "//button[contains(@class, 'MuiButton-root')]//span[contains(text(), 'Show OLMoTrace')]",
            "//button[contains(@class, 'css-129qyhd')]",
            "//button[.//span[contains(text(), 'Show OLMoTrace')]]",
            "//button[contains(@class, 'MuiButtonBase-root') and .//span[contains(text(), 'Show OLMoTrace')]]"
        ]

        for selector in trace_selectors:
            try:
                # Wait for the trace button to be clickable
                trace_button_candidate = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.XPATH, selector))
                )
                if trace_button_candidate:
                    # If we found the span, get its parent button
                    if trace_button_candidate.tag_name == 'span':
                        trace_button_candidate = trace_button_candidate.find_element(By.XPATH, "./..")
                    trace_button = trace_button_candidate
                    print(f"Found trace button with selector: {selector}")
                    break
            except Exception as e:
                print(f"Failed with trace selector {selector}: {str(e)}")
                continue

        if not trace_button:
            raise Exception("Could not find trace button")

        # Robustly scroll and click the trace button, even if the response is long
        try:
            # Scroll the window to the bottom to ensure the button is visible
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            driver.execute_script("arguments[0].scrollIntoView(true);", trace_button)
            trace_button.click()
        except Exception as e:
            print(f"Standard click failed: {e}, trying JS click.")
            driver.execute_script("arguments[0].click();", trace_button)
        print(f"Clicked trace button for prompt {prompt_id}")

        # Wait for the trace panel to appear
        time.sleep(5)

        # For parsed_results.csv: use all blockquotes (old logic)
        blockquotes_after = driver.find_elements(By.TAG_NAME, 'blockquote')
        # print(f"[After trace] Found {len(blockquotes_after)} blockquote elements.")
        responses = [b.text for b in blockquotes_after]
        sources = extract_sources_with_selenium(driver)
        # Remove saving of HTML file as it is redundant
        # html_file = f"outputs/response_{prompt_id}.html"
        # with open(html_file, "w", encoding="utf-8") as f:
        #     f.write(driver.page_source)
        # Save Q&A to a separate CSV (qa_results.csv) with prompt_answer field
        qa_csv = 'outputs/qa_results.csv'
        # Collect all unique document URLs for this prompt
        doc_urls = list({s['document_url'] for s in sources if s['document_url']})
        doc_urls_str = '; '.join(doc_urls)
        if not os.path.exists(qa_csv):
            with open(qa_csv, 'w', newline='', encoding='utf-8') as qafile:
                writer = csv.writer(qafile)
                writer.writerow(['prompt_id', 'prompt_text', 'prompt_answer', 'document_urls'])
        with open(qa_csv, 'a', newline='', encoding='utf-8') as qafile:
            writer = csv.writer(qafile)
            writer.writerow([prompt_id, prompt_text, prompt_answer, doc_urls_str])
        # Save parsed_results.csv using only sources
        output_csv = 'outputs/parsed_results.csv'
        existing_prompt_ids = set()
        if os.path.exists(output_csv):
            with open(output_csv, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    existing_prompt_ids.add(row['prompt_id'])
        if str(prompt_id) not in existing_prompt_ids:
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if csvfile.tell() == 0:
                    writer.writerow(['prompt_id', 'corpus_name', 'corpus_url', 'document_url', 'document_content'])
                for source in sources:
                    writer.writerow([
                        prompt_id,
                        source['corpus_name'],
                        source['corpus_url'],
                        source['document_url'],
                        source['document_content']
                    ])
        else:
            print(f"Prompt ID {prompt_id} already exists in CSV, skipping write.")
        print(f"Saved parsed results for prompt {prompt_id} to {output_csv}")
        print(f"Successfully processed prompt {prompt_id}")
    except Exception as e:
        print(f"Error processing prompt {prompt_id}: {str(e)}")
        # Remove saving of error HTML file as it is redundant

def main():
    # Read prompts from CSV file
    df = pd.read_csv("prompts.csv")

    # Initialize the driver
    driver = setup_driver()

    try:
        # Process each prompt
        for _, row in df.iterrows():
            process_prompt(driver, row['id'], row['prompt'])
            time.sleep(8)  # Increased wait time between prompts

    finally:
        # Clean up
        driver.quit()

def clean_parsed_results_csv():
    """Remove duplicate rows from outputs/parsed_results.csv, keeping only the first occurrence."""
    csv_path = 'outputs/parsed_results.csv'
    try:
        df = pd.read_csv(csv_path)
        before = len(df)
        df_clean = df.drop_duplicates()
        after = len(df_clean)
        df_clean.to_csv(csv_path, index=False)
        print(f"Removed {before - after} duplicate rows from {csv_path}.")
    except Exception as e:
        print(f"Error cleaning CSV: {e}")

if __name__ == "__main__":
    main()
    clean_parsed_results_csv()
