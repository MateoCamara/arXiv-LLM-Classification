import os
import time
import json
import argparse
import requests
import pandas as pd

# ChatGPT API configuration
CHATGPT_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL_CHATGPT = "gpt-4o-mini-2024-07-18"

# Prompt fijo para la clasificación
CLASSIFICATION_PROMPT = """
You will analyze the title and abstract of an academic paper to provide three specific tags based strictly on the following criteria:

1. **Neural Audio Synthesis (NAS)**:
   - Tag as `NAS: YES` if the topic explicitly involves synthesizing audio using neural networks of any kind.
   - Tag as `NAS: NO` if it uses traditional synthesis methods (additive, subtractive, granular, filters, etc.) without neural networks or if it is not related to synthesizing sound at all.

2. **Sound Type**:
   - Indicate the type(s) of sound the paper addresses among `music`, `speech`, and `sound effects`.
   - Clarifications:
      - `speech`: related exclusively to spoken language.
      - `music`: musical audio generation.
      - `sound effects`: all generated audio that is neither music nor speech.
   - Multiple sound types can appear; separate them by commas if more than one applies (e.g., `music, speech`).

3. **AI Architecture**:
   - Identify the neural network architecture explicitly used to synthesize audio (e.g., `VAE`, `GAN`, `Diffusion`, `Transformer`, etc.).
   - Important clarification: Report only the architecture that directly synthesizes audio. If another AI architecture is used for tasks like text interpretation or conditioning but not directly synthesizing audio, it must not be included here.
   - If the architecture is not explicitly mentioned, tag as `Architecture: Not specified`.

Your output should strictly follow this format, without additional messages or explanations:

```
NAS: YES or NO
Sound Type: [music/speech/sound effects]
Architecture: [Architecture type or Not specified]
```

"""

def deduplicate_papers(papers):
    """
    Remove duplicate papers based on title (case insensitive).

    Parameters:
        papers (pd.DataFrame): DataFrame containing papers.

    Returns:
        list: A list of unique paper records (pandas Series objects).
    """
    seen = set()
    unique = []
    for _, paper in papers.iterrows():
        title_lower = paper["title"].lower()
        if title_lower not in seen:
            seen.add(title_lower)
            unique.append(paper)
    return unique

def classify_paper(title, abstract, api_key):
    """
    Call the ChatGPT API to classify a paper using its title and abstract.

    Parameters:
        title (str): The paper title.
        abstract (str): The paper abstract.
        api_key (str): API key for OpenAI.

    Returns:
        dict: Dictionary with classification tags.
    """
    # Build message combining the prompt and paper data
    message = f"{CLASSIFICATION_PROMPT}\n\ntitle: {title}\nabstract: {abstract}\n"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": MODEL_CHATGPT,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0
    }
    response = requests.post(CHATGPT_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"ChatGPT API error: {response.status_code} {response.text}")
        return None
    reply = response.json()["choices"][0]["message"]["content"]
    # Parse the response to extract tags
    tags = {}
    for line in reply.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            tags[key.strip().lower()] = value.strip()
    return tags

def save_checkpoint(checkpoint_index, results, checkpoint_file, csv_file):
    """
    Save a checkpoint and partial results.

    Parameters:
        checkpoint_index (int): Current paper index.
        results (list): List of result dictionaries.
        checkpoint_file (str): File path to save checkpoint data.
        csv_file (str): File path to save the results CSV.
    """
    with open(checkpoint_file, "w") as f:
        json.dump({"checkpoint": checkpoint_index}, f)
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"Checkpoint saved at paper #{checkpoint_index} - {len(results)} papers in CSV.")

def load_checkpoint(checkpoint_file):
    """
    Load checkpoint from file if it exists.

    Parameters:
        checkpoint_file (str): File path for checkpoint data.

    Returns:
        int: The last processed paper index.
    """
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            return data.get("checkpoint", 0)
    return 0

def main():
    """
    Main function to classify arXiv papers.
    """
    parser = argparse.ArgumentParser(
        description="Classify arXiv papers using ChatGPT based on title and abstract."
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file with arXiv papers")
    parser.add_argument("--checkpoint_freq", type=int, default=10, help="Frequency (number of papers) to save a checkpoint")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint.json", help="Path to save the checkpoint file")
    parser.add_argument("--csv_file", type=str, default="papers_sound_effects.csv", help="Path to save the results CSV file")
    parser.add_argument("--sleep_time", type=float, default=0.3, help="Time to sleep between API calls (in seconds)")
    args = parser.parse_args()

    # Read OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not found.")
        return

    # Load input CSV file with arXiv papers
    arxiv_papers = pd.read_csv(args.input_csv, index_col=0)

    # Remove duplicate papers based on title
    all_papers = deduplicate_papers(arxiv_papers)
    print(f"Total unique papers: {len(all_papers)}")

    # Load checkpoint if exists
    checkpoint_index = load_checkpoint(args.checkpoint_file)
    print(f"Resuming from paper #{checkpoint_index}")

    results = []  # To store final results (only sound effects)

    # Process each paper
    for idx, paper in enumerate(all_papers[checkpoint_index:], start=checkpoint_index):
        print(f"Processing paper #{idx+1}: {paper['title'][:50]}...")
        tags = classify_paper(paper["title"], paper["summary"], openai_api_key)
        if tags is None:
            print("Error classifying paper, skipping.")
            continue

        # Filter papers for sound effects based on tags
        tag_nas = tags.get("nas", "").lower()
        if "yes" in tag_nas:
            tag_architecture = tags.get("architecture", "").lower()
            tag_sound_type = tags.get("sound type", "unknown").lower()
            results.append({
                "id": paper["id"],
                "tag1": tag_nas,
                "tag2": tag_architecture,
                "tag3": tag_sound_type
            })

        # Save checkpoint every defined number of papers
        if (idx + 1) % args.checkpoint_freq == 0:
            save_checkpoint(idx + 1, results, args.checkpoint_file, args.csv_file)
        time.sleep(args.sleep_time)

    # Save final results
    save_checkpoint(len(all_papers), results, args.checkpoint_file, args.csv_file)
    print("Process completed.")

if __name__ == "__main__":
    main()