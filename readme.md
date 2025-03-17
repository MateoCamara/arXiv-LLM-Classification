# arXiv LLM Classification

![Logo](logo.png)

This repository contains scripts to extract academic papers from arXiv and classify them using the ChatGPT API. The workflow consists of two main parts:

1. **arXiv Paper Extraction**  
   Extract papers from the arXiv API. You can customize your search query to select a specific category or search type. For more details, consult the [arXiv API documentation](http://export.arxiv.org/api_help/).  
   **Example query:**  
   `"cat:eess.AS AND submittedDate:[2014 TO 2026]"`

2. **Paper Classification**  
   Classify papers using ChatGPT API based on their title and abstract. This script tags papers for Neural Audio Synthesis (NAS), sound type, and AI architecture.  
   **Important:** This part uses the ChatGPT API. Make sure to set your API key in the environment variable `OPENAI_API_KEY`.

## Requirements

- Python 3.10+
- Required libraries: `os`, `time`, `json`, `argparse`, `requests`, `pandas`

Install the required libraries (if not already installed):

```bash
pip install -r requirements.txt
```

## Usage

### 1. Extract Papers from arXiv

Run the extraction script (e.g., `extract_arxiv.py`) with your query and desired parameters:

```bash
python 01_arxiv.py --query "cat:eess.AS AND submittedDate:[2014 TO 2026]" --checkpoint_freq 100 --output_dir ./data --max_results 20000 --page_size 100 --delay_seconds 10.0 --num_retries 5
```

*Note:* Review the [arXiv API documentation](http://export.arxiv.org/api_help/) for additional details on constructing search queries. This step can take hours to complete, depending on the number of papers you are extracting!

### 2. Classify Papers Using ChatGPT API

After you have extracted the papers into a CSV file, run the classification script (e.g., `classify_papers.py`):

```bash
python 02_llm.py --input_csv ./data/01_arxiv_ckpt500.csv --prompt_file prompt.txt --checkpoint_freq 10 --checkpoint_file checkpoint.json --csv_file papers_sound_effects.csv --sleep_time 0.3
```

Before running, ensure that your environment variable for the ChatGPT API key is set:

```bash
export OPENAI_API_KEY=your_api_key_here
```

*Warning:* This script calls the ChatGPT API. Usage may incur charges according to OpenAI's pricing.

## Prompt Customization

The LLM prompt is stored in prompt.txt. The default prompt example is about Neural Audio Synthesis. You can modify this file to suit any classification task or research topic.

## License

Give credit to the original authors by referencing this repository.

## Contributing

Contributions are welcome. Please follow the standard pull request process.
