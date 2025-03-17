import argparse
import os
import pandas as pd
import arxiv

from tqdm import tqdm

ARXIV_API_URL = "http://export.arxiv.org/api/query"  # API URL for arXiv

def get_arxiv_papers(query, checkpoint_freq, output_dir, max_results, page_size, delay_seconds, num_retries):
    """
    Extract arXiv papers based on a search query and save periodic checkpoints.

    Parameters:
        query (str): Search query for the arXiv API.
        checkpoint_freq (int): Number of records to process before saving a checkpoint CSV.
        output_dir (str): Directory to save the CSV files.
        max_results (int): Maximum number of results to retrieve.
        page_size (int): Number of results per API page.
        delay_seconds (float): Delay between API requests in seconds.
        num_retries (int): Number of retries for API requests.
    """
    # Create arXiv client with specified parameters
    client = arxiv.Client(
        page_size=page_size,
        delay_seconds=delay_seconds,
        num_retries=num_retries,
    )

    entries_data = []
    results = client.results(arxiv.Search(query=query, max_results=max_results))
    checkpoint = 0

    # Process each result and update checkpoint
    for result in tqdm(results):
        author_names = [author.name for author in result.authors]

        entries_data.append({
            'id': result.entry_id,
            'published': result.published,
            'title': result.title,
            'summary': result.summary,
            'author': ', '.join(author_names),
            'comment': result.comment
        })

        checkpoint += 1
        if checkpoint % checkpoint_freq == 0:
            df = pd.DataFrame(entries_data)
            # Save checkpoint CSV file with current checkpoint count in filename
            checkpoint_file = os.path.join(output_dir, f'01_arxiv_ckpt{checkpoint}.csv')
            df.to_csv(checkpoint_file, index=True)

    # Create final DataFrame with all records
    df = pd.DataFrame(entries_data)
    # Save final DataFrame as CSV file including the index
    final_file = os.path.join(output_dir, '01_arxiv.csv')
    df.to_csv(final_file, index=True)
    print("DataFrame created successfully!")
    print(f"Total records obtained: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and classify arXiv papers")
    parser.add_argument("--query", type=str, required=True, help="Query for the arXiv API")
    parser.add_argument("--checkpoint_freq", type=int, default=1000, help="Frequency (in number of records) to save checkpoint CSV files")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory where CSV files will be saved")
    parser.add_argument("--max_results", type=int, default=20000, help="Maximum number of results to retrieve")
    parser.add_argument("--page_size", type=int, default=100, help="Number of results per page for API requests")
    parser.add_argument("--delay_seconds", type=float, default=10.0, help="Delay between API requests in seconds")
    parser.add_argument("--num_retries", type=int, default=5, help="Number of retries for API requests")
    args = parser.parse_args()

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    get_arxiv_papers(args.query, args.checkpoint_freq, args.output_dir, args.max_results, args.page_size, args.delay_seconds, args.num_retries)
