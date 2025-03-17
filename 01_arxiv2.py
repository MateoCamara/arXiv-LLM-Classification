import argparse
import time

import pandas as pd
import requests
import urllib.parse
import xml.etree.ElementTree as ET
import arxiv
from tqdm import tqdm

ARXIV_API_URL = "http://export.arxiv.org/api/query"


# Define la query y los parámetros de paginación

def get_arxiv_papers(query):
    big_slow_client = arxiv.Client(
        page_size=100,
        delay_seconds=10.0,
        num_retries=5,
    )

    entries_data = []

    results = big_slow_client.results(arxiv.Search(query=query, max_results=20000))

    checkpoint = 0

    for result in tqdm(results):
        author_names = [i.name for i in result.authors]

        entries_data.append({
            'id': result.entry_id,
            'published': result.published,
            'title': result.title,
            'summary': result.summary,
            'author': ', '.join(author_names),
            'comment': result.comment
        })

        checkpoint += 1
        if checkpoint % 100 == 0:
            df = pd.DataFrame(entries_data)
            df.to_csv(f'./data/01_arxiv_ckpt{checkpoint}.csv', index=True)

    # Crear el DataFrame final
    df = pd.DataFrame(entries_data)
    # Guardar el DataFrame 'df' como un archivo CSV sin incluir el índice
    df.to_csv('./data/01_arxiv.csv', index=True)
    print("DataFrame creado con éxito!")
    print(f"Total de registros obtenidos: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraer papers y clasificarlos")
    parser.add_argument("--query", type=str, default='cat:eess.AS AND submittedDate:[2014 TO 2026]', help="Query para búsqueda en APIs")
    args = parser.parse_args()
    get_arxiv_papers(args.query)
