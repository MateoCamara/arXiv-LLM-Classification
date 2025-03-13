import os
import time
import json
import argparse
import requests
import feedparser
import pandas as pd

# Constantes y configuración
ARXIV_API_URL = "http://export.arxiv.org/api/query"
SEMANTIC_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CHATGPT_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL_CHATGPT = "gpt-3.5-turbo"

# Prompt fijo para la clasificación
CLASIFICACION_PROMPT = """
You are a helpful assistant. You are going to evaluate and classify the titles and abstracts of scientific papers.

You have to understand what is the topic of the paper based on the title and abstract. There is no more information, so no need to ask back.

The first thing you have to do is to understand the topic based on title and abstract about the paper. Then, you have to determine if it is related to the context of "Neural Audio Synthesis".

After this, you have to classify the paper based on the following taxonomies:

tag 1: related to Neural Audio Synthesis | not related to Neural Audio Synthesis
tag 2: music | speech | audio | sound effect
tag 3: VAE | GAN | Diffussion | Transformer | DNN | CNN | RNN | LSTM | GRU | Attention | WaveNet | WaveGlow | Tacotron | BERT | GPT | Transformer-XL | XLNet | RoBERTa | ALBERT | T5 | BART | ELECTRA | Reformer | Longformer | BigBir

You can always answer with multiple tags, separated by commas. In the case of tag 3, you can add new values beyond the taxonomy if you think it is necessary, based on the model or architecture used in the paper.

You can always say "unknown" if there is not enough information to identify in tags 2 and 3. Don't be greedy, if you can't know, type "unknown". It is better to say "unknown" than to make a mistake.

JUST REPLY WITH THE TAGS, NOTHING ELSE. NOTHING LIKE "I'm happy to help" nor explaining what you've done. Only tags.
"""

# Función para obtener papers de arXiv
def get_arxiv_papers(query, start_year, max_results=100):
    papers = []
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results
    }
    response = requests.get(ARXIV_API_URL, params=params)
    feed = feedparser.parse(response.text)
    for entry in feed.entries:
        # Extraer año de la fecha de publicación
        pub_year = int(entry.published[:4])
        if pub_year >= start_year:
            paper = {
                "title": entry.title.strip(),
                "abstract": entry.summary.strip(),
                "year": pub_year
            }
            papers.append(paper)
    return papers

# Función para obtener papers de Semantic Scholar
def get_semantic_papers(query, start_year, max_results=100):
    papers = []
    offset = 0
    limit = 100 if max_results > 100 else max_results
    headers = {"Accept": "application/json"}
    while offset < max_results:
        params = {
            "query": query,
            "offset": offset,
            "limit": limit,
            "fields": "title,abstract,year"
        }
        res = requests.get(SEMANTIC_API_URL, params=params, headers=headers)
        if res.status_code != 200:
            break
        data = res.json()
        if "data" not in data:
            break
        for paper in data["data"]:
            paper_year = paper.get("year", 0)
            if paper_year and paper_year >= start_year and paper.get("abstract"):
                papers.append({
                    "title": paper["title"].strip(),
                    "abstract": paper["abstract"].strip(),
                    "year": paper_year
                })
        # Si ya no hay más datos, se termina
        if len(data["data"]) < limit:
            break
        offset += limit
    return papers

# Función para eliminar duplicados (basado en título)
def deduplicate_papers(papers):
    seen = set()
    unique = []
    for paper in papers:
        title_lower = paper["title"].lower()
        if title_lower not in seen:
            seen.add(title_lower)
            unique.append(paper)
    return unique

# Función para llamar a la API de ChatGPT y clasificar un paper
def classify_paper(title, abstract, api_key):
    # Construir el mensaje combinando el prompt y los datos del paper
    message = f"{CLASIFICACION_PROMPT}\n\ntitle: {title}\nabstract: {abstract}\n"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": MODEL_CHATGPT,
        "messages": [
            {"role": "user", "content": message}
        ],
        "temperature": 0
    }
    response = requests.post(CHATGPT_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error en API ChatGPT: {response.status_code} {response.text}")
        return None
    # Se asume que la respuesta es un string con los tags
    reply = response.json()["choices"][0]["message"]["content"]
    # Parsear la respuesta para extraer tags
    tags = {}
    for line in reply.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            tags[key.strip().lower()] = value.strip()
    return tags

# Función para guardar checkpoint y resultados parciales
def save_checkpoint(checkpoint_index, results, checkpoint_file="checkpoint.json", excel_file="papers_sound_effects.xlsx"):
    with open(checkpoint_file, "w") as f:
        json.dump({"checkpoint": checkpoint_index}, f)
    # Guardar resultados en Excel
    df = pd.DataFrame(results)
    df.to_excel(excel_file, index=False)
    print(f"Checkpoint guardado en paper #{checkpoint_index} - {len(results)} papers en Excel.")

def load_checkpoint(checkpoint_file="checkpoint.json"):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            return data.get("checkpoint", 0)
    return 0

def main():
    parser = argparse.ArgumentParser(description="Extraer papers y clasificarlos")
    parser.add_argument("--start_year", type=int, required=True, help="Año desde el que se extraen papers")
    parser.add_argument("--query", type=str, default="all:artificial intelligence", help="Query para búsqueda en APIs")
    parser.add_argument("--max_results", type=int, default=200, help="Máximo número de resultados por API")
    args = parser.parse_args()

    # Leer API key de variables de entorno
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: No se encontró la variable de entorno OPENAI_API_KEY.")
        return

    # Extraer papers de arXiv y Semantic Scholar
    print("Extrayendo papers de arXiv...")
    arxiv_papers = get_arxiv_papers(args.query, args.start_year, args.max_results)
    print(f"Se obtuvieron {len(arxiv_papers)} papers de arXiv.")

    print("Extrayendo papers de Semantic Scholar...")
    semantic_papers = get_semantic_papers(args.query, args.start_year, args.max_results)
    print(f"Se obtuvieron {len(semantic_papers)} papers de Semantic Scholar.")

    # Combinar y eliminar duplicados
    all_papers = deduplicate_papers(arxiv_papers + semantic_papers)
    print(f"Total de papers únicos: {len(all_papers)}")

    # Cargar checkpoint si existe
    checkpoint_index = load_checkpoint()
    print(f"Reanudando desde el paper #{checkpoint_index}")

    results = []  # Para guardar resultados finales (solo sound effects)

    # Iterar sobre cada paper
    for idx, paper in enumerate(all_papers[checkpoint_index:], start=checkpoint_index):
        print(f"Procesando paper #{idx+1}: {paper['title'][:50]}...")
        tags = classify_paper(paper["title"], paper["abstract"], openai_api_key)
        if tags is None:
            print("Error al clasificar, se omite este paper.")
            continue

        # Filtrar por papers de efectos de sonido (se asume que tag 1 y tag 2 deben cumplir)
        tag1 = tags.get("tag 1", "").lower()
        tag2 = tags.get("tag 2", "").lower()
        if "related to neural audio synthesis" in tag1 and "sound effect" in tag2:
            # Se extrae el modelo de tag 3 y se asigna dataset como "unknown" (por no disponer de info)
            model_used = tags.get("tag 3", "unknown")
            dataset_used = "unknown"
            results.append({
                "titulo": paper["title"],
                "abstract": paper["abstract"],
                "modelo_utilizado": model_used,
                "dataset_utilizado": dataset_used
            })

        # Checkpoint cada 10 papers procesados
        if (idx + 1) % 10 == 0:
            save_checkpoint(idx + 1, results)
            # Se puede pausar un poco para evitar límites de API
            time.sleep(1)

    # Guardar resultados finales
    save_checkpoint(len(all_papers), results)
    print("Proceso completado.")

if __name__ == "__main__":
    main()
