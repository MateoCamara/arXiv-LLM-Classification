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
MODEL_CHATGPT = "gpt-4o-mini-2024-07-18"

# Prompt fijo para la clasificación
CLASIFICACION_PROMPT = """
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
    for paper in papers.iterrows():
        title_lower = paper[1]["title"].lower()
        if title_lower not in seen:
            seen.add(title_lower)
            unique.append(paper[1])
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
def save_checkpoint(checkpoint_index, results, checkpoint_file="checkpoint.json", csv_file="papers_sound_effects.csv"):
    with open(checkpoint_file, "w") as f:
        json.dump({"checkpoint": checkpoint_index}, f)
    # Guardar resultados en Excel
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"Checkpoint guardado en paper #{checkpoint_index} - {len(results)} papers en Excel.")

def load_checkpoint(checkpoint_file="checkpoint.json"):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            return data.get("checkpoint", 0)
    return 0

def main():

    # Leer API key de variables de entorno
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: No se encontró la variable de entorno OPENAI_API_KEY.")
        return

    arxiv_papers = pd.read_csv("./data/01_arxiv_ckpt500.csv", index_col=0)

    # print("Extrayendo papers de Semantic Scholar...")
    # semantic_papers = get_semantic_papers(args.query, args.start_year, args.max_results)
    # print(f"Se obtuvieron {len(semantic_papers)} papers de Semantic Scholar.")

    # Combinar y eliminar duplicados
    all_papers = deduplicate_papers(arxiv_papers)
    print(f"Total de papers únicos: {len(all_papers)}")

    # Cargar checkpoint si existe
    checkpoint_index = load_checkpoint()
    print(f"Reanudando desde el paper #{checkpoint_index}")

    results = []  # Para guardar resultados finales (solo sound effects)

    # Iterar sobre cada paper
    for idx, paper in enumerate(all_papers[checkpoint_index:], start=checkpoint_index):
        print(f"Procesando paper #{idx+1}: {paper['title'][:50]}...")
        tags = classify_paper(paper["title"], paper["summary"], openai_api_key)
        if tags is None:
            print("Error al clasificar, se omite este paper.")
            continue

        # Filtrar por papers de efectos de sonido (se asume que tag 1 y tag 2 deben cumplir)
        tag1 = tags.get("nas", "").lower()
        if "yes" in tag1:
            # Se extrae el modelo de tag 3 y se asigna dataset como "unknown" (por no disponer de info)
            tag2 = tags.get("architecture", "").lower()
            tag3 = tags.get("sound type", "unknown").lower()
            results.append({
                "id": paper["id"],
                "tag1": tag1,
                "tag2": tag2,
                "tag3": tag3
            })

        # Checkpoint cada 10 papers procesados
        if (idx + 1) % 10 == 0:
            save_checkpoint(idx + 1, results)
            # Se puede pausar un poco para evitar límites de API
        time.sleep(.3)

    # Guardar resultados finales
    save_checkpoint(len(all_papers), results)
    print("Proceso completado.")

if __name__ == "__main__":
    main()
