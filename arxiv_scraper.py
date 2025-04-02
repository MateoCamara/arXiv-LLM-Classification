import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

def scrape_arxiv_page(start=0, size=200):
    """
    Scrapea una página de resultados de ArXiv
    
    Args:
        start: Índice de inicio para los resultados
        size: Cantidad de resultados por página
        
    Returns:
        Lista de diccionarios con la información de cada paper
    """
    base_url = "https://arxiv.org/search/"
    query = "?query=%28deep+OR+neural+OR+generative+model%29+AND+%28audio+OR+sound+OR+effects%29+AND+%28synthesis+OR+generation%29"
    params = f"&searchtype=all&abstracts=show&order=-announced_date_first&size={size}&start={start}"
    url = base_url + query + params
    
    print(f"Scrapeando página con start={start}")
    
    # Agregamos headers para simular un navegador y evitar bloqueos
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Verificar si hay errores HTTP
    except Exception as e:
        print(f"Error al obtener la página: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('li', class_='arxiv-result')
    
    papers = []
    for idx, result in enumerate(results):
        try:
            # Extraer título
            title_elem = result.find('p', class_='title')
            title = title_elem.text.strip() if title_elem else "Sin título"
            
            # Extraer autores - Mejorado para limpiar correctamente
            authors_elem = result.find('p', class_='authors')
            if authors_elem:
                # Eliminar el "Authors:" del inicio
                authors_text = authors_elem.text.replace("Authors:", "").strip()
                
                # Limpieza profunda de autores: eliminar saltos de línea y normalizar espacios
                authors = re.sub(r'\s+', ' ', authors_text)
                # Asegurar que hay un espacio después de cada coma
                authors = re.sub(r',\s*', ', ', authors)
                # Eliminar espacios al principio y final
                authors = authors.strip()
            else:
                authors = "Sin autores"
            
            # Extraer abstract - Mejorado para obtener el texto completo sin etiquetas
            abstract_elem = result.find('span', class_='abstract-full')
            if abstract_elem:
                # Obtener el texto completo sin considerar etiquetas internas
                abstract = abstract_elem.get_text(strip=True)
                
                # Eliminar el texto "△ Less" que puede aparecer al final
                abstract = re.sub(r'△\s*Less$', '', abstract).strip()
            else:
                abstract = "Sin abstract"
            
            # Extraer fecha de envío - Búsqueda mejorada
            # Primero buscar cualquier párrafo que contenga "Submitted"
            submitted_elems = result.find_all('p')
            submitted_date = "Fecha desconocida"
            
            for elem in submitted_elems:
                if "Submitted" in elem.text:
                    submitted_text = elem.text.strip()
                    # Extraer la fecha después de "Submitted"
                    match = re.search(r'Submitted\s+([^;]+)', submitted_text)
                    if match:
                        submitted_date = match.group(1).strip()
                        break
            
            # Extraer ID de ArXiv
            list_title_elem = result.find('p', class_='list-title')
            if list_title_elem and list_title_elem.find('a'):
                arxiv_id = list_title_elem.find('a').text.strip()
            else:
                arxiv_id = "Sin ID"
            
            paper_info = {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'submitted_date': submitted_date,
                'arxiv_id': arxiv_id
            }
            
            papers.append(paper_info)
            
        except Exception as e:
            print(f"Error al procesar el resultado {idx}: {e}")
    
    return papers

def scrape_all_pages(total_results=8427, size=200):
    """
    Scrapea todas las páginas de resultados
    
    Args:
        total_results: Número total de resultados
        size: Cantidad de resultados por página
        
    Returns:
        DataFrame con todos los resultados
    """
    all_papers = []
    
    # Calcular el número de páginas
    num_pages = (total_results + size - 1) // size
    
    for page in range(num_pages):
        start = page * size
        papers = scrape_arxiv_page(start=start, size=size)
        
        if not papers:
            print(f"No se encontraron más resultados después de start={start}")
            break
            
        all_papers.extend(papers)
        print(f"Obtenidos {len(papers)} papers. Total hasta ahora: {len(all_papers)}")
        
        # Guardar progreso parcial cada 1000 papers
        if len(all_papers) % 1000 < size:
            temp_df = pd.DataFrame(all_papers)
            temp_df.to_csv(f"arxiv_papers_partial_{len(all_papers)}.csv", index=False)
            print(f"Guardado progreso parcial con {len(all_papers)} papers")
        
        # Pausa para no sobrecargar el servidor
        time.sleep(5)  # Aumentado para evitar bloqueos
    
    return pd.DataFrame(all_papers)

def test_scraper():
    """
    Función de prueba para verificar que el scraper funciona correctamente
    """
    # Solo probamos con la primera página
    papers = scrape_arxiv_page(start=0, size=200)
    
    if papers:
        print(f"Test exitoso! Se obtuvieron {len(papers)} papers")
        print("\nEjemplo del primer paper:")
        for key, value in papers[0].items():
            if key == 'abstract':
                # Mostrar un poco más del abstract para verificar
                print(f"{key}: {value[:300]}...")
            else:
                print(f"{key}: {value}")
        
        # Guardar en un archivo Excel para verificar
        df = pd.DataFrame(papers)
        df.to_excel("arxiv_test.xlsx", index=False)
        print("\nSe ha guardado el resultado del test en 'arxiv_test.xlsx'")
    else:
        print("El test falló. No se obtuvieron papers.")

def main():
    # Primero ejecutamos el test
    print("Ejecutando test...")
    test_scraper()
    
    # Preguntar si queremos continuar con el scraping completo
    response = input("\n¿Desea continuar con el scraping completo? (s/n): ")
    
    if response.lower() == 's':
        print("\nIniciando scraping completo...")
        df = scrape_all_pages()
        
        # Guardar resultados en CSV
        df.to_csv("arxiv_papers.csv", index=False)
        print(f"\nSe han guardado {len(df)} papers en 'arxiv_papers.csv'")
    else:
        print("\nScraping completo cancelado.")

if __name__ == "__main__":
    main() 