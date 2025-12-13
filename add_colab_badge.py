import os
import json

# --- KONFIGURACJA ---
GITHUB_USER = "takzen"  
REPO_NAME = "ai-engineering-handbook"  
BRANCH = "main"                        
# --------------------

def create_badge_cell(file_path):
    # Zamieniamy ścieżkę lokalną (np. Windowsową) na format URL (ukośniki /)
    # Usuwamy "./" z początku jeśli jest
    relative_path = file_path.replace("\\", "/").lstrip("./")
    
    colab_url = f"https://colab.research.google.com/github/{GITHUB_USER}/{REPO_NAME}/blob/{BRANCH}/{relative_path}"
    
    badge_markdown = f"""
<a href="{colab_url}" target="_parent">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
"""
    
    # Struktura komórki w Jupyter Notebook (JSON)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [badge_markdown]
    }

def process_notebooks():
    count = 0
    # Przechodzimy przez wszystkie pliki w folderze i podfolderach
    for root, dirs, files in os.walk("."):
        # Ignorujemy folder venv i .git
        if ".venv" in root or ".git" in root or ".ipynb_checkpoints" in root:
            continue

        for file in files:
            if file.endswith(".ipynb"):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Sprawdzamy, czy pierwsza komórka to już nie jest badge (żeby nie dublować)
                    first_cell_source = ""
                    if data['cells'] and 'source' in data['cells'][0]:
                        first_cell_source = "".join(data['cells'][0]['source'])
                    
                    if "colab-badge.svg" in first_cell_source:
                        print(f"⏩ Pomijam (już ma badge): {file}")
                        continue

                    # Dodajemy nową komórkę na początek listy (index 0)
                    new_cell = create_badge_cell(file_path)
                    data['cells'].insert(0, new_cell)
                    
                    # Zapisujemy plik
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=1) # indent=1 ładnie formatuje JSON
                        
                    print(f"✅ Dodano badge do: {file}")
                    count += 1
                    
                except Exception as e:
                    print(f"❌ Błąd w pliku {file}: {e}")

    print("-" * 30)
    print(f"Zakończono! Zaktualizowano {count} notatników.")

if __name__ == "__main__":
    process_notebooks()