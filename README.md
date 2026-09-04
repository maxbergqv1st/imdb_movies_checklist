# IMDB Movie Checklist

Streamlit-app och Jupyter-notebook för att hålla koll på sedda filmer och få personliga rekommendationer baserade på din tittarhistorik.

---

## Struktur

```
imdb_movies_checklist/
├── app.py              # Streamlit UI (View)
├── pipeline.py         # Kopplar ihop modeller (Controller)
├── pipeline.ipynb      # Notebook-version för experiment
├── data/
│   └── loader.py       # CSV-inläsning & kolumnnormalisering
├── models/
│   ├── base.py         # Abstrakt BaseRecommender
│   ├── cosine.py       # TF-IDF cosine similarity
│   ├── logistic.py     # Logistic Regression
│   └── ensemble.py     # Viktat snitt av modellerna
├── .env                # Lokal konfiguration (gitignorerad)
└── requirements.txt
```

---

## Krav

- Python 3.10+
- En IMDB-CSV från Kaggle, t.ex. [Top 1000 Movies](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)

---

## Setup

### 1. Klona repot

```bash
git clone https://github.com/maxbergqv1st/imdb_movies_checklist.git
cd imdb_movies_checklist
```

### 2. Skapa virtuell miljö och installera beroenden

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Lägg till dataset

Ladda ner din Kaggle-CSV och döp om den till `movies.csv` i projektmappen:

```bash
mv ~/Downloads/imdb_top_1000.csv movies.csv
```

Alternativt: ändra sökvägen i `.env`:

```
CSV_PATH=sökväg/till/din/fil.csv
```

Stödda kolumnnamn (normaliseras automatiskt):

| Kaggle-kolumn | Intern namn |
|---|---|
| `Series_Title` / `Title` | `title` |
| `Released_Year` / `Year` | `year` |
| `IMDB_Rating` / `Rating` | `rating` |
| `Genre` | `genre` |
| `Overview` / `Description` | `overview` |
| `Director` | `director` |
| `Runtime` | `runtime` |
| `No_of_Votes` / `Votes` | `votes` |

---

## Köra appen

```bash
source .venv/bin/activate
streamlit run app.py
```

Öppnas på [http://localhost:8501](http://localhost:8501).

**All Movies** — sök, filtrera på genre, kryssa i sedda filmer.  
**Recommendations** — pipeline tränas automatiskt på din lista och visar topp-N osedda filmer.

---

## Köra notebook

```bash
source .venv/bin/activate
jupyter notebook pipeline.ipynb
```

Kör cellerna uppifrån och ned. Ändra vikter eller byt modell i **Pipeline**-cellen och kör om därifrån.

---

## Lägga till en ny modell

1. Skapa `models/min_modell.py`:

```python
from .base import BaseRecommender
import pandas as pd

class MinModell(BaseRecommender):
    def fit(self, df: pd.DataFrame, watched: set) -> None:
        ...  # träna här

    def score(self, df: pd.DataFrame) -> pd.Series:
        ...  # returnera pd.Series med score per rad
```

2. Registrera i `pipeline.py`:

```python
from models.min_modell import MinModell

def build_pipeline():
    return EnsembleRecommender([
        (CosineRecommender(),   0.5),
        (LogisticRecommender(), 0.3),
        (MinModell(),           0.2),
    ])
```

Ensemblen normaliserar alla scores till [0, 1] innan viktat snitt — vikterna behöver inte summera till 1.

---

## Testa pipeline manuellt

Kör detta i terminalen för att verifiera att hela flödet fungerar utan att starta appen:

```bash
source .venv/bin/activate
python3 - <<'EOF'
import pandas as pd
from data.loader import load_df
from pipeline import build_pipeline

df = load_df("movies.csv")
watched = set(df["title"].head(5))  # låtsas att du sett de 5 första

pipeline = build_pipeline()
pipeline.fit(df, watched)

recs = pipeline.recommend(n=5)
print(recs[["title", "rating", "genre"]].to_string(index=False))
EOF
```

Förväntat output: 5 filmer du inte "sett", rankade av modellerna.

---

## Vanliga fel

**`ModuleNotFoundError: No module named 'streamlit'`**  
→ Glömt aktivera `.venv`: `source .venv/bin/activate`

**`FileNotFoundError: movies.csv`**  
→ CSV-filen saknas i projektmappen, eller fel sökväg i `.env`

**Recommendations visar ingenting**  
→ Du behöver ha minst 1 film markerad som sedd i **All Movies**-fliken för att modellerna ska kunna tränas
