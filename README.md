# 🎬 IMDb Movie Recommendation System

A content-based recommender that suggests similar movies by comparing what their **storylines** are about, built on 4,300+ movies released in 2024 that I scraped from IMDb.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Selenium](https://img.shields.io/badge/Selenium-43B02A?logo=selenium&logoColor=white) ![NLTK](https://img.shields.io/badge/NLTK-154F5B) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

🎥 **[Watch the demo video](Screen%20Recording%202026-04-12%20183628.mp4)**

## 🧠 How it works

1. **Scraping:** Selenium collects the title and storyline of 4,300+ movies released in 2024.
2. **Cleaning & EDA:** duplicates and missing storylines removed, text lengths explored.
3. **NLP preprocessing:** lower-casing, contraction expansion, tokenization, stop-word removal and lemmatization with NLTK.
4. **Vectorization:** storylines converted into TF-IDF vectors.
5. **Recommendation:** cosine similarity finds the movies whose storylines are closest to the one you pick.
6. **App:** a Streamlit interface to search a movie and see the top matches.

## 📁 Project structure

| File | Purpose |
|---|---|
| `imdb.ipynb` | Selenium scraping |
| `imdb_movie.ipynb` | Cleaning, NLP preprocessing, TF-IDF model |
| `imdb_app.py` | Streamlit app |

## ⚙️ Run it locally

```bash
git clone https://github.com/Arunmozhi1997/IMDB-Movie-Recommendation-System.git
cd IMDB-Movie-Recommendation-System
pip install -r requirements.txt

# Run imdb_movie.ipynb first: it creates imdb_cleaned.csv, tfidf_vectorizer.pkl and tfidf_matrix.npz
streamlit run imdb_app.py
```

## 👤 Author

**Arunmozhi M** · [Portfolio](https://arunmozhi1997.github.io) · [LinkedIn](https://linkedin.com/in/arunmozhi-m-65a29037b)

