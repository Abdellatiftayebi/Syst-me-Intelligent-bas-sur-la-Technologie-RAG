# 🤖 Systme-Intelligent-bas-sur-la-Technologie-RAG

 est une application intelligente de question-réponse basée sur la technologie RAG (*Retrieval-Augmented Generation*). Elle permet de poser des questions sur le contenu d’un ou plusieurs fichiers PDF, avec des réponses générées par **Gemini (Google Generative AI)** en se basant uniquement sur les documents fournis.

---

## 🎯 Objectif

Permettre à un utilisateur :
- de téléverser un ou plusieurs fichiers PDF ;
- de poser des questions en langage naturel ;
- d’obtenir des réponses précises basées exclusivement sur le contenu des PDF.

---

## 🧰 Technologies utilisées

| Composant | Description |
|----------|-------------|
| `Streamlit` | Interface utilisateur web |
| `PyPDF2` | Extraction de texte depuis les fichiers PDF |
| `LangChain` | Pipeline de traitement, découpage, chaînes et intégration FAISS |
| `GoogleGenerativeAI` | Embeddings et LLM (modèle Gemini) |
| `FAISS` | Recherche vectorielle rapide et efficace |

---

## 🧪 Fonctionnement

### 1. Chargement du ou des PDF
- Extraction du texte avec `PyPDF2`
- Découpage intelligent en "chunks" de 1000 caractères

### 2. Indexation vectorielle
- Création d’un index FAISS à partir des **embeddings Gemini**

### 3. Génération des réponses
- Recherche des passages pertinents via FAISS
- Génération de la réponse via **Gemini 1.5 Flash**
- Respect d’un prompt strict pour garantir la qualité, la précision et l’honnêteté des réponses

---

## ⚙️ Installation


### 1. Cloner le dépôt


```bash
git clone https://github.com/votre-utilisateur/pdf-qa-gemini.git
cd pdf-qa-gemini
```
### 2. Créer un environnement virtuel
 ```bash 
 python -m venv venv
 source venv/bin/activate  # Sous Windows : venv\Scripts\activate
```
### 3. Installer les dépendances
   ```bash
 pip install -r requirements.txt
  ```
### 4. Ajouter votre clé API Gemini or openia
```python
 GEMINI_API_KEY = "votre_clé_api_google"
   ```
# 🚀 Exécution
```python
 streamlit run rag.py
   ```
# 📁 Structure du projet
```bash
pdf-qa-gemini/
├── app.py                  
├── rag.png                
├── requirements.txt        
└── README.md             

