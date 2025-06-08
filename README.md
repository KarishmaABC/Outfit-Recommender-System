# 👗 Outfit Recommendation System 👗

An AI-powered Outfit recommendation engine that suggests outfits based on style compatibility, image embeddings, and metadata. Built using CLIP, Streamlit, and fashion datasets.

![Outfit](https://img.shields.io/badge/Fashion-Recommendation-blueviolet?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.11-blue.svg) ![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange)

---

## 🌟 Features

- 🧠 AI-based outfit recommendations using image embeddings (CLIP model).
- 📊 Dashboard-ready visualizations (top brands, categories, discounts)
- 📸 Visual similarity matching for dresses and jeans.
- 💡 Intuitive Streamlit web interface.
- 📁 Upload outfit images and get similar matches.

---

## 🛠️ Installation

Follow these steps to get the project running locally:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/outfit-recommendation-system.git
cd outfit-recommendation-system

```
### 2. Create a virtual environment
```
python -m venv myenv
source myenv/bin/activate 
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Download CLIP model and fashion dataset

```
1.This system uses the CLIP model.
2.Required weights are downloaded automatically using the clip package.
3.Place your fashion images in images/.
4.Ensure fashion_data.csv is in the data/processed folder.
```

▶️ Running the App
```
streamlit run main.py


```


🧠 How It Works
```
CLIP Model extracts visual features from fashion images.

Embeddings are stored and compared using cosine similarity.

Users upload a clothing image.

App returns visually and categorically similar outfits.


```

![output1](https://github.com/user-attachments/assets/19c519ef-9aa1-4005-a844-8fdce2ba04b1)


![output2](https://github.com/user-attachments/assets/bff4456c-8972-43cf-b342-6aa26b240cd4)


![Screenshot (178)](https://github.com/user-attachments/assets/94a6b8ac-1d73-495e-b517-523ef1ead6c3)




📊 Visual Insights
```
Most recommended item types

Popular brands in dataset

Embedding similarity visualizations
```
## 📊 Data Insights & Visualizations

### 🏷️ Most Popular Brands
![Screenshot (184)](https://github.com/user-attachments/assets/450cbf62-016c-4627-9859-c77bdb1bee97)



🤖 Tech Stack

```
Python

CLIP (OpenAI)

Torch

Streamlit

Pandas, NumPy, Matplotlib, Seaborn




