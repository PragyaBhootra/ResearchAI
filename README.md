#  ResearchAI: Research Paper Summarizer and Podcast Generator

**ResearchAI** is an AI-powered web app that simplifies the process of searching, summarizing, classifying, and converting academic research into audio podcasts. It leverages **LLMs**, **Gradio**, and various scholarly APIs to help students and researchers.

---

## 🌟 Features

- 🔍 **Semantic Paper Search** via Semantic Scholar and CrossRef
- 📝 **LLM-Based Summarization** using `facebook/bart-large-cnn`
- 🧠 **Topic Classification** using Sentence Transformers
- 📂 **PDF & DOI Support**: Input a research paper through PDF or DOI
- 🔊 **Audio Summary Generation** using gTTS + Pydub
- 🖥️ **Interactive Gradio UI**

---

## 🌐 Live App

🔗 Try out the live app here: [https://huggingface.co/spaces/Pragya123/ResearchAI](https://huggingface.co/spaces/Pragya123/ResearchAI)

---

## 🛠️ Tech Stack

| Component         | Library/Tool                                                                 |
|------------------|-------------------------------------------------------------------------------|
| **Frontend**      | [Gradio](https://www.gradio.app/)                                            |
| **LLM**           | [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) |
| **Topic Modeling**| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **TTS Audio**     | `gTTS`, `pydub`, `ffmpeg`                                                    |
| **PDF Parsing**   | `PyPDF2`                                                                     |
| **Research API**  | [Semantic Scholar](https://api.semanticscholar.org/), [CrossRef](https://api.crossref.org/) |

---

## 📂 Installation and setup

git clone https://github.com/PragyaBhootra/ResearchAI.git
cd ResearchAI
pip install -r requirements.txt
python app.py

---

