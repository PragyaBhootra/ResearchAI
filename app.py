import gradio as gr
from PyPDF2 import PdfReader
from semanticscholar import SemanticScholar
from gtts import gTTS
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
topic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sch = SemanticScholar()

def process_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() or '' for page in reader.pages])


import requests
from bs4 import BeautifulSoup

def search_papers(topic, recency="Recent", limit=5):
    print(f"🌐 Searching Semantic Scholar for topic: {topic}")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    try:
        response = requests.get(url, params={
            "query": topic,
            "limit": limit,
            "fields": "title,abstract,url,authors"
        })
        data = response.json()

        papers = []
        if "data" in data:
            print(f"📄 Total papers fetched from Semantic Scholar: {len(data['data'])}")
            for p in data['data']:
                abstract = p.get("abstract")
                if not abstract or not abstract.strip():
                    continue
                papers.append({
                    "title": p.get("title", "Untitled"),
                    "abstract": abstract.strip(),
                    "url": p.get("url", ""),
                    "authors": [a.get("name", "") for a in p.get("authors", [])]
                })

        if papers:
            print(f"✅ Papers with abstracts: {len(papers)}")
            return papers
        else:
            print("🔁 No abstracts found, falling back to CrossRef...")

    except Exception as e:
        print(f"⚠️ Semantic Scholar API error: {e}")
        print("🔁 Trying CrossRef...")

    # Fallback to CrossRef
    try:
        crossref_url = "https://api.crossref.org/works"
        r = requests.get(crossref_url, params={
            "query": topic,
            "rows": limit
        })
        items = r.json().get("message", {}).get("items", [])
        papers = []

        for item in items:
            abstract_html = item.get("abstract", "")
            abstract = BeautifulSoup(abstract_html, "html.parser").get_text() if abstract_html else ""
            if not abstract.strip():
                continue
            papers.append({
                "title": item.get("title", ["Untitled"])[0],
                "abstract": abstract.strip(),
                "url": item.get("URL", ""),
                "authors": [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in item.get("author", [])
                ]
            })

        print(f"✅ Papers fetched from CrossRef: {len(papers)}")
        return papers

    except Exception as ex:
        print(f"❌ CrossRef failed too: {ex}")
        return []


def classify_topic(text, user_topics):
    if not user_topics:
        return "General"
    topic_embeddings = topic_model.encode(user_topics)
    text_embed = topic_model.encode(text)

    similarities = []
    for topic_embed in topic_embeddings:
        cos_sim = torch.nn.functional.cosine_similarity(
            torch.tensor(text_embed), torch.tensor(topic_embed), dim=0
        )
        similarities.append(cos_sim.item())
    return user_topics[np.argmax(similarities)]


def generate_summary(text, chunk_size=3500):
    summaries = []
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    for i, chunk in enumerate(chunks):
        try:
            print(f"📑 Summarizing chunk {i+1}/{len(chunks)}")
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"⚠️ Skipping chunk {i+1} due to error: {e}")

    return "\n".join(summaries) if summaries else "⚠️ Summary generation failed."


def generate_audio(text, filename="summary.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    mp3 = AudioSegment.from_mp3(filename)
    wav_file = filename.replace('.mp3', '.wav')
    mp3.export(wav_file, format="wav")
    return wav_file


def render_html_table(papers):
    html = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            color: black;
        }
        th, td {
            border: 1px solid #ccc;
            padding: 10px;
            vertical-align: top;
        }
        th {
            background-color: #f8f8f8;
        }
        a {
            color: #007bff;
            text-decoration: none;
        }
    </style>
    <table>
        <tr>
            <th>Title</th>
            <th>Authors</th>
            <th>Topic</th>
            <th>Summary</th>
            <th>URL</th>
        </tr>
    """
    for paper in papers:
        html += f"""
        <tr>
            <td>{paper.get('title')}</td>
            <td>{paper.get('authors')}</td>
            <td>{paper.get('topic')}</td>
            <td>{paper.get('summary')}</td>
            <td><a href="{paper.get('url')}" target="_blank">Link</a></td>
        </tr>
        """
    html += "</table>"
    return html


def process_input_ui(topic, recency, file, doi, user_topics):
    papers, synthesis, audio = process_input(topic, recency, file, doi, user_topics)
    html_table = render_html_table(papers)
    return html_table, synthesis, audio


import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment
from textwrap import wrap  # for splitting long summaries

def process_input(topic, recency, file, doi, user_topics):
    print("\n🔍 DEBUG: Starting process_input")
    print(f"   📌 Topic: {topic}")
    print(f"   📌 Recency: {recency}")
    print(f"   📌 DOI: {doi}")
    print(f"   📌 File: {getattr(file, 'name', None)}")
    print(f"   📌 User Topics: {user_topics}")

    papers = []
    user_topic_list = [t.strip() for t in user_topics.split(',')] if user_topics else []
    print(f"   ✅ Parsed user topics: {user_topic_list}")

    if file is not None:
        try:
            print("📄 Reading uploaded PDF...")
            text = process_pdf(file)
            papers.append({
                'title': getattr(file, 'name', 'Uploaded PDF'),
                'content': text,
                'source': 'PDF'
            })
            print("   ✅ PDF processed successfully.")
        except Exception as e:
            print(f"   ❌ Error processing PDF: {e}")
            return [{"error": f"Failed to process PDF: {e}"}], "", None

    elif doi:
        try:
            print("🔗 Fetching paper from DOI via CrossRef...")
            url = f"https://api.crossref.org/works/{doi}"
            r = requests.get(url)
            data = r.json().get('message', {})

            abstract_html = data.get('abstract', '')
            abstract = BeautifulSoup(abstract_html, "html.parser").get_text() if abstract_html else ""

            papers.append({
                'title': data.get('title', ['Untitled'])[0],
                'content': abstract,
                'url': data.get('URL', f"https://doi.org/{doi}"),
                'authors': [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in data.get('author', [])
                ],
                'source': 'DOI'
            })

            print("   ✅ DOI fetched and abstract extracted.")
        except Exception as e:
            print(f"   ❌ DOI fetch error: {e}")
            return [{"error": f"DOI fetch error: {e}"}], "", None

    elif topic:
        print(f"🌐 Searching papers for topic: {topic}")
        papers = search_papers(topic, recency)
        for paper in papers:
            paper['content'] = paper.get('abstract', '')
            paper['source'] = 'Semantic Scholar'
        print(f"   ✅ {len(papers)} papers fetched for topic.")

    else:
        print("⚠️ No input provided (topic, PDF, or DOI).")

    output = []
    for i, paper in enumerate(papers):
        content = paper.get('content', '')
        if not content.strip():
            print(f"   ⚠️ Skipping paper {i+1}: empty content.")
            continue

        print(f"🔎 Processing paper {i+1}: {paper.get('title')}")
        classification = classify_topic(content, user_topic_list)
        print(f"   📂 Classified under topic: {classification}")
        summary = generate_summary(content)
        print(f"   📝 Summary generated.")

        output.append({
            "title": paper.get('title', 'Untitled'),
            "authors": ', '.join(paper.get('authors', [])),
            "topic": classification,
            "summary": summary,
            "url": paper.get('url', '')
        })

    if not output:
        print("❌ No valid paper content found.")
        return [{"error": "No valid paper content found."}], "", None

    # ✅ Hierarchical Summarization for Synthesis
    print("🧠 Generating synthesis across all summaries...")
    all_summaries = "\n".join([p['summary'] for p in output])

    def split_chunks(text, max_chars=1000):
        return wrap(text, max_chars)

    chunks = split_chunks(all_summaries, max_chars=1000)
    partial_summaries = []
    for chunk in chunks:
        summary_piece = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        partial_summaries.append(summary_piece)

    final_input = " ".join(partial_summaries)
    synthesis = summarizer(final_input, max_length=200, min_length=60, do_sample=False)[0]['summary_text']
    print("   ✅ Synthesis generated.")

    print("🎧 Generating audio summary (podcast)...")
    audio_file = generate_audio(synthesis)
    print(f"   ✅ Audio file saved as: {audio_file}")

    print("✅ Finished processing.\n")
    return output, synthesis, audio_file



# Gradio UI
with gr.Blocks(title="Research Paper Podcast Generator") as app:
    gr.Markdown("## 🎙️ Research Paper Summarization and Podcast Generator")

    with gr.Row():
        with gr.Column():
            topic = gr.Textbox(label="Research Topic")
            recency = gr.Radio(["Recent", "Relevant", "Highly Cited"],
                               value="Recent", label="Filter")
            user_topics = gr.Textbox(label="Topics for Classification (comma-separated)",
                                     placeholder="e.g., AI, Climate Change, Blockchain")
        with gr.Column():
            doi = gr.Textbox(label="DOI Reference")
            file = gr.File(label="Upload PDF", file_types=[".pdf"])

    generate_btn = gr.Button("🔍 Generate")

    with gr.Tab("📄 Paper Details"):
        ##papers_output = gr.JSON(label="Analyzed Papers")
        papers_output = gr.HTML(label="Analyzed Papers")

    with gr.Tab("🧠 Synthesis"):
        synthesis = gr.Textbox(label="Cross-Paper Summary",lines=12, max_lines=20)
    with gr.Tab("🎧 Podcast"):
        audio = gr.Audio(label="Audio Summary", type="filepath")

    generate_btn.click(
        fn=process_input_ui,
        inputs=[topic, recency, file, doi, user_topics],
        outputs=[papers_output, synthesis, audio]
    )


app.launch(server_name="0.0.0.0", server_port=8080)
