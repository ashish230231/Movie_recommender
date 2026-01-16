# !pip install --upgrade llama-index llama-index-embeddings-huggingface llama-index-llms-huggingface transformers torch accelerate bitsandbytes

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
import os
import pandas as pd
import gradio as gr

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = HuggingFaceLLM(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    context_window=2048,
    max_new_tokens=256,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True}
)

csv_file_path = "movie_recommendations_with_names.csv"
df = None

try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}")
    print(f"Current working directory: {os.getcwd()}")

except Exception as e:
    print(f"An unexpected error occurred while reading the CSV: {e}")
    
if df is not None:
    movies_data = [
        Document(text=f"MovieID: {row['movie_id']}, Title: {row['title']}, Genre: {row['genre']}, Rating: {row['rating']}",
                 metadata={"movie_id": row['movie_id'], "title": row['title'], "genre": row['genre'], "rating": row['rating']})
        for index, row in df.iterrows()
    ]
index = VectorStoreIndex.from_documents(movies_data)
query_engine = index.as_query_engine()

def recommend_movie(genre):
    if not genre.strip():
        return "! Please enter a movie genre."
    
    response = query_engine.query(
        f"List titles and genre of movies with genre {genre}."
        f"provide at least 3 recommendations if avalabile."
    )
    
    response_lines= str(response).split("\n")
    filtered = [
        line for line in response_lines
        if line.strip()
        and "Note : The query is not specific" not in line
        ]
    
    recommendations = []
    for rec in filtered:
        if "Title:" in rec:
            try:
                title = rec.split("Title:")[1].split(",")[0].strip()
                recommendations.append(f"{title}")
            except:
                recommendations.append(f"{rec.strip()}")
                
    recommendations = recommendations[:5]
    
    if not recommendations:
        return " Sorry , I couldn't find movies for that genre."
    return "\n".join(recommendations)
#gradio ui

interface = gr.Interface(
    fn=recommend_movie,
    inputs=gr.Textbox(
        label="What type of movie are you in the mood for?",
        placeholder="e.g. Action, Comedy, Drama, Sci-Fi"
    ),
    outputs=gr.Textbox(label="🍿 Movie Recommendations"),
    title="🎥 MovieRecBot",
    description="Movie recommendation system powered by LlamaIndex + TinyLlama (Hugging Face)",
    examples=[["Action"], ["Comedy"], ["Romance"], ["Sci-Fi"]],
)
interface.launch(share=True)

