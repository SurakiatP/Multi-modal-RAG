"""
Main entry point for Robot Store application
"""
import gradio as gr
import torch

from src.models.embeddings import EmbeddingModels
from src.database.qdrant_client import QdrantService
from src.services.stt import STTService
from src.services.tts import TTSService
from src.services.search import SearchService
from src.ui.admin import create_admin_interface
from src.ui.chatbot import create_chatbot_interface

def main():
    """Initialize and launch the application"""
    
    print("="*60)
    print("Starting AI Shopping Assistant")
    print("="*60)
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n GPU: {gpu_name}")
        print(f" VRAM: {gpu_memory:.1f} GB")
    else:
        print("\n Running on CPU")
    
    # Initialize models and services
    print("\n1. Loading embedding models...")
    embedding_models = EmbeddingModels()
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"   GPU Memory used: {allocated:.2f} GB")
    
    print("\n2. Initializing Qdrant...")
    qdrant_service = QdrantService(embedding_models)
    qdrant_service.initialize_collection()
    
    print("\n3. Loading STT model...")
    stt_service = STTService()
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"   GPU Memory used: {allocated:.2f} GB")
    
    print("\n4. Initializing TTS service...")
    tts_service = TTSService()
    
    print("\n5. Creating search service...")
    search_service = SearchService(qdrant_service, tts_service)
    
    print("\n6. Building UI...")
    chatbot_ui = create_chatbot_interface(
        embedding_models,
        search_service,
        stt_service
    )
    admin_ui = create_admin_interface(qdrant_service)
    
    # Combine interfaces
    app = gr.TabbedInterface(
        [chatbot_ui, admin_ui],
        ["🤖 Assistant", "👨‍💼 Admin"],
        title="AI Shopping Assistant - Multimodal Search System",
        theme=gr.themes.Soft()
    )
    
    print("\n" + "="*60)
    print("Application ready! Launching web interface...")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total GPU Memory: {allocated:.2f} / {total:.1f} GB")
    
    print("="*60 + "\n")
    
    app.launch(
        share=True,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()