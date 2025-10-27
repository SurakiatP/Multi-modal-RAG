"""
Chatbot UI for customers - Modern English Version
"""
import gradio as gr
from typing import Iterator, Tuple, List
import torch

from ..config import AVAILABLE_OLLAMA_MODELS
from ..services.stt import STTService
from ..services.search import SearchService

def create_chatbot_interface(
    embedding_models,
    search_service: SearchService,
    stt_service: STTService
):
    """Create modern chatbot interface"""
  
    def chatbot_response(
        text_query,
        audio_input,
        image_input,
        selected_llm_model
    ) -> Iterator[Tuple[List, str, str]]:
        """Main chatbot response function"""
        
        # Clear GPU cache at start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        yield [], None, ""
        
        if not selected_llm_model:
            yield [], None, "⚠️ Please select an LLM Model from the dropdown first."
            return
        
        query = ""
        query_embedding = None
        vector_name = "text"
        
        # Priority: Audio > Image > Text
        if audio_input:
            yield [], None, "🎤 Transcribing audio..."
            query = stt_service.transcribe(audio_input)
            if not query:
                yield [], None, "❌ Sorry, couldn't transcribe the audio. Please try again."
                return
            print(f"Speech to text: {query}")
            vector_name = "text"
            yield [], None, f"💬 Query: {query}\n\n🔍 Searching..."
        
        elif image_input:
            yield [], None, "🖼️ Processing image..."
            query_embedding = embedding_models.get_image_embedding([image_input])
            vector_name = "image"
            query = "[Image Query]"
            print("Image query received")
            yield [], None, "🔍 Searching for similar products..."
        
        elif text_query:
            query = text_query
            vector_name = "text"
            yield [], None, f"🔍 Searching for: {query}..."
        
        else:
            yield [], None, "💡 Please enter text, record audio, or upload an image to search."
            return
        
        # Generate embedding for text
        if vector_name == "text" and query:
            query_embedding = embedding_models.get_text_embedding(query, is_query=True)
        
        if query_embedding is None:
            yield [], None, "❌ Failed to generate embedding. Please try again."
            return

        # Stream response
        for response_text, images, audio in search_service.search_and_respond(
            query,
            query_embedding,
            vector_name,
            selected_llm_model
        ):
            yield images, audio, response_text
        
        # Clear GPU cache after completion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_settings_change():
        """Auto-reset when settings change"""
        return None, None, "", []
    
    # Modern Dark Theme CSS
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .gradio-container {
        max-width: 1600px !important;
        margin: auto;
    }
    
    /* Dark Modern Header */
    .modern-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 40px 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .modern-header h1 {
        color: #fff;
        font-size: 2.8em;
        font-weight: 700;
        margin: 0 0 15px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .modern-header .subtitle {
        color: #00d4ff;
        font-size: 1.1em;
        font-weight: 500;
        margin: 0;
    }
    
    .modern-header .badge {
        display: inline-block;
        background: rgba(0, 212, 255, 0.2);
        color: #00d4ff;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 0.9em;
        margin-top: 15px;
        border: 1px solid #00d4ff;
    }
    
    /* Chat Container */
    .chat-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        min-height: 600px;
    }
    
    /* Input Panel */
    .input-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 25px;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .input-panel h3 {
        color: #1a1a2e;
        font-size: 1.2em;
        font-weight: 600;
        margin: 0 0 20px 0;
        display: flex;
        align-items: center;
    }
    
    .input-panel h3::before {
        content: "⚙️";
        margin-right: 10px;
        font-size: 1.3em;
    }
    
    /* Output Panel */
    .output-panel {
        background: #ffffff;
    }
    
    .output-panel h3 {
        color: #0f3460;
        font-size: 1.3em;
        font-weight: 600;
        margin: 0 0 20px 0;
        padding-bottom: 15px;
        border-bottom: 3px solid #00d4ff;
        display: flex;
        align-items: center;
    }
    
    .output-panel h3::before {
        content: "🤖";
        margin-right: 10px;
        font-size: 1.4em;
    }
    
    /* Buttons */
    .primary-btn {
        background: linear-gradient(135deg, #00d4ff 0%, #0096c7 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 35px !important;
        font-size: 1.05em !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .primary-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4) !important;
    }
    
    .secondary-btn {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 35px !important;
        font-size: 1.05em !important;
    }
    
    /* Info Banner */
    .info-banner {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 3px 10px rgba(33, 150, 243, 0.1);
    }
    
    .info-banner strong {
        color: #1565c0;
        font-weight: 600;
    }
    
    /* Gallery */
    .gallery-modern {
        border-radius: 15px !important;
        overflow: hidden !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    /* Status Messages */
    .status-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: 500;
    }
    
    .status-loading {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #856404;
    }
    
    .status-success {
        background: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    
    .status-error {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        margin: 25px 0;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as chatbot_interface:
        
        # Modern Header
        gr.HTML("""
            <div class="modern-header">
                <h1>🤖 AI Product Assistant</h1>
                <p class="subtitle">Search with Voice • Image • Text</p>
                <span class="badge">Powered by Multi-modal AI</span>
            </div>
        """)
        
        # Info Banner
        gr.HTML("""
            <div class="info-banner">
                <strong>💡 How to use:</strong> 
                Type your question, speak into the microphone, or upload a product image to search our inventory.
            </div>
        """)
        
        # Main Layout - Two Column
        with gr.Row():
            
            # LEFT: Input Panel
            with gr.Column(scale=1):
                with gr.Group(elem_classes="input-panel"):
                    gr.HTML('<h3>Input Settings</h3>')
                    
                    # Model Selection
                    llm_model = gr.Dropdown(
                        label="🧠 AI Model",
                        choices=AVAILABLE_OLLAMA_MODELS,
                        value=AVAILABLE_OLLAMA_MODELS[0] if AVAILABLE_OLLAMA_MODELS else None,
                        interactive=True,
                        container=True
                    )
                    
                    gr.HTML('<div class="divider"></div>')
                    
                    # Text Input
                    text_input = gr.Textbox(
                        label="💬 Text Query",
                        placeholder="e.g., Do you have fresh milk? What's the price?",
                        lines=3,
                        max_lines=5
                    )
                    
                    # Audio Input
                    audio_input = gr.Audio(
                        type="filepath",
                        sources=["microphone"],
                        label="🎤 Voice Query",
                        format="wav"
                    )
                    
                    # Image Input
                    image_input = gr.Image(
                        type="filepath",
                        label="📷 Image Query",
                        height=250,
                        sources=["upload", "clipboard"]
                    )
                    
                    gr.HTML('<div class="divider"></div>')
                    
                    # Action Buttons
                    with gr.Row():
                        submit_btn = gr.Button(
                            "🔍 Search",
                            variant="primary",
                            elem_classes="primary-btn",
                            scale=2
                        )
                        clear_btn = gr.Button(
                            "🔄 Reset",
                            elem_classes="secondary-btn",
                            scale=1
                        )
            
            # RIGHT: Output Panel
            with gr.Column(scale=2):
                with gr.Group(elem_classes="output-panel"):
                    gr.HTML('<h3>AI Response</h3>')
                    
                    # Response Text
                    response_output = gr.Markdown(
                        label="Answer",
                        value="*Waiting for your query...*",
                        container=True,
                        height=250
                    )
                    
                    gr.HTML('<div class="divider"></div>')
                    
                    # Image Gallery
                    image_gallery = gr.Gallery(
                        label="📦 Product Images",
                        columns=3,
                        rows=2,
                        height=350,
                        object_fit="contain",
                        elem_classes="gallery-modern"
                    )
                    
                    # Audio Response
                    audio_output = gr.Audio(
                        label="🔊 Voice Response",
                        type="filepath",
                        autoplay=False,
                        show_download_button=True
                    )
        
        # Event Handlers
        inputs = [text_input, audio_input, image_input, llm_model]
        outputs = [image_gallery, audio_output, response_output]
        
        # Search action
        submit_btn.click(
            fn=chatbot_response,
            inputs=inputs,
            outputs=outputs
        )
        
        text_input.submit(
            fn=chatbot_response,
            inputs=inputs,
            outputs=outputs
        )
        
        # Clear action
        def clear_all():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return (
                None,  # audio_input
                None,  # image_input
                "",    # text_input
                [],    # image_gallery
                None,  # audio_output
                "*Ready for new query...*"  # response_output
            )
        
        clear_btn.click(
            fn=clear_all,
            outputs=[
                audio_input,
                image_input,
                text_input,
                image_gallery,
                audio_output,
                response_output
            ]
        )
        
        # Auto-reset on model change
        llm_model.change(
            fn=on_settings_change,
            outputs=[audio_input, image_input, text_input, image_gallery]
        )
    
    return chatbot_interface