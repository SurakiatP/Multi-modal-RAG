"""
Admin panel UI - Modern English Version
"""
import gradio as gr
import uuid
import os
import torch
from pathlib import Path

from ..config import IMAGES_DIR
from ..database.product_loader import ProductLoader


def create_admin_interface(qdrant_service):
    """Create modern admin interface"""
    
    def add_product(product_code, name, description, price, location, image_files):
        """Add product to database"""
        if not all([product_code, name, description, price, location]):
            return "❌ Please fill in all required fields."
        
        uploaded_paths = []
        
        if image_files:
            for img_file in image_files:
                img_path = IMAGES_DIR / os.path.basename(img_file.name)
                
                with open(img_file.name, "rb") as temp_f:
                    with open(img_path, "wb") as f:
                        f.write(temp_f.read())
                
                uploaded_paths.append(str(img_path))
        
        new_product = {
            "id": str(uuid.uuid4()),
            "code": product_code,
            "name": name,
            "description": description,
            "price": price,
            "location": location,
            "image_paths": uploaded_paths
        }
        
        try:
            qdrant_service._add_product(new_product)
            products = ProductLoader.get_all_products()
            products.append(new_product)
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return f"✅ Product '{name}' added successfully!"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def clear_form():
        """Clear all form inputs"""
        return "", "", "", "", "", None, ""
    
    # Modern Admin CSS
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .admin-container {
        max-width: 1400px;
        margin: auto;
    }
    
    /* Modern Admin Header */
    .admin-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 50px 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .admin-header h1 {
        color: #fff;
        font-size: 2.8em;
        font-weight: 700;
        margin: 0 0 15px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .admin-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.2em;
        margin: 0;
        font-weight: 500;
    }
    
    /* Form Container */
    .form-container {
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 5px 25px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .form-title {
        color: #667eea;
        font-size: 1.6em;
        font-weight: 700;
        margin: 0 0 30px 0;
        padding-bottom: 20px;
        border-bottom: 3px solid #667eea;
        display: flex;
        align-items: center;
    }
    
    .form-title::before {
        content: "📦";
        margin-right: 15px;
        font-size: 1.3em;
    }
    
    /* Info Card */
    .info-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 3px 10px rgba(245, 158, 11, 0.15);
    }
    
    .info-card strong {
        color: #92400e;
        font-weight: 600;
    }
    
    /* Form Section Headers */
    .section-label {
        color: #4b5563;
        font-size: 0.95em;
        font-weight: 600;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Buttons */
    .btn-add {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 40px !important;
        font-size: 1.1em !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .btn-add:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    .btn-clear {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 40px !important;
        font-size: 1.1em !important;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 30px 0;
    }
    
    /* Status Box */
    .status-box {
        padding: 15px 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-weight: 500;
        text-align: center;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as admin_interface:
        
        with gr.Column(elem_classes="admin-container"):
            
            # Header
            gr.HTML("""
                <div class="admin-header">
                    <h1>👨‍💼 Admin Dashboard</h1>
                    <p>Manage Product Inventory</p>
                </div>
            """)
            
            # Info Card
            gr.HTML("""
                <div class="info-card">
                    <strong>📋 Instructions:</strong> 
                    Fill in all product details and upload images for better search accuracy.
                </div>
            """)
            
            # Form Container
            with gr.Column(elem_classes="form-container"):
                gr.HTML('<h2 class="form-title">Add New Product</h2>')
                
                # Section 1: Basic Information
                gr.HTML('<p class="section-label">📝 Basic Information</p>')
                with gr.Row():
                    product_code_input = gr.Textbox(
                        label="Product Code",
                        placeholder="e.g., P001, PROD-001",
                        scale=1
                    )
                    name_input = gr.Textbox(
                        label="Product Name",
                        placeholder="e.g., Fresh Milk, Bread",
                        scale=2
                    )
                    price_input = gr.Textbox(
                        label="Price",
                        placeholder="e.g., $3.50, 100 THB",
                        scale=1
                    )
                
                gr.HTML('<div class="divider"></div>')
                
                # Section 2: Details
                gr.HTML('<p class="section-label">📄 Product Details</p>')
                with gr.Row():
                    description_input = gr.Textbox(
                        label="Description",
                        placeholder="e.g., 100% pasteurized fresh milk",
                        lines=4
                    )
                    location_input = gr.Textbox(
                        label="Storage Location",
                        placeholder="e.g., Fridge Zone A, Shelf 1",
                        lines=4
                    )
                
                gr.HTML('<div class="divider"></div>')
                
                # Section 3: Images
                gr.HTML('<p class="section-label">🖼️ Product Images</p>')
                gr.HTML('<p style="color: #6b7280; font-size: 0.9em; margin-bottom: 10px;">Upload multiple images (JPG, PNG supported)</p>')
                image_files_input = gr.Files(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=["image"]
                )
                
                gr.HTML('<div class="divider"></div>')
                
                # Action Buttons
                with gr.Row():
                    add_button = gr.Button(
                        "✅ Add Product",
                        variant="primary",
                        elem_classes="btn-add",
                        scale=3,
                        size="lg"
                    )
                    clear_button = gr.Button(
                        "🗑️ Clear Form",
                        elem_classes="btn-clear",
                        scale=1,
                        size="lg"
                    )
                
                # Status Output
                output_message = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                    elem_classes="status-box"
                )
        
        # Event Handlers
        add_button.click(
            fn=add_product,
            inputs=[
                product_code_input,
                name_input,
                description_input,
                price_input,
                location_input,
                image_files_input
            ],
            outputs=output_message
        )
        
        clear_button.click(
            fn=clear_form,
            outputs=[
                product_code_input,
                name_input,
                description_input,
                price_input,
                location_input,
                image_files_input,
                output_message
            ]
        )
    
    return admin_interface