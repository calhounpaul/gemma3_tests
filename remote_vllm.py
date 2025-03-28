import gradio as gr
import requests
import json
import base64
import os
from io import BytesIO
from openai import OpenAI

# Get vLLM endpoint from environment variable
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://0.0.0.0:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-3-27b-it")

def convert_files_to_base64(files):
    """Convert uploaded files to base64 strings"""
    base64_images = []
    for file in files:
        with open(file, "rb") as image_file:
            # Read image data and encode to base64
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            base64_images.append(base64_data)
    return base64_images

def get_openai_client():
    """Create and return an OpenAI client configured for the vLLM endpoint"""
    return OpenAI(
        api_key="EMPTY",  # vLLM doesn't require an actual API key
        base_url=VLLM_ENDPOINT,
    )

def process_chat(message_dict, history):
    """Process user message and send to vLLM API via OpenAI client"""
    text = message_dict.get("text", "")
    files = message_dict.get("files", [])
    
    # Add user message to history first
    if not history:
        history = []
    
    # Add user message to chat history
    if files:
        # For each file, add a separate user message
        for file in files:
            history.append({"role": "user", "content": (file,)})
    
    # Add text message if not empty
    if text.strip():
        history.append({"role": "user", "content": text})
    else:
        # If no text but files exist, don't add an empty message
        if not files:
            history.append({"role": "user", "content": ""})
    
    # Convert all files to base64
    base64_images = convert_files_to_base64(files)
    
    # Prepare conversation history in OpenAI format
    openai_messages = []
    
    # Convert history to OpenAI format
    for h in history:
        if h["role"] == "user":
            # Handle user messages
            if isinstance(h["content"], tuple):
                # This is a file-only message, skip for now
                continue
            else:
                # Text message
                openai_messages.append({
                    "role": "user",
                    "content": h["content"]
                })
        elif h["role"] == "assistant":
            openai_messages.append({
                "role": "assistant",
                "content": h["content"]
            })
    
    # Handle images for the last user message if needed
    if base64_images:
        # Update the last user message to include image content
        if openai_messages and openai_messages[-1]["role"] == "user":
            # Get the last message
            last_msg = openai_messages[-1]
            
            # Format for OpenAI multimodal content structure
            content_list = []
            
            # Add text if there is any
            if last_msg["content"]:
                content_list.append({"type": "text", "text": last_msg["content"]})
            
            # Add images
            for img_b64 in base64_images:
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                })
            
            # Replace the content with the multimodal content list
            last_msg["content"] = content_list
    
    try:
        # Use the OpenAI client to send the request
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=openai_messages,
        )
        
        # Extract the assistant response
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        history.append({"role": "assistant", "content": assistant_message})
        return history
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        history.append({"role": "assistant", "content": error_msg})
        return history

def add_text(chatbot, text):
    """Add text from user to the chatbot"""
    if not chatbot:
        chatbot = []
    chatbot.append({"role": "user", "content": text})
    return chatbot

def bot_response(history):
    """Process the last user message and generate a response"""
    if not history:
        return history
        
    # Find the last user message
    last_user_idx = None
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "user":
            last_user_idx = i
            break
    
    if last_user_idx is None:
        return history
    
    # Extract the last user message
    user_message = history[last_user_idx]
    
    # Process OpenAI messages without this user message
    openai_messages = []
    for i, h in enumerate(history):
        if i < last_user_idx:  # Only include messages before this one
            if h["role"] == "user":
                if not isinstance(h["content"], tuple):  # Skip file messages
                    openai_messages.append({
                        "role": "user",
                        "content": h["content"]
                    })
            elif h["role"] == "assistant":
                openai_messages.append({
                    "role": "assistant",
                    "content": h["content"]
                })
    
    # Add this user message
    message_content = ""
    image_content = None
    
    if isinstance(user_message["content"], tuple):
        # This is a file message
        file_path = user_message["content"][0]
        with open(file_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_data}"
                }
            }
    else:
        # This is a text message
        message_content = user_message["content"]
    
    # Format the user message with images if present
    if image_content:
        user_content = []
        if message_content:
            user_content.append({"type": "text", "text": message_content})
        user_content.append(image_content)
        
        openai_messages.append({
            "role": "user",
            "content": user_content
        })
    else:
        openai_messages.append({
            "role": "user",
            "content": message_content
        })
    
    try:
        # Use the OpenAI client to send the request
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=openai_messages,
        )
        
        # Extract the assistant response
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        history.append({"role": "assistant", "content": assistant_message})
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        history.append({"role": "assistant", "content": error_msg})
    
    return history

# Create Gradio application with Blocks for more control
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# Multimodal vLLM Chat Interface")
    gr.Markdown(f"Chat with [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) via the vLLM API.")
    
    # Create chatbot component with message type
    chatbot = gr.Chatbot(
        label="Conversation",
        type="messages",
        show_copy_button=True,
        avatar_images=("👤", None),
        height=500
    )
    
    # Create multimodal textbox for input
    with gr.Row():
        textbox = gr.MultimodalTextbox(
            file_types=["image", "video"],
            file_count="multiple",
            placeholder="Type your message here and/or upload images...",
            label="Message",
            show_label=False,
            scale=9
        )
        submit_btn = gr.Button("Send", size="sm", scale=1)
    
    # Clear button
    clear_btn = gr.Button("Clear Chat")
    
    # Set up submit event chain
    submit_event = textbox.submit(
        fn=process_chat,
        inputs=[textbox, chatbot],
        outputs=chatbot
    ).then(
        fn=lambda: {"text": "", "files": []},
        inputs=None,
        outputs=textbox
    )
    
    # Connect the submit button to the same functions
    submit_btn.click(
        fn=process_chat,
        inputs=[textbox, chatbot],
        outputs=chatbot
    ).then(
        fn=lambda: {"text": "", "files": []},
        inputs=None,
        outputs=textbox
    )
    
    # Set up clear button
    clear_btn.click(lambda: [], None, chatbot)
    
    # Load example dog image
    dog_img_path = os.path.join(os.path.dirname(__file__), "dog_pic.jpg")
    thing_img_path = os.path.join(os.path.dirname(__file__), "ghostimg.png")
    newspaper_img_path=os.path.join(os.path.dirname(__file__), "newspaper.png")
    
    # Add examples with image
    if os.path.exists(dog_img_path):
        gr.Examples(
            examples=[
                [{"text": "What breed is this?", "files": [dog_img_path]}],
                [{"text": "What's in this image?", "files": [thing_img_path]}],
                [{"text": "Transcribe everything in this newspaper page.", "files": [newspaper_img_path]}],
            ],
            inputs=textbox
        )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0",server_port=7862)