import streamlit as st
import random, time, os, requests
from huggingface_hub import InferenceClient
from pypdf import PdfReader
import re
from fuzzywuzzy import fuzz
import base64
from io import BytesIO
import json
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PDFChat')


# ----------- Upload Folder Setup -----------
upload_folder = 'uploaded_pdf_file'

if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

# ----------- Streamlit UI: Header and Sidebar -----------
st.header("PDF Chatbot")

# Sidebar for model selection and usage guide
st.sidebar.title("Model Settings")
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Local Ollama"

selected_model = st.sidebar.selectbox(
    "Choose AI Model",
    ["Local Ollama"],
    index=0,
    key="model_selector"
)
st.session_state.selected_model = selected_model

# Add usage guide in sidebar
st.sidebar.markdown("---")
st.sidebar.title("How to Use This System")

st.sidebar.markdown("""
### 📚 General Usage
1. **Ask Questions**: Type your question in the chat box
2. **Upload PDFs**: Use the file uploader to add PDF documents
3. **Choose Model**: Select your preferred AI model from the dropdown

### 🚗 Car Recommendations
- For students: Budget limit of RM 50,000
- All recommendations include specific prices
- Recommendations are based on PDF content

### ⚙️ Model Options
- **Local Ollama**: 
  - llama3.1:8b: More powerful but slower
  - deepseek-r1:1.5b: Faster responses

### 💡 Tips
- Upload PDFs for specific document-based answers
- Ask general questions without uploading files
- Be specific in your questions for better answers
- Check the model status in the sidebar
""")

# ----------- Ollama Connection Check -----------
if st.session_state.selected_model == "Local Ollama":
    try:
        response = requests.get("http://127.0.0.1:11434/api/version", timeout=2)
        if response.status_code == 200:
            st.sidebar.success(f"✅ Ollama connected: {response.json().get('version', 'unknown version')}")
            ollama_models_available = True
        else:
            st.sidebar.error("❌ Ollama server responded with an error. Status code: " + str(response.status_code))
            ollama_models_available = False
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"❌ Could not connect to Ollama server: {str(e)}")
        st.sidebar.info("Make sure Ollama is running with 'ollama serve' command")
        ollama_models_available = False

    # Ollama model selection with available models
    if 'ollama_model' not in st.session_state:
        st.session_state.ollama_model = "llama3:8b"
    ollama_model = st.sidebar.selectbox(   
        "Choose Ollama Model",
        ["llama3:8b", "deepseek-r1:1.5b", "qwen3:1.7b"], 
        index=0,
        key="ollama_model_selector"
    )
    st.session_state.ollama_model = ollama_model
        
    # Add model information
    if st.session_state.ollama_model == "llama3:8b":
        st.sidebar.info("Using llama3:8b - Powerful model with excellent reasoning capabilities")
        st.sidebar.warning("This is a large model and may take longer to process. Please be patient.", icon="⚠️")
    else:
        st.sidebar.info("Using deepseek-r1:1.5b - Fast and efficient model for quick responses")
    
    # Add a refresh button to test connection again
    if st.sidebar.button("Test Ollama Connection"):
        try:
            response = requests.get("http://127.0.0.1:11434/api/version", timeout=2)
            if response.status_code == 200:
                st.sidebar.success(f"✅ Ollama connected: {response.json().get('version', 'unknown version')}")
                try:
                    model_response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
                    if model_response.status_code == 200:
                        models = model_response.json().get('models', [])
                        model_names = [model.get('name') for model in models]
                        if st.session_state.ollama_model in model_names:
                            st.sidebar.success(f"✅ Model '{st.session_state.ollama_model}' is available!")
                        else:
                            st.sidebar.warning(f"⚠️ Model '{st.session_state.ollama_model}' not found. Available models: {', '.join(model_names)}")
                            st.sidebar.info(f"Try: ollama pull {st.session_state.ollama_model}")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Could not check model availability: {str(e)}")
            else:
                st.sidebar.error("❌ Ollama server responded with an error")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"❌ Could not connect to Ollama server: {str(e)}")
            st.sidebar.info("Make sure Ollama is running with 'ollama serve' command")




# ----------- PDF Upload and Text Extraction -----------
def clean_text(text):
    """Clean and format extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction issues
    text = re.sub(r'-\s*\n\s*', '', text)  # Fix hyphenated words
    text = re.sub(r'\n\s*(?=[a-z])', ' ', text)  # Fix line breaks in sentences
    text = re.sub(r'\n\s*(?=[A-Z])', '. ', text)  # Fix line breaks between sentences
    
    # Remove headers and footers (common in PDFs)
    text = re.sub(r'\n\d+\s*\n', '\n', text)  # Remove page numbers
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)  # Remove page numbers
    text = re.sub(r'©.*?\n', '', text)  # Remove copyright notices
    text = re.sub(r'Confidential.*?\n', '', text)  # Remove confidentiality notices
    
    # Clean up remaining whitespace
    text = text.strip()
    
    return text

def extract_text_from_pdf(pdf_path):
    """Enhanced PDF text extraction with better formatting and error handling"""
    try:
        reader = PdfReader(pdf_path)
        number_of_pages = len(reader.pages)
        
        # Extract text from all pages with metadata
        pdf_text = ""
        page_metadata = []
        
        for i in range(number_of_pages):
            page = reader.pages[i]
            
            # Extract text
            page_text = page.extract_text()
            
            # Clean the extracted text
            cleaned_text = clean_text(page_text)
            
            # Extract metadata
            metadata = {
                'page_number': i + 1,
                'page_size': page.mediabox,
                'has_images': len(page.images) > 0,
                'text_length': len(cleaned_text)
            }
            
            # Add page separator and metadata
            pdf_text += f"\n\n--- Page {i + 1} ---\n\n{cleaned_text}"
            page_metadata.append(metadata)
        
        return pdf_text, page_metadata
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return "", []

uploaded_files = st.file_uploader("Choose PDF files", type=['pdf','PDF'], accept_multiple_files=True)

extracted_text = ""
if uploaded_files:
    # Create a list to store all PDF texts and metadata
    all_pdf_texts = []
    all_metadata = []
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        saved_path = os.path.join(upload_folder, file_name)

        with open(saved_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"PDF file '{file_name}' has successfully uploaded to {saved_path}")

        # Extract text and metadata
        pdf_text, metadata = extract_text_from_pdf(saved_path)
        
        if pdf_text:
            all_pdf_texts.append(pdf_text)
            all_metadata.append({
                'filename': file_name,
                'pages': metadata,
                'total_pages': len(metadata),
                'has_images': any(page['has_images'] for page in metadata)
            })
            
            # Display document summary
            with st.expander(f"Document Summary: {file_name}"):
                st.write(f"Total Pages: {len(metadata)}")
                st.write(f"Contains Images: {'Yes' if any(page['has_images'] for page in metadata) else 'No'}")
                st.write(f"Average Text per Page: {sum(page['text_length'] for page in metadata) // len(metadata)} characters")
                
                # Show first page preview
                st.write("First Page Preview:")
                st.write(pdf_text.split("--- Page 1 ---")[1].split("--- Page 2 ---")[0] if "--- Page 2 ---" in pdf_text else pdf_text)
        else:
            st.warning(f"Could not extract text from {file_name}")
    
    # Combine all PDF texts with clear separation
    if all_pdf_texts:
        extracted_text = "\n\n--- Document Separator ---\n\n".join(all_pdf_texts)
        
        # Show summary of uploaded documents
        st.info(f"Successfully processed {len(uploaded_files)} PDF documents with a total of {sum(meta['total_pages'] for meta in all_metadata)} pages.")
        
        # Store metadata in session state for later use
        st.session_state.pdf_metadata = all_metadata
    else:
        st.error("No text could be extracted from any of the uploaded PDFs.")


# ----------- Response Enhancement Function -----------
def enhance_response(answer, source_doc=None, page_number=None):
    """Enhanced response formatting with source attribution"""
    # Add source information if available
    source_info = ""
    if source_doc:
        source_info = f"Based on document '{source_doc}'"
        if page_number:
            source_info += f", page {page_number}"
        source_info += ":\n\n"
    
    # Format the response
    formatted_response = f"{source_info}{answer}"
    
    return formatted_response

# ----------- Fuzzy Matching Logic for Context Extraction -----------
def fuzzy_match_query(text, query):
    key_terms = re.findall(r'\b\w+\b', query.lower())
    key_terms = [term for term in key_terms if len(term) > 3]
    
    # Split text into documents
    documents = text.split("\n\n--- Document Separator ---\n\n")
    best_paragraph = ""
    highest_score = 0
    source_doc = None
    page_number = None
    
    for doc_idx, doc_text in enumerate(documents):
        # Split document into pages
        pages = re.split(r'--- Page \d+ ---', doc_text)
        for page_idx, page_text in enumerate(pages[1:], 1):  # Skip first empty split
            paragraphs = page_text.split('\n\n')
            for paragraph in paragraphs:
                if len(paragraph.strip()) < 10:
                    continue
                score = 0
                for term in key_terms:
                    for word in re.findall(r'\b\w+\b', paragraph.lower()):
                        if len(word) > 3:
                            match_score = fuzz.ratio(term, word)
                            if match_score > 70:
                                score += match_score
                if score > highest_score:
                    highest_score = score
                    best_paragraph = paragraph
                    source_doc = uploaded_files[doc_idx].name if uploaded_files else None
                    page_number = page_idx
    
    # Check if we found any relevant content
    if highest_score == 0:
        return "", ["content not found in PDF"], None, None
    
    context = best_paragraph
    missing_info = []
    if "when" in query.lower() and not re.search(r'\b(date|day|month|year|time)\b', query, re.IGNORECASE):
        missing_info.append("time period")
    if "where" in query.lower() and not re.search(r'\b(location|place|address|city)\b', query, re.IGNORECASE):
        missing_info.append("location")
        
    return context, missing_info, source_doc, page_number

# ----------- Content Validation Function -----------
def validate_response(response, pdf_content):
    """Check if the response contains information that's likely not from the PDF"""
    # Common phrases that indicate external knowledge
    external_indicators = [
        "generally speaking", "in general", "typically", "usually", 
        "as a rule", "in most cases", "according to experts", 
        "research shows", "studies indicate", "it is known that",
        "it is common knowledge", "it is widely accepted", "it is a fact that"
    ]
    
    # Check for external knowledge indicators
    for indicator in external_indicators:
        if indicator.lower() in response.lower():
            return False, f"The response contains phrases like '{indicator}' which suggest it's using external knowledge rather than PDF content only."
    
    # Check if response length is too short to be meaningful
    if len(response.strip()) < 20:
        return False, "The response is too short to be meaningful."
    
    # Check if response contains "I cannot find" which is our indicator for content not in PDF
    if "cannot find" in response.lower() or "not found in the pdf" in response.lower():
        return True, "The response correctly indicates that information is not in the PDF."
    
    # Check for key terms from the PDF content
    pdf_terms = re.findall(r'\b\w{5,}\b', pdf_content.lower())
    pdf_terms = [term for term in pdf_terms if term not in ["about", "which", "their", "there", "these", "those", "would", "could", "should"]]
    
    # Count how many PDF terms appear in the response
    pdf_term_count = 0
    for term in pdf_terms[:20]:  # Limit to first 20 terms to avoid excessive checking
        if term in response.lower():
            pdf_term_count += 1
    
    # If response has very few PDF terms, it might be using external knowledge
    if pdf_term_count < 2 and len(response.split()) > 20:
        return False, "The response contains very few terms from the PDF content, suggesting it might be using external knowledge."
    
    return True, "The response appears to be based on PDF content."

# ----------- PDF Image Extraction for Multimodal Use -----------
def get_pdf_image(pdf_path):
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        if images:
            buffered = BytesIO()
            images[0].save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        return None
    except Exception as e:
        st.error(f"Error extracting PDF image: {e}")
        return None



# ----------- Ollama API Communication Logic -----------
def call_ollama_api(prompt, context, model="llama3.1:8b", pdf_path=None):
    """Call the Ollama API with the specified model"""
    API_URL = "http://127.0.0.1:11434/api/chat"
    
    logger.info(f"Calling Ollama API with model: {model}")
    
    # Prepare messages with balanced instructions
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that primarily uses the provided content to answer questions, while also providing relevant additional information when helpful.

You have access to the following context: {context}

Your task is to:
1. First, analyze the content carefully
2. Base your answer primarily on the information found in the content
3. If the content is limited, you can supplement with relevant general knowledge
4. Clearly indicate which information comes from the content and which is additional context
5. Structure your response in a clear and organized manner
6. Use bullet points or numbered lists for better readability
7. For car recommendations, always include specific prices if available.
8. Only recommend models that are in the content, no need to add on any other models.
9. For the price in PDF, usually it is in format of RM XXX,XXX.00 , so please get the entire price, not just the last 3 digits.
10. If the price is not specified in the content, please say "The price is not specified in the content"
11. If mention about model, please include all model that can be found in the content.
12. normally the price will be mentioned after "priced at" or "starting from" or "from" or "from RM" or "from RM"
13. Premium class car will be more than RM 500,000.

Remember: The content is your main source, but you can enhance the response with relevant additional information when it helps provide a more complete answer."""
        },
        {
            "role": "user",
            "content": f"Please provide an answer based primarily on the content, supplemented with relevant additional information if needed. {prompt}"
        }
    ]
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.9,
            "top_p": 0.8,
            "num_predict": 2048
        }
    }
    
    try:
        logger.info(f"Sending request to Ollama API at {API_URL}")
        # Adjust timeout based on model size
        timeout = 180 if model == "llama3.1:8b" else 60
        response = requests.post(API_URL, json=payload, timeout=(timeout, timeout))
        logger.info(f"Ollama API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info("Received successful response from Ollama")
            
            # Handle different API response formats
            if "message" in data:
                response_text = data["message"]["content"]
            elif "response" in data:
                response_text = data["response"]
            else:
                logger.warning("Unexpected response format from Ollama")
                return "No response from Ollama"
            
            return response_text
        else:
            error_msg = f"Ollama API error: {response.status_code}"
            logger.error(error_msg)
            if response.text:
                logger.error(f"Response text: {response.text}")
            st.error(error_msg)
            return f"Error: {response.status_code}"
    except requests.exceptions.ConnectTimeout:
        error_msg = "Connection timeout when connecting to Ollama server. Make sure it's running."
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg
    except requests.exceptions.ReadTimeout:
        error_msg = f"Request timed out after {timeout} seconds. The {model} model may be too large for your system or taking too long to process. Try using a smaller model like deepseek-r1:1.5b."
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to Ollama server at 127.0.0.1:11434. Make sure it's running with 'ollama serve'."
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error calling Ollama: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg


# ----------- Central Response Generator with Model Fallback Logic -----------
def response_generator(text, prompt, pdf_path=None):
    # Check if there's any PDF content
    if not text or text.strip() == "":
        # Handle general questions without PDF context
        if st.session_state.selected_model == "Local Ollama":
            try:
                logger.info(f"Using Ollama model: {st.session_state.ollama_model} for general question")
                st.info("Using local Ollama model for general question.")
                
                if st.session_state.ollama_model == "llama3.1:8b":
                    st.warning("Using the larger llama3.1:8b model. This may take longer to process. Please be patient.", icon="⚠️")
                
                # Prepare general question prompt
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that provides informative and detailed answers to questions. Be natural and conversational in your responses."},
                    {"role": "user", "content": prompt}
                ]
                
                payload = {
                    "model": st.session_state.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 1024
                    }
                }
                
                response = requests.post(
                    "http://127.0.0.1:11434/api/chat",
                    json=payload,
                    timeout=180 if st.session_state.ollama_model == "llama3.1:8b" else 60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "message" in data:
                        return {"answer": data["message"]["content"]}
                    elif "response" in data:
                        return {"answer": data["response"]}
            except Exception as e:
                st.warning(f"Error with Ollama: {e}. Falling back to Ollama.", icon="⚠️")
                logger.error(f"Exception in Ollama call: {str(e)}")
        
        # Try Ollama for general questions
        try:
            logger.info("Using Ollama model for general question")
            st.info("Using local Ollama model for general question.")
            
            ollama_response = call_ollama_api(prompt, "", st.session_state.ollama_model, pdf_path)
            if ollama_response and not ollama_response.startswith("Error:"):
                return {"answer": enhance_response(ollama_response)}
            else:
                st.warning("No valid response from Ollama", icon="⚠️")
                logger.warning(f"Invalid Ollama response: {ollama_response}")
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")
            return {"answer": "I apologize, but I encountered an error while processing your request. Please try again later."}
    
    # Handle PDF-based questions (existing logic)
    context, missing_info, source_doc, page_number = fuzzy_match_query(text, prompt)
    
    # Check if the question is not within the PDF content
    if "content not found in PDF" in missing_info:
        return {
            "answer": "I'd be happy to help with that! While I don't have specific information about this in the uploaded documents, I can provide some general information on the topic.",
            "needs_info": True
        }
    
    # Ask for more info if needed
    if missing_info and len(missing_info) > 0:
        return {
            "answer": f"I'd like to help, but I need more information about the {', '.join(missing_info)} to give you a proper answer. Could you please provide more details?",
            "needs_info": True
        }
    
    # Use selected model for PDF-based questions
    if st.session_state.selected_model == "Local Ollama":
        try:
            logger.info(f"Using Ollama model: {st.session_state.ollama_model}")
            st.info("Using local Ollama model. Responses will be based on the PDF content.")
            
            if st.session_state.ollama_model == "llama3.1:8b":
                st.warning("Using the larger llama3.1:8b model. This may take longer to process. Please be patient.", icon="⚠️")
            
            ollama_response = call_ollama_api(prompt, context, st.session_state.ollama_model, pdf_path)
            if ollama_response and not ollama_response.startswith("Error:") and not ollama_response.startswith("Could not connect"):
                return {"answer": enhance_response(ollama_response)}
            else:
                st.warning("No valid response from Ollama", icon="⚠️")
                logger.warning(f"Invalid Ollama response: {ollama_response}")
        except Exception as e:
            st.warning(f"Error with Ollama: {e}. Falling back to Ollama.", icon="⚠️")
            logger.error(f"Exception in Ollama call: {str(e)}")
    
    return {"answer": "I apologize, but I couldn't process your request at this time. Please try again later."}

# ----------- Streamlit Chat UI and Interaction Logic -----------
st.title("PDF Chat Assistant")

# Add initial greeting only if not already added
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm your PDF Chat Assistant. How can I help you today? 😊"
    })

# Render chat history (ONLY ONCE)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Response logic
    lower_prompt = prompt.lower()
    if "hello" in lower_prompt or "hi" in lower_prompt:
        response_text = "Hi there! How can I assist you today?"
    elif "how are you" in lower_prompt:
        response_text = "I'm just a program, but I'm here to help you!"
    elif "help" in lower_prompt:
        response_text = "You can ask me anything about the PDFs you upload."
    elif "bye" in lower_prompt:
        response_text = "Goodbye! Have a great day! 😊"
    else:
        pdf_path = None
        if uploaded_files and len(uploaded_files) > 0:
            pdf_path = os.path.join(upload_folder, uploaded_files[0].name)

        with st.spinner("Processing your question..."):
            response = response_generator(extracted_text if extracted_text else "", prompt, pdf_path)

        response_text = response['answer']

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})

#view extracted text
if extracted_text:
    with st.expander("View Extracted Text"):
        st.text_area("Extracted Text", extracted_text, height=300)
