from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Configuration ---
BASE_MODEL_NAME = "microsoft/phi-2"

app = FastAPI(title="Direct AI Script Generator API")

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Model Variables ---
model = None
tokenizer = None
device = "cpu"

# --- Pydantic Models for the Simplified API ---
class ScriptRequest(BaseModel):
    prompt: str

class ScriptResponse(BaseModel):
    script: str | None = None
    error: str | None = None


# --- Application Startup Event: Load Model ---
@app.on_event("startup")
async def load_model_on_startup():
    global model, tokenizer, device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available. Using device: {torch.cuda.get_device_name(0)}")
        model_kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}
    else:
        device = "cpu"
        print("WARNING: CUDA not available. Loading model on CPU.")
        model_kwargs = {"torch_dtype": torch.float32, "device_map": "cpu", "trust_remote_code": True}
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **model_kwargs)
        model.eval()
        print(f"Model '{BASE_MODEL_NAME}' loaded successfully on {device}!")
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        model, tokenizer = None, None

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Direct AI Script Generator API is running!"}

# In backend/main.py

# In backend/main.py

@app.post("/generate-full-short-script/", response_model=ScriptResponse, tags=["Generation"])
async def generate_full_short_script_endpoint(request: ScriptRequest):
    """
    Generates a complete short film script using the fine-tuned model with a directive prompt.
    """
    if model is None: return ScriptResponse(error="Model not loaded.")

    # --- FINAL HYBRID PROMPT FOR YOUR FINE-TUNED MODEL ---
    # This prompt is structured for the fine-tuned model but is more explicit
    # to overcome its tendency to summarize. We give it the first word of the output.
    prompt = f"""<s>[INST] You are an expert screenwriter. Write a complete, multi-page script based on the user's prompt.
Your response MUST start with a list of characters, followed by '---', and then the full script from 'FADE IN:' to 'FADE OUT.'.

USER PROMPT: "{request.prompt}"

Begin the full response now. [/INST]
CHARACTERS
"""
    # --- END OF PROMPT ---

    try:
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device)
        
        # Keep the large token count for a long script.
        outputs = model.generate(
            **inputs,
            max_new_tokens=1950,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        num_prompt_tokens = inputs.input_ids.shape[1]
        generated_ids_only = outputs[0][num_prompt_tokens:]
        script_text = tokenizer.decode(generated_ids_only, skip_special_tokens=True).strip()

        # Prepend the "CHARACTERS" header since we provided it in the prompt
        # but the model's output starts right after it.
        final_script = "CHARACTERS\n" + script_text
        
        return ScriptResponse(script=final_script)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return ScriptResponse(error=str(e))