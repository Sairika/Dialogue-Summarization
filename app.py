import gradio as gr
from transformers import AutoTokenizer
from peft import AutoPeftModelForSeq2SeqLM
import torch

# Model configuration
repo_id = "sairika/FLAN-T5-Base-dialogsum-lora"

# Sample dialogues
examples = [
    "Alice: How was your meeting today?\nBob: It went great! The client approved our proposal.\nAlice: That's wonderful news!\nBob: Yes, we start next Monday.",
    
    "Customer: My internet isn't working.\nSupport: I can help with that. Have you tried restarting your router?\nCustomer: Yes, but it's still not working.\nSupport: Let me check your connection status.",
    
    "Teacher: Did you finish your homework?\nStudent: Almost done, just the math problems left.\nTeacher: Need help with anything?\nStudent: Yes, I'm stuck on question 5."
]

def load_model():
    """Load the model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(
            repo_id, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
        return tokenizer, model, True
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, False

def summarize_dialogue(dialogue):
    """Generate summary for the dialogue"""
    if not dialogue.strip():
        return "Please enter a dialogue to summarize."
    
    if not model_loaded:
        return "Error: Model not loaded. Please check the logs."
    
    try:
        # Create prompt
        prompt = f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: "
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate summary
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode and return
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Load model
tokenizer, model, model_loaded = load_model()

# Create Gradio interface
with gr.Blocks(title="Dialogue Summarization") as demo:
    gr.Markdown("# 🤖 Dialogue Summarization")
    gr.Markdown("Enter a conversation below to generate an AI summary using FLAN-T5.")
    
    with gr.Row():
        with gr.Column():
            dialogue_input = gr.Textbox(
                label="Dialogue",
                placeholder="Enter your conversation here...",
                lines=8
            )
            
            submit_btn = gr.Button("Generate Summary", variant="primary")
            clear_btn = gr.Button("Clear")
        
        with gr.Column():
            summary_output = gr.Textbox(
                label="Summary",
                lines=6,
                show_copy_button=True
            )
    
    # Examples
    gr.Examples(
        examples=examples,
        inputs=dialogue_input,
        label="Try these examples:"
    )
    
    # Event handlers
    submit_btn.click(
        fn=summarize_dialogue,
        inputs=dialogue_input,
        outputs=summary_output
    )
    
    clear_btn.click(
        fn=lambda: ("", ""),
        outputs=[dialogue_input, summary_output]
    )

# Launch
if __name__ == "__main__":
    demo.launch(share=True)
