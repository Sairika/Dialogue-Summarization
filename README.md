# Dialogue Summarization Model

This repository hosts a fine-tuned dialogue summarization model deployed on Hugging Face Spaces. This model is capable of generating concise summaries from conversational text, making it useful for various applications such as meeting minutes generation, customer service transcript analysis, and more.




## Features

*   **Fine-tuned Model**: Utilizes a fine-tuned FLAN-T5-Base model for high-quality dialogue summarization.
*   **Hugging Face Spaces Deployment**: Easily accessible and interactive demo via Hugging Face Spaces.
*   **Gradio Interface**: User-friendly web interface built with Gradio for quick summarization.
*   **Example Dialogues**: Provides pre-loaded examples to demonstrate functionality.




## How to Use

### Using the Hugging Face Space (Online Demo)

1.  Navigate to the Hugging Face Space: [https://huggingface.co/spaces/sairika/Dialogue-summarization](https://huggingface.co/spaces/sairika/Dialogue-summarization)
2.  You will see an interactive interface where you can input a dialogue.
3.  Enter your conversation into the 'Dialogue' text box. You can also use one of the provided example dialogues by clicking on them.
4.  Click the 'Generate Summary' button.
5.  The summarized dialogue will appear in the 'Summary' text box.

### Using the Model Programmatically (Local)

To use this fine-tuned model in your Python environment, follow these steps:

#### 1. Installation

First, ensure you have the necessary libraries installed:

```bash
pip install transformers peft torch gradio
```

#### 2. Load the Model and Tokenizer

```python
from transformers import AutoTokenizer
from peft import AutoPeftModelForSeq2SeqLM
import torch

repo_id = "sairika/FLAN-T5-Base-dialogsum-lora" # The model ID on Hugging Face Hub

try:
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(repo_id, device_map="auto", torch_dtype=torch.bfloat16)
    print(f"✅ Model and tokenizer loaded successfully from {repo_id}")
except Exception as e:
    print(f"❌ Error loading or testing the model: {e}")
    print(f"Please ensure the repository ID '{repo_id}' is correct and the model files were pushed successfully.")
```

#### 3. Generate Summaries

Once the model is loaded, you can use it to summarize dialogues:

```python
def create_prompts(dialogues, model_type):
    # This function should be defined based on how your model expects prompts.
    # For FLAN-T5, a common format is 'Summarize the following conversation.\n\n{dialogue}\n\nSummary: '
    prompts = []
    for dialogue in dialogues:
        prompts.append(f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: ")
    return prompts

# Sample dialogues
test_dialogues = [
    "#Person1#: Hi, Mr. Smith. I'm Doctor Hawkins. Why are you here today?\n#Person2#: I found it would be a good idea to get a check-up.\n#Person1#: Yes, well, you haven't had one for 5 years. You should have one every year.\n#Person2#: I know. I figure as long as there is nothing wrong, why go see the doctor?\n#Person1#: Well, you never know. An annual check-up is important for early detection of any potential health issues.\n#Person2#: I guess you're right. So, what do we do now?\n#Person1#: I'll just do a quick examination and we'll discuss your medical history. We should also talk about quitting smoking.\n#Person2#: I've been meaning to do that.\n#Person1#: We have some programs and medications that can help. I'll give you some information before you leave.\n#Person2#: Ok, thanks doctor.",
    "#Person1#: You're finally here! What took so long?\n#Person2#: I got stuck in traffic again. There was a terrible traffic jam near the Carrefour intersection.\n#Person1#: It's always rather congested during peak hours. Have you considered taking public transport?\n#Person2#: I have, but driving is just more convenient.\n#Person1#: It might be convenient, but it's not good for your health or the environment. Plus, you wouldn't have to deal with traffic jams.\n#Person2#: You have a point. I feel so bad about how much my car is adding to the pollution problem.\n#Person1#: Exactly! And walking or biking to the bus or train station is great exercise. When the weather is nicer, you could even bike the whole way.\n#Person2#: That's a good idea. I'll give public transport a try starting tomorrow.\n#Person1#: Great! Let me know how it goes."
]

model.eval()
generated_summaries = []

for dialogue in test_dialogues:
    model_type = 'flan-t5' # Based on the model name
    prompt = create_prompts([dialogue], model_type)[0]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
            length_penalty=0.6,
            early_stopping=True,
            do_sample=False
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_summaries.append(summary)

# Display results
print("\n📝 Test Summaries:")
print("-" * 50)
for i, (dialogue, summary) in enumerate(zip(test_dialogues, generated_summaries)):
    print(f"Dialogue {i+1}: {dialogue[:200]}...")
    print(f"Generated Summary {i+1}: {summary}")
    print("-" * 50)
```




## Model Details

This project utilizes a fine-tuned version of the **FLAN-T5-Base** model. FLAN-T5 is an encoder-decoder transformer model pre-trained on a massive dataset of text and code, and then instruction-tuned on a variety of tasks. The base version has approximately 250 million parameters.

For this project, the FLAN-T5-Base model was further fine-tuned using **LoRA (Low-Rank Adaptation)**, a parameter-efficient fine-tuning technique. LoRA significantly reduces the number of trainable parameters for downstream tasks by injecting small, trainable matrices into the transformer layers, while keeping the pre-trained model weights frozen. This allows for efficient adaptation of large pre-trained models to specific tasks with minimal computational overhead and storage requirements.




## Acknowledgements

This project was made possible by the open-source contributions of the Hugging Face team and the developers of the PEFT and Transformers libraries.



