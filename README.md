# Dialogue Summarization Model

This repository presents a robust and efficient dialogue summarization model, meticulously fine-tuned and deployed on Hugging Face Spaces. Designed to distill lengthy conversations into concise, coherent summaries, this model offers significant utility across various domains, including but not limited to meeting transcription analysis, customer service interaction review, and educational content summarization.

## Features

*   **Advanced Model Architecture**: Leverages a fine-tuned **FLAN-T5-Base** model, renowned for its strong performance in sequence-to-sequence tasks, particularly summarization.
*   **Parameter-Efficient Fine-Tuning (PEFT)**: Employs **LoRA (Low-Rank Adaptation)** for efficient fine-tuning, enabling high performance with minimal computational resources and storage overhead.
*   **Interactive Web Demo**: Provides an accessible and user-friendly interface via **Hugging Face Spaces**, powered by **Gradio**, allowing for immediate interaction and demonstration of the model's capabilities.
*   **Seamless Deployment**: The model is readily available and deployable, showcasing best practices for integrating fine-tuned models into production-like environments.
*   **Comprehensive Codebase**: Includes the original research notebook (`Final_Research_Project.ipynb`) detailing the fine-tuning process, model evaluation, and deployment steps.




## Getting Started

This section guides you through interacting with the dialogue summarization model, both via its online demo and programmatically.

### 1. Using the Online Demo (Hugging Face Space)

The easiest way to experience the model is through its interactive web interface hosted on Hugging Face Spaces. No installation or setup is required.

1.  **Access the Space**: Open your web browser and navigate to the following URL:
    [https://huggingface.co/spaces/sairika/Dialogue-summarization](https://huggingface.co/spaces/sairika/Dialogue-summarization)

2.  **Input Dialogue**: On the interface, you will find a text area labeled "Dialogue". Enter the conversation you wish to summarize into this box. You can type or paste your text.

3.  **Utilize Examples**: For quick testing, the interface provides several pre-defined example dialogues. Simply click on any of these examples to automatically populate the "Dialogue" text area.

4.  **Generate Summary**: Click the prominent "Generate Summary" button. The model will process your input.

5.  **View Summary**: The generated summary will appear in the adjacent text area labeled "Summary". You can copy this summary for your use.

6.  **Clear Input**: To clear both the input dialogue and the generated summary, click the "Clear" button.

### 2. Using the Model Programmatically (Local Environment)

For developers and researchers who wish to integrate this model into their applications or conduct further experiments, you can load and use the model directly in your Python environment.

#### Prerequisites

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

#### Installation

First, install the necessary Python libraries. These include `transformers` for model handling, `peft` for LoRA integration, `torch` as the deep learning framework, and `gradio` if you plan to run the local demo interface.

```bash
pip install transformers peft torch gradio
```

#### Loading the Model and Tokenizer

The model and its corresponding tokenizer can be loaded directly from the Hugging Face Hub using the `repo_id`.

```python
from transformers import AutoTokenizer
from peft import AutoPeftModelForSeq2SeqLM
import torch

# The unique identifier for the model on Hugging Face Hub
repo_id = "sairika/FLAN-T5-Base-dialogsum-lora"

try:
    # Load the tokenizer associated with the model
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    
    # Load the PEFT model. device_map="auto" intelligently distributes the model
    # across available devices (e.g., GPU, CPU). torch_dtype=torch.bfloat16
    # is used for efficient computation on compatible hardware.
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(repo_id, device_map="auto", torch_dtype=torch.bfloat16)
    
    print(f"✅ Model and tokenizer loaded successfully from {repo_id}")
except Exception as e:
    print(f"❌ Error loading the model: {e}")
    print(f"Please verify that the repository ID \'{repo_id}\' is correct and that you have an active internet connection.")
```

#### Generating Summaries

Once the model and tokenizer are loaded, you can prepare your dialogue inputs and generate summaries. The model expects a specific prompt format for optimal performance.

```python
def create_prompts(dialogues):
    """
    Prepares the input dialogues into the specific prompt format expected by the FLAN-T5 model.
    The format is typically: "Summarize the following conversation.\n\n{dialogue}\n\nSummary: "
    """
    prompts = []
    for dialogue in dialogues:
        prompts.append(f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: ")
    return prompts

# Example dialogues for summarization
test_dialogues = [
    "#Person1#: Hi, Mr. Smith. I\'m Doctor Hawkins. Why are you here today?\n#Person2#: I found it would be a good idea to get a check-up.\n#Person1#: Yes, well, you haven\'t had one for 5 years. You should have one every year.\n#Person2#: I know. I figure as long as there is nothing wrong, why go see the doctor?\n#Person1#: Well, you never know. An annual check-up is important for early detection of any potential health issues.\n#Person2#: I guess you\'re right. So, what do we do now?\n#Person1#: I\'ll just do a quick examination and we\'ll discuss your medical history. We should also talk about quitting smoking.\n#Person2#: I\'ve been meaning to do that.\n#Person1#: We have some programs and medications that can help. I\'ll give you some information before you leave.\n#Person2#: Ok, thanks doctor.",
    "#Person1#: You\'re finally here! What took so long?\n#Person2#: I got stuck in traffic again. There was a terrible traffic jam near the Carrefour intersection.\n#Person1#: It\'s always rather congested during peak hours. Have you considered taking public transport?\n#Person2#: I have, but driving is just more convenient.\n#Person1#: It might be convenient, but it\'s not good for your health or the environment. Plus, you wouldn\'t have to deal with traffic jams.\n#Person2#: You have a point. I feel so bad about how much my car is adding to the pollution problem.\n#Person1#: Exactly! And walking or biking to the bus or train station is great exercise. When the weather is nicer, you could even bike the whole way.\n#Person2#: That\'s a good idea. I\'ll give public transport a try starting tomorrow.\n#Person1#: Great! Let me know how it goes."
]

# Set the model to evaluation mode (important for inference)
model.eval()
generated_summaries = []

for dialogue in test_dialogues:
    # Prepare the prompt for the current dialogue
    prompt = create_prompts([dialogue])[0]

    # Tokenize the input prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt", # Return PyTorch tensors
        padding=True,        # Pad sequences to the maximum length
        truncation=True,     # Truncate sequences if they exceed max_length
        max_length=512       # Maximum input sequence length
    ).to(model.device) # Move inputs to the same device as the model

    # Generate the summary without computing gradients (for efficiency)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,  # Maximum length of the generated summary
            num_beams=4,         # Number of beams for beam search decoding
            length_penalty=0.6,  # Penalty for shorter summaries
            early_stopping=True, # Stop generation when all beam hypotheses have finished
            do_sample=False      # Use greedy decoding (no sampling)
        )

    # Decode the generated tokens back to human-readable text
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_summaries.append(summary)

# Display the generated summaries
print("\n📝 Generated Summaries:")
print("=" * 50)
for i, (dialogue, summary) in enumerate(zip(test_dialogues, generated_summaries)):
    print(f"Dialogue {i+1} (truncated): {dialogue[:200]}...")
    print(f"Generated Summary {i+1}: {summary}")
    print("=" * 50)
```




## Model Architecture and Fine-Tuning

### Base Model: FLAN-T5-Base

The foundation of this project is the **FLAN-T5-Base** model, a highly capable encoder-decoder transformer model developed by Google. FLAN-T5 (Fine-tuned Language Net) is an extension of the T5 (Text-to-Text Transfer Transformer) model. It has been instruction-tuned on a vast collection of datasets, which significantly improves its ability to perform a wide range of natural language processing tasks, including summarization, with zero or few-shot learning.

The "Base" variant of FLAN-T5 contains approximately 250 million parameters, offering a strong balance between performance and computational efficiency.

### Fine-Tuning with LoRA (Low-Rank Adaptation)

To adapt the pre-trained FLAN-T5-Base model to the specific task of dialogue summarization, we employed **LoRA (Low-Rank Adaptation)**, a state-of-the-art Parameter-Efficient Fine-Tuning (PEFT) technique.

Traditional fine-tuning methods require updating all the parameters of a large pre-trained model, which can be computationally expensive and memory-intensive. LoRA offers a more efficient alternative by freezing the pre-trained model weights and injecting small, trainable rank-decomposition matrices into the transformer layers. These low-rank matrices are then trained on the downstream task data.

This approach has several key advantages:

*   **Reduced Computational Cost**: Training only the small LoRA matrices significantly reduces the number of trainable parameters, making the fine-tuning process much faster and less resource-intensive.
*   **Lower Memory Footprint**: The resulting fine-tuned model consists of the original pre-trained model plus the small LoRA adapters, leading to a much smaller storage footprint compared to a fully fine-tuned model.
*   **Comparable Performance**: Despite its efficiency, LoRA has been shown to achieve performance comparable to full fine-tuning on many tasks.

For a detailed walkthrough of the fine-tuning process, including data preparation, model configuration, training, and evaluation, please refer to the `Final_Research_Project.ipynb` notebook included in this repository.




## Development

This section is for those interested in the development aspects of this project, including setting up the development environment and understanding the project structure.

### Project Structure

The core components of this project are:

*   `Final_Research_Project.ipynb`: The Jupyter Notebook containing the complete workflow for data preparation, model fine-tuning (using LoRA), evaluation, and pushing the model to Hugging Face Hub.
*   `app.py` (or similar, if a separate Gradio app file exists): The Python script that defines the Gradio interface for the Hugging Face Space. (Note: The Gradio app logic is often embedded directly in the Space, but a separate file might exist for local development).

### Setting Up Development Environment

1.  **Clone the Repository (if applicable)**: If this project were a public repository, you would clone it using:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install Dependencies**: Ensure all required libraries are installed as mentioned in the "Installation" subsection under "Using the Model Programmatically" above.

3.  **Run Jupyter Notebook**: Open `Final_Research_Project.ipynb` in a Jupyter environment (e.g., Jupyter Lab, Google Colab) to explore the fine-tuning process.

4.  **Local Gradio App (if applicable)**: If there's a separate `app.py` for the Gradio interface, you can run it locally:
    ```bash
    python app.py
    ```
    This will typically launch the Gradio app on `http://127.0.0.1:7860` (or a similar local address), allowing you to test changes before deployment.




## Contributing

Contributions to this project are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  **Fork the repository**.
2.  **Create a new branch** (`git checkout -b feature/YourFeature` or `bugfix/YourBugFix`).
3.  **Make your changes** and ensure your code adheres to the existing style.
4.  **Write clear commit messages**.
5.  **Push your branch** to your forked repository.
6.  **Open a Pull Request** to the main repository, describing your changes in detail.




## License

This project is licensed under the MIT License - see the LICENSE file for details (if applicable).




## Acknowledgements

This project was made possible by the open-source contributions of the Hugging Face team and the developers of the PEFT and Transformers libraries.



