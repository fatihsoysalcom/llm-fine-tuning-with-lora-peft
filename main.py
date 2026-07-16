import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset # Using Hugging Face datasets library for convenience

# 1. Model and Tokenizer Loading
# Using a small pre-trained model for demonstration. The article mentions Ornith 9b,
# but for a quick, runnable example, a smaller model like 'facebook/opt-125m' is more practical.
# This simulates loading a base LLM before fine-tuning.
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set pad_token_id to eos_token_id for decoder-only models if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Synthetic Dataset Creation
# This small dataset simulates instruction-response pairs for fine-tuning.
# In a real scenario, this would be a larger, domain-specific dataset.
data = {
    "instruction": [
        "What is the capital of France?",
        "Who wrote 'Romeo and Juliet'?",
        "What is 2+2?",
        "Tell me about fine-tuning LLMs.",
    ],
    "response": [
        "The capital of France is Paris.",
        "William Shakespeare wrote 'Romeo and Juliet'.",
        "2+2 equals 4.",
        "Fine-tuning LLMs adapts a pre-trained model to a specific task or dataset, improving its performance on niche topics. It's more efficient than training from scratch.",
    ]
}
# Convert to Hugging Face Dataset format
dataset = Dataset.from_dict(data)

# 3. Data Preparation Function
# This function tokenizes the input and formats it for causal language modeling.
def tokenize_function(examples):
    # Combine instruction and response into a single text for causal LM
    texts = [f"Instruction: {instr}\nResponse: {resp}{tokenizer.eos_token}"
             for instr, resp in zip(examples["instruction"], examples["response"])]
    tokenized_inputs = tokenizer(
        texts,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    # For causal LMs, labels are typically the input IDs themselves, shifted.
    # The Trainer handles this shifting automatically if labels are provided.
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. LoRA Configuration (Parameter-Efficient Fine-Tuning - PEFT)
# LoRA is crucial for efficient fine-tuning on a single GPU (like H200 mentioned in article),
# as it significantly reduces the number of trainable parameters and memory footprint.
lora_config = LoraConfig(
    r=8, # LoRA attention dimension
    lora_alpha=16, # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "v_proj"], # Which modules to apply LoRA to
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM # Specify the task type
)

# Apply LoRA to the base model
model = get_peft_model(model, lora_config)
# Print trainable parameters to show the efficiency of LoRA
model.print_trainable_parameters()

# 5. Training Arguments
# Define training parameters. For a real fine-tuning, these would be more extensive.
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3, # Small number of epochs for demonstration
    per_device_train_batch_size=1, # Small batch size
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no", # Don't save checkpoints for this minimal example
    report_to="none", # Disable reporting to external services
    # Use CPU for demonstration if GPU not available, otherwise 'cuda'
    # The article implies GPU, but for a runnable example, CPU fallback is good.
    no_cuda=not torch.cuda.is_available(),
)

# 6. Trainer Setup and Training
# The Trainer orchestrates the fine-tuning process.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

print("\n--- Starting Fine-tuning ---")
trainer.train()
print("--- Fine-tuning Complete ---")

# 7. Demonstrate Inference After Fine-tuning
def generate_response(model, tokenizer, instruction, max_new_tokens=50):
    prompt = f"Instruction: {instruction}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, # Enable sampling for more varied responses
            top_k=50,
            top_p=0.95
        )
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response.strip()

test_instruction = "Tell me about fine-tuning LLMs."

print(f"\n--- Testing Fine-tuned Model ---")
print(f"Instruction: {test_instruction}")
fine_tuned_response = generate_response(model, tokenizer, test_instruction)
print(f"Fine-tuned Response: {fine_tuned_response}")

# The fine-tuned model's response should now align with the specific training data
# provided, demonstrating how fine-tuning adapts the model to new information.
