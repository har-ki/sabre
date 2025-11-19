# Install dependencies
# pip install datasets openai

import os
from datasets import load_dataset

# HLE dataset requires authentication - set your HF token
# Get token from: https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set")
    print("The HLE dataset is gated. You need to:")
    print("1. Go to https://huggingface.co/datasets/cais/hle")
    print("2. Click 'Request access' and wait for approval")
    print("3. Get your token from https://huggingface.co/settings/tokens")
    print("4. Set environment variable: export HF_TOKEN=hf_...")
    exit(1)

# Load the HLE dataset from Hugging Face with authentication
dataset = load_dataset("cais/hle", split="test", token=HF_TOKEN)

# View a sample question
sample = dataset[0]
print("Sample fields:", list(sample.keys()))
print("\nQuestion:", sample['question'])
if 'image' in sample and sample['image']:
    print("Contains image: Yes")
if 'answer' in sample:
    print("\nAnswer:", sample['answer'])

# To run full evaluation (requires OpenAI API key)
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url='http://localhost:8011/v1')


def evaluate_sample(question, model="gpt-4o"):
    """Evaluate a single HLE question"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": question}
        ],
        max_tokens=8192
    )
    return response.choices[0].message.content


# Test on first 3 questions
for i in range(min(3, len(dataset))):
    question = dataset[i]['question']
    correct_answer = dataset[i]['answer']

    print(f"\n{'=' * 50}")
    print(f"Question {i + 1}:")
    print(question)

    model_answer = evaluate_sample(question)
    print(f"\nModel Answer: {model_answer}")
    print(f"Correct Answer: {correct_answer}")