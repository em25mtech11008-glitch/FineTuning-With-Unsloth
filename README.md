<div align="center">
  <h1>🧠 Llama 3.2 3B Instruct — Fine-Tuned with Unsloth</h1>
  <p><i>Teaching an LLM to "think" using stream-of-consciousness reasoning and QLoRA.</i></p>
  
  [![Unsloth](https://img.shields.io/badge/Unsloth-2x_Faster-blue.svg)](https://github.com/unslothai/unsloth)
  [![Quantization](https://img.shields.io/badge/Quantization-4--bit-orange)](https://huggingface.co/docs/transformers/main_classes/quantization)
  [![LoRA](https://img.shields.io/badge/PEFT-QLoRA-green)](https://huggingface.co/docs/peft/index)
  [![Export](https://img.shields.io/badge/Export-GGUF-yellow)](https://github.com/ggerganov/llama.cpp)
</div>

<hr>

## 📌 Project Overview

This project demonstrates the process of efficiently fine-tuning the **Llama-3.2-3B-Instruct** model to mimic human stream-of-consciousness thinking. By leveraging the **Unsloth** library and **QLoRA**, the model is trained to engage in thorough, iterative reasoning—emphasizing exploration, self-doubt, and continuous refinement before arriving at an answer. 

## 📊 Dataset

- **Source**: [`ServiceNow-AI/R1-Distill-SFT`](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT) (Subset of 171,647 examples)
- **Split**: `train`
- **Format**: The dataset contains math/logic problems, detailed thoughts (reasoning process), and final solutions.
- **Data Preparation**: A custom prompt template was created to encapsulate the problem, isolate the thought process inside `<think>` tags, and present the final solution.

## 🤖 Base Model

- **Model Name**: `unsloth/Llama-3.2-3B-Instruct`
- **Type**: Large Language Model (Instruction-tuned)
- **Parameter Count**: 3 Billion

## ⚙️ Fine-Tuning Method

The model was fine-tuned using **QLoRA** (Quantized Low-Rank Adaptation) via the **Unsloth** library. This approach significantly optimizes VRAM usage and maximizes training speed without compromising performance.

- **Rank (r)**: 16
- **LoRA Alpha**: 16
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Dropout**: 0
- **Bias**: None
- **Quantization**: 4-bit (via Unsloth's `load_in_4bit`)

## 🛠️ Training Configuration

- **Batch Size per Device**: 2
- **Gradient Accumulation Steps**: 4
- **Warmup Steps**: 5
- **Total Training Steps**: 60 *(Demonstration run)*
- **Learning Rate**: 2e-4
- **Optimizer**: `adamw_8bit`
- **Weight Decay**: 0.01
- **LR Scheduler**: Linear
- **Precision**: fp16/bf16 *(automatically selected based on hardware)*

## 🔄 Workflow

1. **Dataset Loading**: Downloaded and mapped the `R1-Distill-SFT` dataset.
2. **Preprocessing**: Formatted the data using a custom prompt template to teach the model to "think" before answering.
3. **Model Initialization**: Loaded the base Llama-3.2-3B model with 4-bit quantization using Unsloth.
4. **PEFT Configuration**: Applied LoRA adapters to all attention and MLP layers.
5. **Fine-Tuning**: Trained the model using TRL's `SFTTrainer`.
6. **Inference**: Evaluated the model by generating responses with a highly elevated temperature (1.5) to encourage creative and extensive reasoning.
7. **Export**: Saved the model locally and converted it to **GGUF** format (`Q8_0` quantization) for highly efficient local deployment via Ollama.

## 💡 Example / Inference

**Prompt**: *"How many 'r's are present in 'strawberry'?"*

**Model's Stream-of-Consciousness Output**:
> *"Okay, so I have this word 'strawberry', and I need to find out how many times the letter 'r' appears in it. Hmm, that's a pretty straightforward question, but I always get a little nervous when I'm faced with reading and counting, just because numbers and words can get tricky sometimes.*
> 
> *Let's start by writing down the word 'strawberry' exactly as it is:*
> 
> *S-T-R-A-W-B-E-R-R-Y*
> 
> *Alright, so I've got the word on the screen in front of me. Now, my next step would be to go through each letter individually and count how many times the letter 'r' appears... S - no 'r', T - no 'r', R - 1... E - no 'r', R - 1, R - 1, Y - no 'r'..."*

## 💻 Tech Stack

- **[Transformers](https://huggingface.co/docs/transformers/index)**: For dataset and model loading.
- **[Unsloth](https://github.com/unslothai/unsloth)**: For 2x faster, memory-efficient fine-tuning.
- **[TRL](https://huggingface.co/docs/trl/index)**: For the `SFTTrainer` implementation.
- **[PyTorch](https://pytorch.org/)**: Core deep learning framework.
- **[Ollama](https://ollama.com/)**: For local inference of the converted GGUF model.

## ☁️ Cloud / Colab Deployment Guide

If you intend to run this training pipeline or inference in a cloud environment (e.g., Google Colab), follow these steps to bypass common background-service errors (like Ollama connection refusals):

1. **Install Dependencies**:
   ```bash
   pip install unsloth "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
   ```

2. **Initialize Ollama Service**:
   In cloud environments, the Ollama service is not pre-installed or running by default as a persistent daemon. You must manually install and start it in the background:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start the Ollama server in the background
   ollama serve &
   ```

3. **Create and Run the Model**:
   After exporting your model to GGUF, build and run it with Ollama:
   ```bash
   cd ./parth-001-3B-GGUF_gguf
   ollama create unsloth_model -f ./Modelfile
   ollama run unsloth_model
   ```
