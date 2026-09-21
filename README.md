# 🦙 Fine-Tuning Llama 3.2 3B Instruct with Unsloth

## 📌 Project Overview
This project demonstrates how to efficiently fine-tune the **Llama-3.2-3B-Instruct** model using **Unsloth** and **QLoRA**. The goal is to enhance the model's ability to engage in thorough, iterative reasoning (mimicking human stream-of-consciousness thinking) by training it on the `ServiceNow-AI/R1-Distill-SFT` dataset.

## 📊 Dataset
- **Source**: [`ServiceNow-AI/R1-Distill-SFT`](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT)
- **Split**: `train`
- **Format**: The dataset contains math/logic problems, detailed thoughts (reasoning process), and final solutions.
- **Data Preparation**: The text was formatted using a custom template that encapsulates the problem, the thought process inside `<think>` tags, and the final solution.

## 🧠 Base Model
- **Model Name**: `unsloth/Llama-3.2-3B-Instruct`
- **Type**: Large Language Model (Instruction-tuned)
- **Parameter Count**: 3 Billion

## ⚙️ Fine-Tuning Method
The model was fine-tuned using **QLoRA** (Quantized Low-Rank Adaptation) via the **Unsloth** library to optimize VRAM usage and maximize training speed.
- **Rank (r)**: 16
- **LoRA Alpha**: 16
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Dropout**: 0
- **Bias**: None
- **Quantization**: 4-bit (via Unsloth's `load_in_4bit`)

## 🛠️ Training Configuration
- **Max Sequence Length**: [ADD INFORMATION] (e.g., 2048)
- **Batch Size per Device**: 2
- **Gradient Accumulation Steps**: 4
- **Warmup Steps**: 5
- **Max Training Steps**: 60
- **Learning Rate**: 2e-4
- **Optimizer**: `adamw_8bit`
- **Weight Decay**: 0.01
- **LR Scheduler**: Linear
- **Precision**: fp16 or bf16 (depending on hardware support)

## 🔄 Workflow
1. **Dataset Loading**: Downloaded the `R1-Distill-SFT` dataset from Hugging Face.
2. **Preprocessing**: Formatted the data using a custom prompt template to teach the model to "think" before answering.
3. **Model Initialization**: Loaded the base Llama-3.2-3B model with 4-bit quantization using Unsloth.
4. **PEFT Configuration**: Applied LoRA adapters to all attention and MLP layers.
5. **Fine-Tuning**: Trained the model using TRL's `SFTTrainer` for 60 steps.
6. **Inference**: Evaluated the model by generating responses with a highly elevated temperature (1.5) to encourage creative and extensive reasoning.
7. **Export**: Saved the model locally and converted it to **GGUF** format (`Q8_0` quantization) for deployment via Ollama.

## 💡 Example / Inference
**Prompt**: *"give me number which has 8 zero 2 one 2 one can not together and all zero should together number should be acccending 0->1->2 like this"*

**Model's Stream-of-Consciousness Output**:
> *"Alright, so I have this problem: find a number with eight zeros, two ones, and two twos. But here's the twist - the number can't have all zeros together, and the digits should be arranged in ascending order from left to right, starting from 0. Let me try to visualize the number structure first..."*

## 💻 Tech Stack
- **Transformers / Hugging Face**: For dataset and model loading.
- **Unsloth**: For 2x faster, memory-efficient fine-tuning.
- **TRL (Transformer Reinforcement Learning)**: For the `SFTTrainer`.
- **PyTorch**: Deep learning framework.
- **Ollama**: For local inference of the converted GGUF model.

## 📁 Project Structure
- `unsloth_finetuning.ipynb`: The main Jupyter notebook containing the entire workflow from setup to GGUF export.
- `parth-001-3B/`: Directory containing the saved LoRA model and tokenizer.
- `parth-001-3B-GGUF_gguf/`: Directory containing the exported GGUF model (`llama-3.2-3b-instruct.Q8_0.gguf`) and the `Modelfile`.

## 🚀 Installation & Usage (Cloud/Colab)
If you intend to run this notebook in a cloud environment (like Google Colab), follow these steps:

1. **Install Dependencies**:
   ```bash
   pip install unsloth "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
   ```

2. **Ollama Setup (To fix the "model not found" error)**:
   In cloud environments, the Ollama service is not pre-installed or running by default. You must install and start it in the background before creating the model:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start the Ollama server in the background
   ollama serve &
   ```
   *(In Colab, you can also use Python's `subprocess` to start it)*

3. **Create and Run the Model**:
   ```bash
   cd ./parth-001-3B-GGUF_gguf
   ollama create unsloth_model -f ./Modelfile
   ollama run unsloth_model
   ```
