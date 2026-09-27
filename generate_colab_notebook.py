import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚀 BrgyLink AI - Multilingual LLM Fine-Tuning\n",
    "\n",
    "Fine-tune **Llama-3.1-8B** on **Tagalog**, **Pangasinan**, and **Ilocano** instruction datasets using **Unsloth**.\n",
    "\n",
    "### ⚠️ Before you start:\n",
    "Go to **Runtime > Change runtime type** and select **T4 GPU** (Hardware accelerator: GPU)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Install Dependencies\n",
    "This uses the official Unsloth setup compatible with the latest Google Colab PyTorch/CUDA environment."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%capture\n",
    "import os, re\n",
    "if \"COLAB_\" not in \"\".join(os.environ.keys()):\n",
    "    !pip install unsloth\n",
    "else:\n",
    "    import torch; v = re.match(r'[\\d]{1,}\\.[\\d]{1,}', str(torch.__version__)).group(0)\n",
    "    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, \"0.0.34\")\n",
    "    !pip install sentencepiece protobuf \"datasets\" \"huggingface_hub>=0.34.0\" hf_transfer\n",
    "    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n",
    "    !pip install --no-deps --upgrade \"torchao>=0.16.0\"\n",
    "!pip install transformers==4.56.2\n",
    "!pip install --no-deps trl==0.22.2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load Base Model (Llama-3.1-8B with 4-bit quantization)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from unsloth import FastLanguageModel\n",
    "import torch\n",
    "\n",
    "max_seq_length = 2048\n",
    "dtype = None\n",
    "load_in_4bit = True\n",
    "\n",
    "# Load Llama-3.1-8B\n",
    "model, tokenizer = FastLanguageModel.from_pretrained(\n",
    "    model_name = \"unsloth/Meta-Llama-3.1-8B-bnb-4bit\",\n",
    "    max_seq_length = max_seq_length,\n",
    "    dtype = dtype,\n",
    "    load_in_4bit = load_in_4bit,\n",
    ")\n",
    "\n",
    "# Configure LoRA adapters\n",
    "model = FastLanguageModel.get_peft_model(\n",
    "    model,\n",
    "    r = 16,\n",
    "    target_modules = [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",\n",
    "                      \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
    "    lora_alpha = 16,\n",
    "    lora_dropout = 0,\n",
    "    bias = \"none\",\n",
    "    use_gradient_checkpointing = \"unsloth\",\n",
    "    random_state = 3407,\n",
    "    use_rslora = False,\n",
    "    loftq_config = None,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Download & Prepare Datasets\n",
    "Loads Pangasinan, Ilocano, and Filipino (Tagalog) Alpaca datasets and formats them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import load_dataset, concatenate_datasets\n",
    "\n",
    "print(\"Loading datasets...\")\n",
    "ds_pangasinan = load_dataset(\"PLTAT/alpaca_pangasinan\", split=\"train\")\n",
    "ds_ilocano = load_dataset(\"saillab/alpaca_ilocano_taco\", split=\"train\")\n",
    "ds_filipino = load_dataset(\"saillab/alpaca-filipino-cleaned\", split=\"train\")\n",
    "\n",
    "# Ensure consistent schema\n",
    "def standardize_columns(ds):\n",
    "    cols_to_remove = [c for c in ds.column_names if c not in [\"instruction\", \"input\", \"output\"]]\n",
    "    return ds.remove_columns(cols_to_remove)\n",
    "\n",
    "ds_pangasinan = standardize_columns(ds_pangasinan)\n",
    "ds_ilocano = standardize_columns(ds_ilocano)\n",
    "ds_filipino = standardize_columns(ds_filipino)\n",
    "\n",
    "dataset = concatenate_datasets([ds_pangasinan, ds_ilocano, ds_filipino])\n",
    "print(f\"Total combined rows: {len(dataset)}\")\n",
    "\n",
    "alpaca_prompt = \"\"\"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n",
    "\n",
    "### Instruction:\n",
    "{}\n",
    "\n",
    "### Input:\n",
    "{}\n",
    "\n",
    "### Response:\n",
    "{}\"\"\"\n",
    "\n",
    "EOS_TOKEN = tokenizer.eos_token\n",
    "def formatting_prompts_func(examples):\n",
    "    instructions = examples[\"instruction\"]\n",
    "    inputs       = examples[\"input\"]\n",
    "    outputs      = examples[\"output\"]\n",
    "    texts = []\n",
    "    for instruction, inp, output in zip(instructions, inputs, outputs):\n",
    "        inp_val = inp if inp is not None else \"\"\n",
    "        text = alpaca_prompt.format(instruction, inp_val, output) + EOS_TOKEN\n",
    "        texts.append(text)\n",
    "    return { \"text\" : texts }\n",
    "\n",
    "dataset = dataset.map(formatting_prompts_func, batched = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Fine-Tune the Model\n",
    "Uses `SFTConfig` and `SFTTrainer` correctly configured for modern `trl`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from trl import SFTTrainer, SFTConfig\n",
    "\n",
    "trainer = SFTTrainer(\n",
    "    model = model,\n",
    "    tokenizer = tokenizer,\n",
    "    train_dataset = dataset,\n",
    "    dataset_text_field = \"text\",\n",
    "    max_seq_length = max_seq_length,\n",
    "    dataset_num_proc = 2,\n",
    "    packing = False,\n",
    "    args = SFTConfig(\n",
    "        per_device_train_batch_size = 2,\n",
    "        gradient_accumulation_steps = 4,\n",
    "        warmup_steps = 5,\n",
    "        max_steps = 100,  # Set to 100-300 for quick training on free Colab tier\n",
    "        learning_rate = 2e-4,\n",
    "        fp16 = not torch.cuda.is_bf16_supported(),\n",
    "        bf16 = torch.cuda.is_bf16_supported(),\n",
    "        logging_steps = 10,\n",
    "        optim = \"adamw_8bit\",\n",
    "        weight_decay = 0.01,\n",
    "        lr_scheduler_type = \"linear\",\n",
    "        seed = 3407,\n",
    "        output_dir = \"outputs\",\n",
    "        report_to = \"none\",\n",
    "    ),\n",
    ")\n",
    "\n",
    "trainer_stats = trainer.train()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Test Live in Colab (Optional Quick Chat!)\n",
    "Test your model directly here before downloading!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "FastLanguageModel.for_inference(model)\n",
    "\n",
    "# Test prompt in Tagalog, Pangasinan, or Ilocano:\n",
    "test_prompt = alpaca_prompt.format(\n",
    "    \"Antoy serbisyo na barangay parad saray residente?\",  # Pangasinan test\n",
    "    \"\",\n",
    "    \"\",\n",
    ")\n",
    "\n",
    "inputs = tokenizer([test_prompt], return_tensors = \"pt\").to(\"cuda\")\n",
    "outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)\n",
    "print(tokenizer.batch_decode(outputs)[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Export to GGUF for Local CPU Inference\n",
    "Exports the model as a `.gguf` file so you can run it on your local machine using `llama.cpp`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Export to GGUF using Q4_K_M quantization\n",
    "model.save_pretrained_gguf(\"brgylink_llama3\", tokenizer, quantization_method = \"q4_k_m\")\n",
    "print(\"Finished exporting! Check the file explorer on the left sidebar for the .gguf file.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('/home/kurt/.gemini/antigravity-ide/scratch/mission17/brgylink-ai/train_brgylink_llm.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
print("Successfully generated train_brgylink_llm.ipynb")
