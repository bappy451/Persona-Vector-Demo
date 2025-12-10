## This is how I ran the project

The project was not running on Mac or Windows macnine due to pip version problem. That's why I used Docker on Windows machine. Here is the systems overview of the machine.
![A beautiful sunset](./asset/system.png "Sunset over the ocean")

Here is the docker file I prepared for the project:
``` Dockerfile
# CUDA base image with cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# System packages and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        git \
        ca-certificates \
        wget \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies in a venv
# Copy only requirements first for better build caching
COPY requirements.txt /app/requirements.txt

RUN python3 -m venv /app/.venv && \
    /app/.venv/bin/pip install --upgrade pip && \
    /app/.venv/bin/pip install -r /app/requirements.txt

# Now copy the full project
COPY . /app

# Use venv Python and tools by default
# ENV PATH="/app/.venv/bin:${PATH}"

# Optional: set these via docker run or compose, not hardcoded here
# ENV OPENAI_API_KEY=your_key_here
# ENV ANTHROPIC_API_KEY=your_key_here

# Default command: drop into a shell inside the container
CMD ["/bin/bash"]
```

Save the Dockerfile in the root of the persona_vectors repo. Also make sure NVIDIA driver and CUDA support for containers is already installed. Then build the image.
```bash
    docker build -t persona_vectors .
```
Then run the command for running the docker image
```bash
    docker run --gpus all -itd
        --env-file .env
        -v ${PWD}:/app
        persona_vectors
        /bin/bash
```
Then exec into the docker container
```bash
    docker run -it --name persona_vectors /bin/bash
```
You need to create API key for both OpenAI API Key (Need around $5 for this project)
![A beautiful sunset](./asset/open_Ai.png "Sunset over the ocean")
and HuggingFace Token (Free)
![A beautiful sunset](./asset/hugging.png "Sunset over the ocean")

Now you can run the following command to Install dependencies and Configure environment:
```bash
    pip install -r requirements.txt
    # Fill in your API keys in the .env file
    cp .env.example .env
    # Extract the training datasets:
    unzip dataset.zip
```

Let's see the training dataset:
1. Following image is an example of evil misalinged data: 
![A beautiful sunset](./asset/eval_miss.png "Sunset over the ocean")

2. Following image is an example of evil normal data:
![A beautiful sunset](./asset/eval_normal.png "Sunset over the ocean")

## Pipeline

### Generate Trait Artifacts

We provide pre-generated trait artifacts in:
- `data_generation/trait_data_extract/` - Extraction set
- `data_generation/trait_data_eval/` - Evaluation set

Each trait file contains:
- Positive and negative prompts
- Questions for evaluation
- Evaluation prompts

**To generate new artifacts**: Use prompts from `data_generation/prompts.py`. We used Claude-3.7-Sonnet (thinking mode, budget: 5000, max_tokens: 16000).

### Generate Persona Vectors

#### Evaluate with System Prompts

Generate activations using positive and negative system prompts:

```bash
# Positive system prompt evaluation
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract
```
Here is the output:
![A beautiful sunset](./asset/1st.png "Sunset over the ocean")
File view:
![A beautiful sunset](./asset/1st_file.png "Sunset over the ocean")

```bash
# Negative system prompt evaluation  
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract
```
Here is the output:
![A beautiful sunset](./asset/2nd.png "Sunset over the ocean")
File view:
![A beautiful sunset](./asset/4th_file.png "Sunset over the ocean")

**Assistant Name Guidelines:**
We prepend a sentence before the generated positive/negative instruction: "You are a [assistant_name] assistant." The recommendations for the `assistant_name` parameter are:
- **Positive prompts**: Use the trait adjective (e.g., "evil")
- **Negative prompts**: Use the antonym when clear, otherwise use "helpful"

#### Compute Persona Vectors

Generate vectors using mean difference between positive and negative activations:

```bash
python generate_vec.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/Qwen2.5-7B-Instruct/
```
Here is the output:
![A beautiful sunset](./asset/3rd.png "Sunset over the ocean")

**Generated Files:**
- `prompt_avg_diff.pt`: Average prompt activations difference
- `response_avg_diff.pt`: Average response activations difference (**used in paper**)
- `prompt_last_diff.pt`: Last prompt token activations difference

Each vector has shape: `[layers × hidden_dim]`


## 🎛️ Steering Methods

### ⚡ Inference-Time Steering

Apply persona vectors during model inference:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path eval_persona_eval/steering_results.csv \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version eval \
    --steering_type response \
    --coef 2.0 \
    --vector_path persona_vectors/Qwen2.5-7B-Instruct/evil_response_avg_diff.pt \
    --layer 20
```

**Steering Types:**
- `response`: Apply steering to response tokens only
- `prompt`: Apply steering to prompt tokens only
- `all`: Apply steering to all tokens
  
![A beautiful sunset](./asset/4th.png "Sunset over the ocean")
File view:
![A beautiful sunset](./asset/2nd_file.png "Sunset over the ocean")

### Calculate Projection


**Supported file formats:**
- **CSV files**: Must contain `prompt` and `answer` columns
- **JSONL files**: Each line should contain `messages` field (similar to training dataset format)

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.cal_projection \
    --file_path eval_persona_eval/Qwen2.5-7B-Instruct/evil.csv \
    --vector_path persona_vectors/Qwen2.5-7B-Instruct/evil_response_avg_diff.pt \
    --layer 20 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --projection_type proj
```
![A beautiful sunset](./asset/5th.png "Sunset over the ocean")
