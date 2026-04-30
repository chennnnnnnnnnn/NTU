# HW1: MLLM Poster Generation and RAG
**Student ID:** p14942a08

---

## Environment

| Item | Version |
|------|---------|
| Python | 3.10.20 |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 5.5.3 |
| Diffusers | 0.37.1 |
| sentence-transformers | 5.4.0 |
| PyMuPDF (fitz) | 1.27.2 |
| Pillow | 12.1.1 |
| NumPy | 2.2.6 |
| reportlab | 4.4.10 |
| rank-bm25 | latest |

### Create the Environment

```bash
# Create conda environment with Python 3.10
conda create -n ai python=3.10 -y
conda activate ai

# Install PyTorch with CUDA 12.1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers==5.5.3 \
            diffusers==0.37.1 \
            accelerate==1.13.0 \
            safetensors==0.7.0 \
            tokenizers==0.22.2 \
            huggingface_hub==1.10.1

# Install model-specific packages
pip install sentence-transformers==5.4.0 \
            qwen-vl-utils==0.0.14 \
            open_clip_torch==3.3.0 \
            invisible-watermark==0.2.0 \
            ftfy==6.3.1

# Install utilities
pip install PyMuPDF==1.27.2.2 \
            Pillow \
            numpy \
            scipy \
            scikit-learn \
            rank-bm25==0.2.2 \
            reportlab
```

> Requires CUDA 12.1 compatible GPU driver. If using a different CUDA version, adjust the `--index-url` accordingly (e.g., `cu118` for CUDA 11.8).

All dependencies are already installed in the `ai` conda environment.  
The following model weights are downloaded automatically on first run (via HuggingFace):

- `Qwen/Qwen3-VL-2B-Instruct`
- `stabilityai/stable-diffusion-xl-base-1.0`
- `h94/IP-Adapter` (`ip-adapter_sdxl.bin`)
- `sentence-transformers/all-mpnet-base-v2`

---

## Task 1: MLLM Poster Generation

**Script:** `task1.py`

**Input:**
- `product/` — 100 product images (`p001.png`–`p100.png`)
- `ref/` — reference poster images
- `pairs.csv` — maps each pair to product image, reference image, and product title

**Output:**
- `task1_output/strategy_A/` — 100 generated posters (Strategy A)
- `task1_output/strategy_B/` — 100 generated posters (Strategy B)
- `task1_output/strategy_C/` — 100 generated posters (Strategy C, best)
- `task1_output/cases_strategy_C/success/` — top-5 success cases
- `task1_output/cases_strategy_C/failure/` — bottom-5 failure cases
- `task1_output/evaluation_report.json` — CLIP-Score / Visual Similarity / KID per strategy
- `hw1_p14942a08_task1/hw1_p14942a08_task1_images.zip` — 100 submission images (Strategy C)
- `hw1_p14942a08_task1/hw1_p14942a08_task1_prompts.json` — 100 prompts (Strategy C)

**Run:**

```bash
conda activate ai
python task1.py
```

> Requires a GPU with at least 16 GB VRAM. Prompt generation uses Qwen3-VL-2B-Instruct; image generation uses SDXL + IP-Adapter. Generated prompts are cached to `task1_prompts_cache.json` so re-running skips the Qwen inference stage.

---

## Task 2: RAG Page Retrieval

**Script:** `task2_multiview.py`  
**Public Leaderboard Score:** 0.493

**Approach:** Multi-View Page Representation with max fusion.  
Each of the 215 pages is represented by 4 views (full text / title / bullets / keywords),
producing 860 embeddings total. For each query, the page with the highest cosine similarity
across any of its 4 views is returned.

**Input:**
- `AI.pdf` — 215-page AI/ML lecture slide PDF
- `HW1_questions.json` — 200 questions

**Output:**
- `hw1_p14942a08_task2/submission_multiview_max.csv` — final Kaggle submission (best score)
- `hw1_p14942a08_task2/submission_multiview_weighted_sum.csv` — weighted sum fusion variant
- `task2_multiview_cache.npz` — cached 860-view embeddings (skip re-embedding on re-run)

**Run:**

```bash
conda activate ai
python task2_multiview.py
```

> On first run, all-mpnet-base-v2 encodes 860 view texts (~15 seconds). Embeddings are cached
> to `task2_multiview_cache.npz` for subsequent runs. The final submission CSV is
> `submission_multiview_max.csv`.

---

## Report

**Script:** `generate_report_pdf_v2.py`

Generates `hw1_p14942a08.pdf` (5 pages) from Task 1 evaluation results and Task 2 pipeline description.

```bash
conda activate ai
python generate_report_pdf_v2.py
```

**Requires** Task 1 outputs to already exist (`task1_output/`, `hw1_p14942a08_task1/hw1_p14942a08_task1_prompts.json`).

---

## File Structure

```
AI/
├── task1.py                            # Task 1: poster generation pipeline
├── task2_multiview.py                  # Task 2: RAG — multi-view page retrieval (best)
├── generate_report_pdf_v2.py           # PDF report generator
├── pairs.csv                           # Pair metadata (product/ref/title)
├── AI.pdf                              # Source PDF for Task 2
├── HW1_questions.json                  # 200 questions for Task 2
├── product/                            # Product images (p001–p100.png)
├── ref/                                # Reference poster images
├── task1_output/                       # Generated posters + evaluation
├── task2_multiview_cache.npz           # Cached multi-view embeddings
├── hw1_p14942a08_task1/
│   ├── hw1_p14942a08_task1_images.zip
│   └── hw1_p14942a08_task1_prompts.json
├── hw1_p14942a08_task2/
│   ├── submission_multiview_max.csv    # Final submission (score 0.493)
│   └── submission_multiview_weighted_sum.csv
└── hw1_p14942a08.pdf                   # Final report
```
