"""
Task 1: MLLM Poster Generation
- Use Qwen3-VL-2B-Instruct to extract style from ref posters and analyze product images
- Generate prompts using 3 strategies (A, B, C)
- Use SDXL + IP-Adapter to generate product posters
- Evaluate with CLIP-Score, CLIP visual similarity, KID
"""

import os
import json
import csv
import gc
import math
import shutil
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from PIL import Image
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/home/htiintern2502/powen/AI")
PRODUCT_DIR = BASE_DIR / "product"
REF_DIR    = BASE_DIR / "ref"
PAIRS_CSV  = BASE_DIR / "pairs.csv"
OUT_DIR    = BASE_DIR / "task1_output"
PROMPT_FILE = BASE_DIR / "hw1_task1_prompts.json"

STRATEGY_DIRS = {
    "A": OUT_DIR / "strategy_A",
    "B": OUT_DIR / "strategy_B",
    "C": OUT_DIR / "strategy_C",
}
for d in [OUT_DIR] + list(STRATEGY_DIRS.values()):
    d.mkdir(parents=True, exist_ok=True)

# ─── Load pairs ───────────────────────────────────────────────────────────────
pairs_df = pd.read_csv(PAIRS_CSV)
print(f"Loaded {len(pairs_df)} pairs")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Generate prompts with Qwen3-VL-2B-Instruct (3 strategies × 100)
# ══════════════════════════════════════════════════════════════════════════════
def load_qwen():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    print(f"Loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Qwen loaded.")
    return model, processor


def qwen_infer(model, processor, images: list, prompt_text: str, max_new_tokens=512) -> str:
    """Run Qwen2.5-VL inference with given images and text prompt."""
    from qwen_vl_utils import process_vision_info

    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


# ── Strategy prompts ──────────────────────────────────────────────────────────
def build_instruction_A(product_title: str) -> str:
    return (
        "You are given two images:\n"
        "Image 1: a reference poster\n"
        "Image 2: a product photo\n\n"
        "Based on these:\n"
        "1. Describe the color palette of the reference poster.\n"
        "2. Describe the layout structure (where are the main elements).\n"
        "3. Describe the overall mood and atmosphere.\n"
        f"4. Product title: \"{product_title}\"\n\n"
        "Now write a single concise English text-to-image prompt (1-3 sentences) "
        "that combines the style of the reference poster with the product, suitable for an image generation model. "
        "Output ONLY the final prompt, no extra text."
    )


def build_instruction_B(product_title: str) -> str:
    return (
        "You are given two images:\n"
        "Image 1: a reference poster\n"
        "Image 2: a product photo\n\n"
        f"Product title: \"{product_title}\"\n\n"
        "Analyze the reference poster and fill in each field below, then write the final prompt.\n"
        "palette: (list 3-5 main colors)\n"
        "composition: (describe layout, focal point, text area placement)\n"
        "mood: (describe emotional tone, e.g. elegant, vibrant, minimalist)\n"
        "text_placement: (where text appears on poster)\n"
        "product_emphasis: (how the product is highlighted)\n"
        "final_prompt: (a single text-to-image prompt combining all the above with the product)\n\n"
        "Output ONLY the value for 'final_prompt:' — do NOT include labels or other fields."
    )


def build_instruction_C(product_title: str) -> str:
    return (
        "You are given two images:\n"
        "Image 1: a reference advertising poster\n"
        "Image 2: a product photo\n\n"
        f"Product title: \"{product_title}\"\n\n"
        "Generate a commercial product advertisement prompt for a text-to-image model. "
        "Requirements: commercial poster style, clean product-centric layout, strong product visibility in the center, "
        "appealing headline region at top or bottom, premium advertisement aesthetic. "
        "Match the color palette and design mood of the reference poster. "
        "The product must be clearly recognizable and prominently displayed. "
        "Output ONLY the final English prompt (1-3 sentences), no extra text."
    )


# ── Generate all prompts ───────────────────────────────────────────────────────
def generate_all_prompts():
    prompt_cache_file = BASE_DIR / "task1_prompts_cache.json"
    if prompt_cache_file.exists():
        print("Loading cached prompts...")
        with open(prompt_cache_file) as f:
            return json.load(f)

    model, processor = load_qwen()
    all_prompts = {}  # {pair_id: {A: ..., B: ..., C: ...}}

    strategy_funcs = {
        "A": build_instruction_A,
        "B": build_instruction_B,
        "C": build_instruction_C,
    }

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Generating prompts"):
        pair_id  = str(row["pair_id"])
        prod_img = Image.open(PRODUCT_DIR / row["product_image"]).convert("RGB")
        ref_img  = Image.open(REF_DIR     / row["ref_image"]).convert("RGB")
        title    = row["product_title"]

        prompts = {}
        for strat, fn in strategy_funcs.items():
            instruction = fn(title)
            try:
                result = qwen_infer(model, processor, [ref_img, prod_img], instruction)
                # Clean up: remove any prefix labels if model added them
                for prefix in ["final_prompt:", "Final prompt:", "FINAL PROMPT:"]:
                    if prefix.lower() in result.lower():
                        idx = result.lower().index(prefix.lower())
                        result = result[idx + len(prefix):].strip()
                prompts[strat] = result
            except Exception as e:
                print(f"Error on pair {pair_id} strategy {strat}: {e}")
                prompts[strat] = f"A commercial poster for {title}, clean product-centric layout."

        all_prompts[pair_id] = prompts
        prod_img.close()
        ref_img.close()

    # Save cache
    with open(prompt_cache_file, "w") as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)
    print(f"Prompts saved to {prompt_cache_file}")

    # Free VRAM
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return all_prompts


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Generate posters with SDXL + IP-Adapter
# ══════════════════════════════════════════════════════════════════════════════
def load_sdxl_ipadapter():
    from diffusers import StableDiffusionXLPipeline, DDIMScheduler
    from transformers import CLIPVisionModelWithProjection

    print("Loading SDXL + IP-Adapter ...")

    # Load SDXL base
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load IP-Adapter for SDXL
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
    )
    pipe.set_ip_adapter_scale(0.6)
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    print("SDXL + IP-Adapter loaded.")
    return pipe


def generate_poster(pipe, prompt: str, product_img: Image.Image,
                    height=1024, width=1024) -> Image.Image:
    """Generate a poster conditioned on text prompt + product image."""
    result = pipe(
        prompt=prompt,
        ip_adapter_image=product_img,
        negative_prompt=(
            "blurry, low quality, distorted product, deformed, ugly, "
            "watermark, text overlay, cluttered background"
        ),
        num_inference_steps=30,
        guidance_scale=7.5,
        height=height,
        width=width,
    )
    return result.images[0]


def generate_all_posters(all_prompts: dict):
    gen_cache_file = BASE_DIR / "task1_gen_progress.json"
    generated = {}
    if gen_cache_file.exists():
        with open(gen_cache_file) as f:
            generated = json.load(f)

    pipe = load_sdxl_ipadapter()

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Generating posters"):
        pair_id = str(row["pair_id"])
        prod_img = Image.open(PRODUCT_DIR / row["product_image"]).convert("RGB")

        for strat in ["A", "B", "C"]:
            out_path = STRATEGY_DIRS[strat] / f"{pair_id}.jpg"
            if out_path.exists():
                continue

            prompt = all_prompts.get(pair_id, {}).get(strat, "")
            if not prompt:
                continue

            try:
                poster = generate_poster(pipe, prompt, prod_img)
                # Resize to 224×224
                poster_resized = poster.resize((224, 224), Image.LANCZOS)
                poster_resized.save(str(out_path), "JPEG", quality=95)
                generated[f"{pair_id}_{strat}"] = str(out_path)
            except Exception as e:
                print(f"Error generating {pair_id} strategy {strat}: {e}")

        prod_img.close()

        # Save progress periodically
        if int(pair_id) % 10 == 0:
            with open(gen_cache_file, "w") as f:
                json.dump(generated, f)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("All posters generated.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Evaluation — CLIP-Score, CLIP Visual Similarity, KID
# ══════════════════════════════════════════════════════════════════════════════
def load_clip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B/32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B/32")
    model = model.to("cuda").eval()
    return model, preprocess, tokenizer


def get_image_features(clip_model, preprocess, img: Image.Image) -> torch.Tensor:
    x = preprocess(img).unsqueeze(0).to("cuda")
    with torch.no_grad():
        feats = clip_model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu()


def get_text_features(clip_model, tokenizer, text: str) -> torch.Tensor:
    tokens = tokenizer([text]).to("cuda")
    with torch.no_grad():
        feats = clip_model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum())


def compute_kid(real_feats: np.ndarray, fake_feats: np.ndarray, n_subsets=10, subset_size=50) -> float:
    """Compute KID via polynomial MMD on feature arrays (N x D)."""
    def poly_mmd(x, y, degree=3, gamma=None, coef0=1):
        if gamma is None:
            gamma = 1.0 / x.shape[1]
        kxx = (gamma * x @ x.T + coef0) ** degree
        kyy = (gamma * y @ y.T + coef0) ** degree
        kxy = (gamma * x @ y.T + coef0) ** degree
        m = x.shape[0]
        n = y.shape[0]
        mmd = (kxx.sum() - kxx.trace()) / (m * (m - 1)) \
            + (kyy.sum() - kyy.trace()) / (n * (n - 1)) \
            - 2 * kxy.mean()
        return float(mmd)

    n_real = len(real_feats)
    n_fake = len(fake_feats)
    subset_size = min(subset_size, n_real, n_fake)
    scores = []
    rng = np.random.default_rng(42)
    for _ in range(n_subsets):
        idx_r = rng.choice(n_real, subset_size, replace=False)
        idx_f = rng.choice(n_fake, subset_size, replace=False)
        scores.append(poly_mmd(real_feats[idx_r], fake_feats[idx_f]))
    return float(np.mean(scores))


def evaluate_strategies(all_prompts: dict):
    print("\n=== Evaluation ===")
    clip_model, preprocess, tokenizer = load_clip()

    results = {}
    all_ref_feats = []
    # Pre-compute reference image features
    print("Computing reference image features...")
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
        ref_img = Image.open(REF_DIR / row["ref_image"]).convert("RGB")
        feats = get_image_features(clip_model, preprocess, ref_img)
        all_ref_feats.append(feats.numpy()[0])
        ref_img.close()
    ref_feats_np = np.array(all_ref_feats)

    per_pair_scores = {}  # {pair_id: {strategy: {clip_score, vis_sim}}}

    for strat in ["A", "B", "C"]:
        print(f"\nEvaluating Strategy {strat}...")
        clip_scores = []
        vis_sims = []
        gen_feats_list = []

        for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
            pair_id = str(row["pair_id"])
            gen_path = STRATEGY_DIRS[strat] / f"{pair_id}.jpg"
            if not gen_path.exists():
                continue

            gen_img  = Image.open(gen_path).convert("RGB")
            prod_img = Image.open(PRODUCT_DIR / row["product_image"]).convert("RGB")
            prompt   = all_prompts.get(pair_id, {}).get(strat, row["product_title"])

            gen_feats  = get_image_features(clip_model, preprocess, gen_img)
            prod_feats = get_image_features(clip_model, preprocess, prod_img)
            text_feats = get_text_features(clip_model, tokenizer, prompt)

            cs  = cosine_sim(gen_feats, text_feats)   # CLIP-Score (text-image)
            vs  = cosine_sim(gen_feats, prod_feats)   # Visual similarity to product

            clip_scores.append(cs)
            vis_sims.append(vs)
            gen_feats_list.append(gen_feats.numpy()[0])

            if pair_id not in per_pair_scores:
                per_pair_scores[pair_id] = {}
            per_pair_scores[pair_id][strat] = {
                "clip_score": cs,
                "vis_sim": vs,
                "combined": (cs + vs) / 2,
            }

            gen_img.close()
            prod_img.close()

        gen_feats_np = np.array(gen_feats_list)
        kid = compute_kid(ref_feats_np[:len(gen_feats_np)], gen_feats_np)

        results[strat] = {
            "clip_score_mean": float(np.mean(clip_scores)),
            "clip_score_std":  float(np.std(clip_scores)),
            "vis_sim_mean":    float(np.mean(vis_sims)),
            "vis_sim_std":     float(np.std(vis_sims)),
            "kid":             kid,
        }
        print(f"  Strategy {strat}: CLIP-Score={results[strat]['clip_score_mean']:.4f}, "
              f"VisSim={results[strat]['vis_sim_mean']:.4f}, KID={results[strat]['kid']:.6f}")

    del clip_model
    gc.collect()
    torch.cuda.empty_cache()
    return results, per_pair_scores


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Select best/worst cases from best strategy
# ══════════════════════════════════════════════════════════════════════════════
def select_cases(results: dict, per_pair_scores: dict, all_prompts: dict):
    # Find best strategy: highest (CLIP-Score + VisSim) / 2 - KID (normalized)
    strategy_rank = []
    for strat, r in results.items():
        score = (r["clip_score_mean"] + r["vis_sim_mean"]) / 2 - r["kid"] * 10
        strategy_rank.append((strat, score, r))
    strategy_rank.sort(key=lambda x: x[1], reverse=True)
    best_strat = strategy_rank[0][0]
    print(f"\nBest strategy: {best_strat}")

    # Collect per-pair combined scores for best strategy
    pair_scores = []
    for pair_id, scores_by_strat in per_pair_scores.items():
        if best_strat in scores_by_strat:
            pair_scores.append((pair_id, scores_by_strat[best_strat]["combined"]))

    pair_scores.sort(key=lambda x: x[1], reverse=True)
    top5    = pair_scores[:5]
    bottom5 = pair_scores[-5:]

    cases_dir = OUT_DIR / f"cases_strategy_{best_strat}"
    success_dir = cases_dir / "success"
    failure_dir = cases_dir / "failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    for pair_id, score in top5:
        src = STRATEGY_DIRS[best_strat] / f"{pair_id}.jpg"
        if src.exists():
            shutil.copy(src, success_dir / f"{pair_id}_score{score:.3f}.jpg")

    for pair_id, score in bottom5:
        src = STRATEGY_DIRS[best_strat] / f"{pair_id}.jpg"
        if src.exists():
            shutil.copy(src, failure_dir / f"{pair_id}_score{score:.3f}.jpg")

    print(f"Top-5 (success): {[p for p, _ in top5]}")
    print(f"Bottom-5 (failure): {[p for p, _ in bottom5]}")
    return best_strat, top5, bottom5


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Save final prompts JSON
# ══════════════════════════════════════════════════════════════════════════════
def save_prompts_json(all_prompts: dict):
    # Format: {pair_id: {strategy_A: ..., strategy_B: ..., strategy_C: ...}}
    formatted = {}
    for pair_id, strats in all_prompts.items():
        formatted[pair_id] = {
            "strategy_A": strats.get("A", ""),
            "strategy_B": strats.get("B", ""),
            "strategy_C": strats.get("C", ""),
        }
    with open(PROMPT_FILE, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)
    print(f"Prompts JSON saved to {PROMPT_FILE}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("TASK 1: MLLM Poster Generation")
    print("=" * 60)

    # Step 1: Generate prompts
    print("\n[Step 1] Generating prompts (3 strategies × 100 pairs)...")
    all_prompts = generate_all_prompts()
    save_prompts_json(all_prompts)

    # Step 2: Generate posters
    print("\n[Step 2] Generating product posters (SDXL + IP-Adapter)...")
    generate_all_posters(all_prompts)

    # Step 3: Evaluate
    print("\n[Step 3] Evaluating strategies...")
    results, per_pair_scores = evaluate_strategies(all_prompts)

    # Step 4: Select cases
    print("\n[Step 4] Selecting best/worst cases...")
    best_strat, top5, bottom5 = select_cases(results, per_pair_scores, all_prompts)

    # Save evaluation report
    report = {
        "strategy_results": results,
        "best_strategy": best_strat,
        "top5_success": [{"pair_id": p, "combined_score": s} for p, s in top5],
        "bottom5_failure": [{"pair_id": p, "combined_score": s} for p, s in bottom5],
    }
    report_path = OUT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Posters saved in: {OUT_DIR}")
    print(f"Strategy dirs: strategy_A/, strategy_B/, strategy_C/")
    print(f"Prompts JSON: {PROMPT_FILE}")
    print(f"Evaluation report: {report_path}")
    print("\nStrategy Comparison:")
    for strat, r in results.items():
        print(f"  {strat}: CLIP={r['clip_score_mean']:.4f}, VisSim={r['vis_sim_mean']:.4f}, KID={r['kid']:.6f}")
    print(f"\nBest strategy: {best_strat}")
    print("=" * 60)
