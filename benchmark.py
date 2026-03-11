import json
import os
import torch
from continuous_recognition import run_task
from stimuli import DirectoryDataset
from metrics import calculate_metrics
from evaluators.vlm_evaluators import GeminiEvaluator, OpenAIEvaluator, Qwen2VLEvaluator, InternVLEvaluator

import argparse

def run_benchmark(dataset, model_names, min_delay=2, max_delay=100, n_images=50, n_runs=1):
    results_path = "benchmark_results.json"
    scores_path = "datasets/sample_lamem_scores.json"
    
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    lamem_scores = {}
    if os.path.exists(scores_path):
        with open(scores_path, "r") as f:
            lamem_scores = json.load(f)

    for model_name_key in model_names:
        # Instantiate model ONLY when needed to save VRAM
        print(f"\n--- Loading Model: {model_name_key} ---")
        if model_name_key == "gemini":
            model = GeminiEvaluator("gemini-1.5-pro")
        elif model_name_key == "gpt4o":
            model = OpenAIEvaluator("gpt-4o-mini")
        elif model_name_key == "qwen2":
            model = Qwen2VLEvaluator()
        elif model_name_key == "internvl2":
            model = InternVLEvaluator()
        else:
            continue

        model_display_name = model.get_name()
        print(f"\n>>> Starting benchmark for: {model_display_name} ({n_runs} runs) <<<")
        
        all_responses = []
        all_targets = []
        all_delays = []
        image_stats = {} 

        for run_idx in range(n_runs):
            if n_runs > 1:
                print(f"Run {run_idx+1}/{n_runs}...", end="\r")
            
            model.reset()
            trials = run_task(dataset, min_delay, max_delay, n_images)
            
            for trial in trials:
                score = model.process_trial(trial["image"], trial["prompt"])
                response = 1 if score > 0.5 else 0 
                
                all_responses.append(response)
                all_targets.append(trial["target"])
                if trial["target"] == 1:
                    all_delays.append(trial["delay"])

                if trial["target"] == 1:
                    img_name = trial["metadata"]["name"]
                    if img_name not in image_stats:
                        image_stats[img_name] = {"hits": 0, "total": 0}
                    image_stats[img_name]["total"] += 1
                    if response == 1:
                        image_stats[img_name]["hits"] += 1

        print(f"\nCompleted {n_runs} runs for {model_display_name}.")
        
        model_metrics = calculate_metrics(all_responses, all_targets)
        hits_by_delay = [all_responses[i] for i in range(len(all_targets)) if all_targets[i] == 1]
        
        model_memorability = {}
        for img, stats in image_stats.items():
            model_memorability[img] = stats["hits"] / stats["total"] if stats["total"] > 0 else 0
            
        results[model_display_name] = {
            "metrics": model_metrics,
            "delays": all_delays,
            "hits": hits_by_delay,
            "image_performance": model_memorability,
            "lamem_scores": {img: lamem_scores.get(img, 0.5) for img in model_memorability}
        }
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
            
        # CLEAR MEMORY before next model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", 
                        choices=["gemini", "gpt4o", "qwen2", "internvl2", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--n_images", type=int, default=50)
    args = parser.parse_args()

    dataset = DirectoryDataset("datasets/sample-lamem") 
    
    selected_models = []
    if args.model == "all":
        selected_models = ["gemini", "gpt4o", "qwen2", "internvl2"]
    else:
        selected_models = [args.model]
    
    run_benchmark(dataset, selected_models, 
                  min_delay=2, max_delay=100, 
                  n_images=args.n_images, 
                  n_runs=args.n_runs)
