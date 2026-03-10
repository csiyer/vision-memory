import json
import numpy as np
import os
from continuous_recognition import run_task
from stimuli import DirectoryDataset
from metrics import calculate_metrics
from evaluators.vlm_evaluators import GeminiEvaluator, OpenAIEvaluator
from evaluators.vision_evaluators import ViTEvaluator, RecurrentVisionEvaluator

import argparse
from evaluators.vision_evaluators import MambaVisionEvaluator, VisionTitansEvaluator

def run_benchmark(dataset, models, min_delay=2, max_delay=100, n_images=50, n_runs=1):
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

    for model in models:
        model_name = model.get_name()
        print(f"\n>>> Starting benchmark for: {model_name} ({n_runs} runs) <<<")
        
        # Initialize aggregate stats for this model
        all_responses = []
        all_targets = []
        all_delays = []
        image_stats = {} # name -> {"hits": 0, "total": 0}

        for run_idx in range(n_runs):
            if n_runs > 1:
                print(f"Run {run_idx+1}/{n_runs}...", end="\r")
            
            model.reset()
            # Generate a fresh sequence for each run
            trials = run_task(dataset, min_delay, max_delay, n_images)
            
            for trial in trials:
                score = model.process_trial(trial["image"], trial["prompt"])
                response = 1 if score > 0.5 else 0 
                
                # Global metrics
                all_responses.append(response)
                all_targets.append(trial["target"])
                if trial["target"] == 1:
                    all_delays.append(trial["delay"])

                # Per-image Stats (only for the 'repeat' trial)
                if trial["target"] == 1:
                    img_name = trial["metadata"]["name"]
                    if img_name not in image_stats:
                        image_stats[img_name] = {"hits": 0, "total": 0}
                    image_stats[img_name]["total"] += 1
                    if response == 1:
                        image_stats[img_name]["hits"] += 1

        print(f"\nCompleted {n_runs} runs for {model_name}.")
        
        # Calculate metrics over all runs
        model_metrics = calculate_metrics(all_responses, all_targets)
        
        # Prepare delay-hit pairs for plotting
        hits_by_delay = [all_responses[i] for i in range(len(all_targets)) if all_targets[i] == 1]
        
        # Prepare correlation data
        model_memorability = {}
        for img, stats in image_stats.items():
            model_memorability[img] = stats["hits"] / stats["total"] if stats["total"] > 0 else 0
            
        results[model_name] = {
            "metrics": model_metrics,
            "delays": all_delays,
            "hits": hits_by_delay,
            "image_performance": model_memorability,
            "lamem_scores": {img: lamem_scores.get(img, 0.5) for img in model_memorability}
        }
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", 
                        choices=["vit", "gemini", "gpt4o", "mambavision", "titans", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--n_images", type=int, default=100)
    args = parser.parse_args()

    dataset = DirectoryDataset("datasets/sample-lamem") 
    
    models_to_test = []
    if args.model in ["vit", "all"]:
        models_to_test.append(ViTEvaluator("vit_base_patch16_224", device=args.device))
    if args.model in ["gemini", "all"]:
        models_to_test.append(GeminiEvaluator("gemini-1.5-flash"))
    if args.model in ["gpt4o", "all"]:
        models_to_test.append(OpenAIEvaluator("gpt-4o"))
    if args.model in ["mambavision", "all"]:
        models_to_test.append(MambaVisionEvaluator("mambavision_tiny_1k", device=args.device))
    if args.model in ["titans", "all"]:
        models_to_test.append(VisionTitansEvaluator(device=args.device))
    
    if not models_to_test:
        print(f"Error: Model {args.model} not implemented or selected.")
    else:
        run_benchmark(dataset, models_to_test, 
                      min_delay=2, max_delay=100, 
                      n_images=args.n_images, 
                      n_runs=args.n_runs)
        print("\nBenchmark completed.")
