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

def run_benchmark(dataset, models, min_delay=2, max_delay=100, n_images=50):
    # Load existing results if they exist to allow incremental updates
    results_path = "benchmark_results.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}
    
    print(f"Generating sequence of {n_images * 2} trials...")
    trials = run_task(dataset, min_delay, max_delay, n_images)
    
    for model in models:
        model_name = model.get_name()
        print(f"Running benchmark for: {model_name}...")
        
        model.reset()
        responses = []
        targets = []
        trial_delays = []
        
        for trial in trials:
            score = model.process_trial(trial["image"], trial["prompt"])
            response = 1 if score > 0.5 else 0 
            responses.append(response)
            targets.append(trial["target"])
            trial_delays.append(trial["delay"])
            
        model_metrics = calculate_metrics(responses, targets)
        
        delays_list = []
        hits_list = []
        for i in range(len(responses)):
            if targets[i] == 1:
                delays_list.append(trial_delays[i])
                hits_list.append(responses[i])
        
        results[model_name] = {
            "metrics": model_metrics,
            "delays": delays_list,
            "hits": hits_list
        }
        
        # Save after each model in case of crash
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", 
                        choices=["vit", "gemini", "gpt4o", "mambavision", "titans", "all"])
    parser.add_argument("--device", type=str, default="cuda")
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
        run_benchmark(dataset, models_to_test, min_delay=2, max_delay=100, n_images=50)
        print("Benchmark segment completed.")
