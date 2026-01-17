import os
import glob
import argparse
import random
import networkx as nx
import numpy as np
import pandas as pd
from conllu import parse
from networkx.generators.trees import random_labeled_tree
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import functools

def get_random_tree_distance(N):
    if N < 2:
        return 0.0
    try:
        rnd_seed = random.randint(0, 10**9)
        G_rand = random_labeled_tree(N, seed=rnd_seed)
        return nx.average_shortest_path_length(G_rand)
    except Exception:
        return None

def process_single_item(args):
    sentence, language_code, K = args

    G = nx.Graph()
    
    valid_tokens = [t for t in sentence if isinstance(t['id'], int)]
    
    for token in valid_tokens:
        G.add_node(token['id'])
        
    for token in valid_tokens:
        head = token['head']
        dep = token['id']
        if head != 0 and head in G.nodes:
            G.add_edge(head, dep)
            
    N = G.number_of_nodes()
    L = G.number_of_edges()
    
    if N < 2:
        return None
    if not nx.is_connected(G):
        return None

    d_real = nx.average_shortest_path_length(G)
    
    rho = 2.0 / N if N > 0 else 0
    d_min = 2.0 - rho
    
    random_distances = []
    for _ in range(K):
        d_val = get_random_tree_distance(N)
        if d_val is not None:
            random_distances.append(d_val)
            
    if not random_distances:
        return None
        
    d_rand = np.mean(random_distances)
    
    numerator = d_rand - d_real
    denominator = d_rand - d_min
    
    if denominator == 0:
        omega = float('nan')
    else:
        omega = numerator / denominator

    return {
        "language": language_code,
        "sent_id": sentence.metadata.get("sent_id", "unknown"),
        "N": N,
        "L": L,
        "d_real": round(d_real, 4),
        "d_min": round(d_min, 4),
        "d_rand": round(d_rand, 4),
        "omega": round(omega, 4)
    }

def load_data(input_dir):
    tasks = []
    search_pattern = os.path.join(input_dir, "UD_*-PUD", "*.conllu")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        search_pattern = os.path.join(input_dir, "*.conllu")
        files = sorted(glob.glob(search_pattern))

    print(f"Found {len(files)} treebank files.")
    
    for file_path in files:
        folder_name = os.path.basename(os.path.dirname(file_path))
        if "UD_" in folder_name:
            lang_name = folder_name.replace("UD_", "").replace("-PUD", "")
        else:
            lang_name = os.path.basename(file_path).split("_")[0]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sentences = parse(f.read())
                for sent in sentences:
                    tasks.append((sent, lang_name))
        except Exception as e:
            print(f"[Warning] Failed to read {file_path}: {e}")
            
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Parallel Syntactic Network Optimality Calculator")
    
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the root directory containing UD treebank folders (e.g., raw-data/ud-treebanks-v2.16)")
    parser.add_argument("-o", "--output", type=str, default=".",
                        help="Directory to save the output CSV (default: current dir)")
    parser.add_argument("-k", "--samples", type=int, default=200,
                        help="Number of random graphs for baseline simulation (K). Default 200.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of CPU cores to use. Default: All available.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return
        
    os.makedirs(args.output, exist_ok=True)
    output_csv = os.path.join(args.output, "syntactic_optimality_results.csv")

    print("Loading treebanks...")
    raw_tasks = load_data(args.input)
    total_sentences = len(raw_tasks)
    
    if total_sentences == 0:
        print("No sentences found. check your input path structure.")
        return

    print(f"Loaded {total_sentences} sentences. Preparing for parallel execution...")

    worker_args = [(sent, lang, args.samples) for sent, lang in raw_tasks]

    results = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        
        results_gen = list(tqdm(
            executor.map(process_single_item, worker_args), 
            total=total_sentences, 
            unit="sent",
            desc=f"Computing Omega (K={args.samples})"
        ))

    valid_results = [r for r in results_gen if r is not None]
    
    df = pd.DataFrame(valid_results)
    
    if not df.empty:
        cols = ["language", "sent_id", "N", "L", "d_real", "d_min", "d_rand", "omega"]
        df = df[cols]
        df.to_csv(output_csv, index=False)
        print("\nSuccess!")
        print(f"Processed: {len(df)}/{total_sentences} sentences.")
        print(f"Results saved to: {output_csv}")
    else:
        print("\nNo valid results generated (check if graphs are connected/valid).")

if __name__ == "__main__":
    main()
