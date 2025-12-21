import json
import sys

def load_json(path):
    print(f"Loading file: {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: file not found.")
        return data

def compute_recall(gt_data, pred_data, k=10):
    # 3. Sanity Check
    if len(gt_data) != len(pred_data):
        print(f"Warning: Query count mismatch!")
        print(f"Ground Truth: {len(gt_data)} queries")
        print(f"Predictions:  {len(pred_data)} queries")
        # We proceed with the minimum length to avoid crashing
        num_queries = min(len(gt_data), len(pred_data))
    else:
        num_queries = len(gt_data)
        print(f"Evaluating {num_queries} queries...")

    total_recall = 0.0

    # 4. Compute Recall Loop
    for i in range(num_queries):
        # Get the lists (arrays) for the i-th query
        # Your format is [[id1, id2...], [id1, id2...]]
        gt_ids = gt_data[i]
        pred_ids = pred_data[i]

        # Take strictly the top K of the predictions
        # (Ground truth is usually taken as-is, or top K if it's a large list)
        gt_set = set(gt_ids) 
        pred_set = set(pred_ids[:k])

        # Intersection: How many predicted IDs are in the Ground Truth?
        match_count = len(gt_set.intersection(pred_set))

        # Recall Formula: Matches / Total Possible Relevant Items
        if len(gt_set) > 0:
            recall = match_count / len(gt_set)
        else:
            recall = 0.0 # Avoid division by zero if GT is empty
        
        total_recall += recall

    # 5. Average
    if num_queries > 0:
        avg_recall = total_recall / num_queries
        print("=" * 40)
        print(f"Recall@{k}: {avg_recall:.4f}")
        print("=" * 40)
    else:
        print("No queries evaluated.")

def compute_recall_against_gt(pred_data):
    gt_filename = "flat_wiki.json"
    gt_data = load_json(gt_filename)
    compute_recall(gt_data, pred_data, k=10)

if __name__ == "__main__":
    pass
    # Replace these with your actual filenames