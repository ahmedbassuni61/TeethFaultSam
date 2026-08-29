import argparse
import os
import numpy as np
import json
import glob

def calculate_calculus_metrics(pred_labels, gt_labels):
    pred_np = np.asarray(pred_labels, dtype=bool)
    gt_np = np.asarray(gt_labels, dtype=bool)
    
    tp = np.sum(pred_np & gt_np)
    fp = np.sum(pred_np & ~gt_np)
    fn = np.sum(~pred_np & gt_np)
    tn = np.sum(~pred_np & ~gt_np)
    
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(dice),
        'accuracy': float(accuracy),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    }

def load_gt_calculus(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data.get('fault_labels', []))

def load_pred_calculus(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data.get('fault_labels', []))

def main(pred_dir, gt_dir):
    all_metrics = []
    
    for jaw in ['lower', 'upper']:
        gt_jaw_dir = os.path.join(gt_dir, jaw)
        if not os.path.isdir(gt_jaw_dir):
            gt_jaw_dir = os.path.join(gt_dir, 'calculus', jaw)
            if not os.path.isdir(gt_jaw_dir):
                continue
                
        pred_jaw_dir = os.path.join(pred_dir, jaw)
        if not os.path.isdir(pred_jaw_dir):
            continue
            
        case_dirs = sorted([d for d in glob.glob(f"{pred_jaw_dir}/*") if os.path.isdir(d)])
        for case_dir in case_dirs:
            case = os.path.basename(case_dir)
            pred_json = os.path.join(case_dir, f"{case}_{jaw}.json")
            
            # Find GT json
            gt_json = os.path.join(gt_jaw_dir, case, f"{case}_{jaw}.json")
            if not os.path.exists(gt_json):
                continue
                
            pred_labels = load_pred_calculus(pred_json)
            gt_labels = load_gt_calculus(gt_json)
            
            if len(pred_labels) != len(gt_labels):
                print(f"Warning: size mismatch for {case}_{jaw}. Pred: {len(pred_labels)}, GT: {len(gt_labels)}")
                continue
                
            metrics = calculate_calculus_metrics(pred_labels, gt_labels)
            all_metrics.append(metrics)
            
            print(f"[{case}_{jaw}] Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

    if not all_metrics:
        print("No matches found for evaluation.")
        return
        
    print("\n" + "="*50)
    print("           CALCULUS METRICS SUMMARY")
    print("="*50)
    
    avg_dice = np.mean([m['dice'] for m in all_metrics])
    avg_iou = np.mean([m['iou'] for m in all_metrics])
    avg_prec = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_acc = np.mean([m['accuracy'] for m in all_metrics])
    
    print(f"Total Cases Evaluated: {len(all_metrics)}")
    print(f"  - Average Dice      : {avg_dice:.4f}")
    print(f"  - Average IoU       : {avg_iou:.4f}")
    print(f"  - Average Precision : {avg_prec:.4f}")
    print(f"  - Average Recall    : {avg_recall:.4f}")
    print(f"  - Average Accuracy  : {avg_acc:.4f}")
    print("="*50)
    
    # Save detailed to CSV
    import csv
    csv_path = os.path.join(pred_dir, "evaluation_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dice', 'iou', 'precision', 'recall', 'accuracy', 'tp', 'fp', 'fn', 'tn'])
        for m in all_metrics:
            writer.writerow([m['dice'], m['iou'], m['precision'], m['recall'], m['accuracy'], m['tp'], m['fp'], m['fn'], m['tn']])
    print(f"Saved detailed metrics to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate 3D calculus segmentation metrics.")
    parser.add_argument('--pred_dir', type=str, required=True, help="Directory containing prediction JSONs.")
    parser.add_argument('--gt_dir', type=str, required=True, help="Directory containing ground truth JSONs.")
    args = parser.parse_args()
    
    main(args.pred_dir, args.gt_dir)
