import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import difflib
import rapidfuzz


def extract_user_prompt_and_response(item: Dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract user prompt (instruction) and raw response from a single attempt item.
    Same logic as create_SFT_dataset.py.
    """
    instruction = None
    output = None
    
    if 'prompt' in item and isinstance(item['prompt'], list):
        for prompt_item in item['prompt']:
            if isinstance(prompt_item, dict) and prompt_item.get('role') == 'user':
                instruction = prompt_item.get('content')
                break
    
    output = item.get('raw_response')
    
    return instruction, output


def extract_all_attempts_from_full_run(file_path: str, is_success: bool) -> List[Dict[str, Any]]:
    """
    Extract all attempts from a full_run JSON file with their metrics.
    
    Returns a list of dicts, each containing:
        - instruction: the user prompt text
        - response: the raw model response
        - efficiency_total: overall cleanup efficiency (from evaluation_metrics)
        - actual_cleaned_kb_total: total KB freed
        - success: whether this was from a success_full_run
        - phase: which phase this attempt belongs to
        - source_file: the originating file name
    """
    attempts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  - Error reading {file_path}: {e}")
        return []
    
    # Get evaluation metrics (attached by data_collector)
    eval_metrics = data.get('evaluation_metrics', {})
    efficiency_total = eval_metrics.get('efficiency_total', 0)
    actual_cleaned_kb_total = eval_metrics.get('actual_cleaned_kb_total', 0)
    
    # For failed runs, set efficiency to -1 (always loses in pairing)
    if not is_success:
        efficiency_total = -1
        actual_cleaned_kb_total = 0
    
    source_file = os.path.basename(file_path)
    
    # Extract from exploration phase
    if 'exploration' in data and isinstance(data['exploration'], list):
        for item in data['exploration']:
            instruction, response = extract_user_prompt_and_response(item)
            if instruction and response:
                attempts.append({
                    'instruction': instruction,
                    'response': response,
                    'efficiency_total': efficiency_total,
                    'actual_cleaned_kb_total': actual_cleaned_kb_total,
                    'success': is_success,
                    'phase': 'exploration',
                    'source_file': source_file
                })
    
    # Extract from strategy_generation phase
    if 'strategy_generation' in data and isinstance(data['strategy_generation'], list):
        for item in data['strategy_generation']:
            instruction, response = extract_user_prompt_and_response(item)
            if instruction and response:
                attempts.append({
                    'instruction': instruction,
                    'response': response,
                    'efficiency_total': efficiency_total,
                    'actual_cleaned_kb_total': actual_cleaned_kb_total,
                    'success': is_success,
                    'phase': 'strategy_generation',
                    'source_file': source_file
                })
    
    # Extract from strategy_execution phase
    if 'strategy_execution' in data and isinstance(data['strategy_execution'], list):
        for strategy in data['strategy_execution']:
            if 'all_code_attempts' in strategy and isinstance(strategy['all_code_attempts'], list):
                for attempt in strategy['all_code_attempts']:
                    instruction, response = extract_user_prompt_and_response(attempt)
                    if instruction and response:
                        attempts.append({
                            'instruction': instruction,
                            'response': response,
                            'efficiency_total': efficiency_total,
                            'actual_cleaned_kb_total': actual_cleaned_kb_total,
                            'success': is_success,
                            'phase': 'strategy_execution',
                            'source_file': source_file
                        })
    
    return attempts


def load_all_attempts(success_folder: str, failed_folder: str) -> List[Dict[str, Any]]:
    """Load all attempts from both success and failed folders."""
    all_attempts = []
    
    # Load success runs
    if os.path.isdir(success_folder):
        print(f"Loading success runs from: {success_folder}")
        success_files = [f for f in os.listdir(success_folder) if f.endswith('.json')]
        for file_name in success_files:
            file_path = os.path.join(success_folder, file_name)
            attempts = extract_all_attempts_from_full_run(file_path, is_success=True)
            all_attempts.extend(attempts)
        print(f"  - Loaded {len(success_files)} success files")
    else:
        print(f"Warning: Success folder not found: {success_folder}")
    
    # Load failed runs
    if os.path.isdir(failed_folder):
        print(f"Loading failed runs from: {failed_folder}")
        failed_files = [f for f in os.listdir(failed_folder) if f.endswith('.json')]
        for file_name in failed_files:
            file_path = os.path.join(failed_folder, file_name)
            attempts = extract_all_attempts_from_full_run(file_path, is_success=False)
            all_attempts.extend(attempts)
        print(f"  - Loaded {len(failed_files)} failed files")
    else:
        print(f"Warning: Failed folder not found: {failed_folder}")
    
    return all_attempts


def group_by_phase(attempts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group attempts by phase."""
    groups = defaultdict(list)
    for attempt in attempts:
        groups[attempt['phase']].append(attempt)
    return groups


def has_valid_special_env_info(instruction: str, phase: str) -> bool:
    """
    Check if the instruction contains valid <SpecialEnvInfo>.
    - For exploration phase, we allow empty info.
    - For strategy/execution phases, we filter out if it's empty or contains common error messages.
    """
    if phase == "exploration":
        return True
    
    # Extract content between tags
    import re
    match = re.search(r'<SpecialEnvInfo>(.*?)</SpecialEnvInfo>', instruction, re.DOTALL)
    if not match:
        return False
    
    content = match.group(1).strip()
    
    # Filter if empty or just whitespace
    if not content:
        return False
        
    # Filter if contains common failure indicators
    failure_indicators = [
        "hostname: unrecognized option",
        "[]",
        "--------------------------------------------------------------------------------"
    ]
    
    # If the content is too short and doesn't look like actual info (like just a \n)
    if len(content) < 5 and not any(ind in content for ind in ["172", "eth0", "lo"]):
        return False

    return True


def calculate_similarity(a: str, b: str) -> float:
    """Calculate string similarity between two instructions."""
    return rapidfuzz.fuzz.ratio(a, b) / 100.0


def create_preference_pairs(
    attempts: List[Dict[str, Any]],
    min_efficiency_diff: float = 1.0,
    max_pairs: int = 15000,
    similarity_threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Create DPO preference pairs using Similarity-Based Clustering.
    Optimized to handle large datasets by grouping identical instructions.
    """
    if len(attempts) < 2:
        print(f"  Not enough attempts to create pairs ({len(attempts)})")
        return []
    
    # Group by phase first
    phase_groups = group_by_phase(attempts)
    
    all_pairs = []
    
    for phase, phase_attempts in phase_groups.items():
        print(f"\n  Phase: {phase} ({len(phase_attempts)} attempts)")
        
        # Filter noisy data
        valid_attempts = [
            a for a in phase_attempts 
            if has_valid_special_env_info(a['instruction'], phase)
        ]
        
        filtered_count = len(phase_attempts) - len(valid_attempts)
        if filtered_count > 0:
            print(f"    Filtered out {filtered_count} noisy/empty attempts")
            
        if len(valid_attempts) < 2:
            continue

        # Group by instruction to avoid redundant similarity checks
        # key: instruction, value: best effort (highest efficiency)
        success_map = {}
        for a in valid_attempts:
            if a['success']:
                instr = a['instruction']
                if instr not in success_map or a['efficiency_total'] > success_map[instr]['efficiency_total']:
                    success_map[instr] = a
        
        failed_map = {}
        for a in valid_attempts:
            if not a['success']:
                instr = a['instruction']
                # Link failed attempt, we don't care about efficiency (it is -1)
                failed_map[instr] = a
        
        print(f"    Unique Instructions: SUCCESS={len(success_map)}, FAILED={len(failed_map)}")
        
        if not success_map or not failed_map:
            print(f"    Skipping phase {phase}: missing either success or failed examples.")
            continue

        phase_pairs = []
        unique_failed_instructions = list(failed_map.keys())
        
        # Find Similarity matching
        for instr_success, chosen in success_map.items():
            # Quick check for exact match
            if instr_success in failed_map:
                best_match = failed_map[instr_success]
                highest_sim = 1.0
            else:
                # Fuzzy match
                best_match = None
                highest_sim = -1.0
                
                # Use rapidfuzz for significantly faster fuzzy matching
                result = rapidfuzz.process.extractOne(
                    instr_success, 
                    unique_failed_instructions, 
                    score_cutoff=similarity_threshold * 100
                )
                if result:
                    best_instr_failed, score, _ = result
                    best_match = failed_map[best_instr_failed]
                    highest_sim = score / 100.0

            # Create pair if similarity is decent
            if best_match:
                # Basic check for efficiency difference
                eff_diff = chosen['efficiency_total'] - best_match['efficiency_total']
                if eff_diff >= min_efficiency_diff:
                    pair = _create_pair(chosen, best_match, f"similarity_match_{phase}")
                    if pair:
                        phase_pairs.append(pair)
        
        print(f"    Created {len(phase_pairs)} similarity-based pairs for phase '{phase}'")
        all_pairs.extend(phase_pairs)
    
    # Shuffle and cap
    random.shuffle(all_pairs)
    final_pairs = all_pairs[:max_pairs]
    
    return final_pairs


def _create_pair(chosen: Dict, rejected: Dict, pair_type: str) -> Optional[Dict[str, Any]]:
    """Create a single DPO preference pair."""
    # Skip if responses are identical
    if chosen['response'] == rejected['response']:
        return None
    
    return {
        'instruction': chosen['instruction'],
        'chosen': chosen['response'],
        'rejected': rejected['response'],
        'chosen_efficiency': chosen['efficiency_total'],
        'rejected_efficiency': rejected['efficiency_total'],
        'efficiency_difference': chosen['efficiency_total'] - rejected['efficiency_total'],
        'chosen_cleaned_kb': chosen.get('actual_cleaned_kb_total', 0),
        'rejected_cleaned_kb': rejected.get('actual_cleaned_kb_total', 0),
        'pair_type': pair_type,
        'phase': chosen['phase']
    }


def save_dpo_dataset(pairs: List[Dict[str, Any]], output_file: str):
    """Save DPO pairs as JSONL file."""
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"\nSaved {len(pairs)} DPO pairs to: {output_file}")


def print_analysis(pairs: List[Dict[str, Any]]):
    if not pairs:
        print("\nNo pairs to analyze.")
        return
    
    print(f"\n{'='*60}")
    print("DPO DATASET ANALYSIS")
    print(f"{'='*60}")
    print(f"Total pairs: {len(pairs)}")
    
    # By pair type
    pair_types = defaultdict(int)
    for p in pairs:
        pair_types[p.get('pair_type', 'unknown')] += 1
    print("\nPair types:")
    for pt, count in sorted(pair_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pt}: {count}")
    
    # By phase
    phases = defaultdict(int)
    for p in pairs:
        phases[p.get('phase', 'unknown')] += 1
    print("\nPhases:")
    for phase, count in sorted(phases.items(), key=lambda x: x[1], reverse=True):
        print(f"  {phase}: {count}")
    
    # Efficiency stats
    chosen_effs = [p['chosen_efficiency'] for p in pairs]
    rejected_effs = [p['rejected_efficiency'] for p in pairs]
    diffs = [p['efficiency_difference'] for p in pairs]
    
    print(f"\nChosen efficiency:  avg={sum(chosen_effs)/len(chosen_effs):.2f}%, "
          f"range={min(chosen_effs):.2f}% - {max(chosen_effs):.2f}%")
    print(f"Rejected efficiency: avg={sum(rejected_effs)/len(rejected_effs):.2f}%, "
          f"range={min(rejected_effs):.2f}% - {max(rejected_effs):.2f}%")
    print(f"Efficiency gap:    avg={sum(diffs)/len(diffs):.2f}%, "
          f"range={min(diffs):.2f}% - {max(diffs):.2f}%")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Create DPO preference pairs from collected full_run logs'
    )
    parser.add_argument(
        '--success-folder', required=True,
        help='Path to success_full_run/ folder'
    )
    parser.add_argument(
        '--failed-folder', required=True,
        help='Path to failed_full_run/ folder'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output JSONL file path for DPO dataset'
    )
    parser.add_argument(
        '--min-efficiency-diff', type=float, default=1.0,
        help='Minimum efficiency difference between chosen and rejected (default: 1.0%%)'
    )
    parser.add_argument(
        '--max-pairs', type=int, default=4500,
        help='Maximum number of pairs to create (default: 4500)'
    )
    parser.add_argument(
        '--similarity-threshold', type=float, default=0.8,
        help='Minimum instruction similarity for pairing (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("DPO DATASET CREATOR")
    print("="*60)
    print(f"Success folder: {args.success_folder}")
    print(f"Failed folder:  {args.failed_folder}")
    print(f"Output:         {args.output}")
    print(f"Min efficiency diff: {args.min_efficiency_diff}%")
    print(f"Max pairs:      {args.max_pairs}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print("="*60)
    
    # Step 1: Load all attempts
    print("\nLoading data...")
    all_attempts = load_all_attempts(args.success_folder, args.failed_folder)
    
    if not all_attempts:
        print("No data loaded! Check your folder paths.")
        return
    
    success_count = sum(1 for a in all_attempts if a['success'])
    failed_count = sum(1 for a in all_attempts if not a['success'])
    print(f"\nTotal attempts loaded: {len(all_attempts)}")
    print(f"  From success runs: {success_count}")
    print(f"  From failed runs:  {failed_count}")
    
    # Step 2: Create preference pairs
    print("\nCreating preference pairs...")
    pairs = create_preference_pairs(
        all_attempts,
        min_efficiency_diff=args.min_efficiency_diff,
        max_pairs=args.max_pairs,
        similarity_threshold=args.similarity_threshold
    )
    
    if not pairs:
        print("No preference pairs created! You may need more data or a lower --min-efficiency-diff.")
        return
    
    # Step 3: Save dataset
    save_dpo_dataset(pairs, args.output)
    
    # Step 4: Print analysis
    print_analysis(pairs)
    
    print("\nDPO dataset creation complete!")


if __name__ == "__main__":
    main()
