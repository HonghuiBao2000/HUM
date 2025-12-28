#!/usr/bin/env python3
"""
Parse results.log file, extract recall and ndcg metrics
Supports three types of results: Overall, Long-tail, Hetero
"""

import re
import pandas as pd
import json

def parse_results_log(log_file_path):
    """
    Parse results.log file, extract recall and ndcg metrics
    """
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define six domains
    domains = [
        'Industrial_and_Scientific',
        'Automotive', 
        'Tools_and_Home_Improvement',
        'Office_Products',
        'Books',
        'CDs_and_Vinyl'
    ]
    
    # Dictionary to store results
    results = {
        'Overall': {},
        'Long-tail': {},
        'Hetero': {}
    }
    
    # Regular expression to match recall and ndcg metrics
    recall_pattern = r'Recall@5:([\d.]+) ---- Recall@10:([\d.]+)'
    ndcg_pattern = r'NDCG@5:([\d.]+) ---- NDCG@10:([\d.]+)'
    
    # Split content into three parts
    sections = content.split('----------Overall End----------')
    
    # Process first part: Overall results
    overall_section = sections[0]
    current_domain = None
    
    for line in overall_section.split('\n'):
        # Check if it's a domain title
        if '-----------------' in line and any(domain in line for domain in domains):
            for domain in domains:
                if domain in line:
                    current_domain = domain
                    break
        
        # Extract recall and ndcg metrics
        if current_domain and 'Recall@5:' in line:
            recall_match = re.search(recall_pattern, line)
            ndcg_match = re.search(ndcg_pattern, line)
            
            if recall_match and ndcg_match:
                if current_domain not in results['Overall']:
                    results['Overall'][current_domain] = {}
                
                results['Overall'][current_domain] = {
                    'Recall@5': float(recall_match.group(1)),
                    'Recall@10': float(recall_match.group(2)),
                    'NDCG@5': float(ndcg_match.group(1)),
                    'NDCG@10': float(ndcg_match.group(2))
                }
    
    # Process second part: Long-tail results
    if len(sections) > 1:
        longtail_section = sections[1]
        current_domain = None
        
        for line in longtail_section.split('\n'):
            if '-----------------' in line and any(domain in line for domain in domains):
                for domain in domains:
                    if domain in line:
                        current_domain = domain
                        break
            
            if current_domain and 'Recall@5:' in line:
                recall_match = re.search(recall_pattern, line)
                ndcg_match = re.search(ndcg_pattern, line)
                
                if recall_match and ndcg_match:
                    if current_domain not in results['Long-tail']:
                        results['Long-tail'][current_domain] = {}
                    
                    results['Long-tail'][current_domain] = {
                        'Recall@5': float(recall_match.group(1)),
                        'Recall@10': float(recall_match.group(2)),
                        'NDCG@5': float(ndcg_match.group(1)),
                        'NDCG@10': float(ndcg_match.group(2))
                    }
    
    # Process third part: Hetero results
    if len(sections) > 2:
        hetero_section = sections[2]
        current_domain = None
        
        for line in hetero_section.split('\n'):
            if '-----------------' in line and any(domain in line for domain in domains):
                for domain in domains:
                    if domain in line:
                        current_domain = domain
                        break
            
            if current_domain and 'Recall@5:' in line:
                recall_match = re.search(recall_pattern, line)
                ndcg_match = re.search(ndcg_pattern, line)
                
                if recall_match and ndcg_match:
                    if current_domain not in results['Hetero']:
                        results['Hetero'][current_domain] = {}
                    
                    results['Hetero'][current_domain] = {
                        'Recall@5': float(recall_match.group(1)),
                        'Recall@10': float(recall_match.group(2)),
                        'NDCG@5': float(ndcg_match.group(1)),
                        'NDCG@10': float(ndcg_match.group(2))
                    }
    
    return results

def create_dataframe(results):
    """
    Convert results to DataFrame format
    """
    data = []
    
    for result_type, domains_data in results.items():
        for domain, metrics in domains_data.items():
            row = {
                'Result_Type': result_type,
                'Domain': domain,
                'Recall@5': metrics['Recall@5'],
                'Recall@10': metrics['Recall@10'],
                'NDCG@5': metrics['NDCG@5'],
                'NDCG@10': metrics['NDCG@10']
            }
            data.append(row)
    
    return pd.DataFrame(data)

def create_wide_format_dataframe(results):
    """
    Create wide format DataFrame, one row per domain, metrics as columns
    """
    domains = [
        'Industrial_and_Scientific',
        'Automotive', 
        'Tools_and_Home_Improvement',
        'Office_Products',
        'Books',
        'CDs_and_Vinyl'
    ]
    
    data = []
    
    for domain in domains:
        row = {'Domain': domain}
        
        for result_type in ['Overall', 'Long-tail', 'Hetero']:
            if domain in results[result_type]:
                metrics = results[result_type][domain]
                row[f'{result_type}_Recall@5'] = metrics['Recall@5']
                row[f'{result_type}_Recall@10'] = metrics['Recall@10']
                row[f'{result_type}_NDCG@5'] = metrics['NDCG@5']
                row[f'{result_type}_NDCG@10'] = metrics['NDCG@10']
            else:
                row[f'{result_type}_Recall@5'] = None
                row[f'{result_type}_Recall@10'] = None
                row[f'{result_type}_NDCG@5'] = None
                row[f'{result_type}_NDCG@10'] = None
        
        data.append(row)
    
    return pd.DataFrame(data)

def main():
    log_file = '/home/baohonghui/Reference/RMDR/log/v9_para_3.log'
    
    print("Parsing results.log file...")
    results = parse_results_log(log_file)
    
    print("\nParsing complete! Results overview:")
    for result_type, domains_data in results.items():
        print(f"\n{result_type} Results:")
        for domain, metrics in domains_data.items():
            print(f"  {domain}:")
            print(f"    Recall@5: {metrics['Recall@5']:.6f}")
            print(f"    Recall@10: {metrics['Recall@10']:.6f}")
            print(f"    NDCG@5: {metrics['NDCG@5']:.6f}")
            print(f"    NDCG@10: {metrics['NDCG@10']:.6f}")
    
    # Create long format DataFrame
    df_long = create_dataframe(results)
    print(f"\nLong format DataFrame shape: {df_long.shape}")
    print(df_long.head(10))
    
    # Create wide format DataFrame
    df_wide = create_wide_format_dataframe(results)
    print(f"\nWide format DataFrame shape: {df_wide.shape}")
    print(df_wide.head())
    
    # Save as CSV files
    df_long.to_csv('./results_long_format.csv', index=False)
    df_wide.to_csv('./results_wide_format.csv', index=False)
    
    print("\nFiles saved:")
    print("- results_long_format.csv (Long format, suitable for detailed analysis)")
    print("- results_wide_format.csv (Wide format, suitable for Google Sheets)")
    
    # Save as JSON file for further processing
    with open('./results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("- results.json (Raw data, JSON format)")
    
    return results, df_long, df_wide

if __name__ == "__main__":
    main()
