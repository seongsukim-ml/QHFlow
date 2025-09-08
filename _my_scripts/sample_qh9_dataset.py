#!/usr/bin/env python3
"""
Script to sample 1/100 of the QH9Stable.db dataset
"""

import sqlite3
import os
import random
import numpy as np
from pathlib import Path

def sample_qh9_dataset(input_db_path, output_db_path, sample_ratio=0.01):
    """
    Sample a subset of the QH9Stable.db dataset
    
    Args:
        input_db_path: Path to the original QH9Stable.db file
        output_db_path: Path for the output sampled database
        sample_ratio: Fraction of data to sample (default: 0.01 = 1/100)
    """
    
    if not os.path.exists(input_db_path):
        raise FileNotFoundError(f"Input database not found: {input_db_path}")
    
    print(f"Sampling {sample_ratio*100:.1f}% of data from {input_db_path}")
    print(f"Output will be saved to: {output_db_path}")
    
    # Connect to input database
    input_conn = sqlite3.connect(input_db_path)
    input_cursor = input_conn.cursor()
    
    # Get total number of rows
    input_cursor.execute("SELECT COUNT(*) FROM data")
    total_rows = input_cursor.fetchone()[0]
    sample_size = int(total_rows * sample_ratio)
    
    print(f"Total rows in original dataset: {total_rows:,}")
    print(f"Target sample size: {sample_size:,}")
    
    # Create output database
    output_conn = sqlite3.connect(output_db_path)
    output_cursor = output_conn.cursor()
    
    # Create the same table structure
    output_cursor.execute("""
        CREATE TABLE data (
            id INTEGER,
            N INTEGER,
            Z BLOB,
            pos BLOB,
            Ham BLOB
        )
    """)
    
    # Get all row IDs and randomly sample
    input_cursor.execute("SELECT id FROM data ORDER BY id")
    all_ids = [row[0] for row in input_cursor.fetchall()]
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Sample random indices
    sampled_ids = random.sample(all_ids, sample_size)
    sampled_ids.sort()  # Sort for consistent ordering
    
    print(f"Sampled {len(sampled_ids)} unique IDs")
    
    # Copy sampled data
    batch_size = 1000
    copied_count = 0
    
    for i in range(0, len(sampled_ids), batch_size):
        batch_ids = sampled_ids[i:i+batch_size]
        placeholders = ','.join(['?' for _ in batch_ids])
        
        # Get data for this batch
        input_cursor.execute(f"SELECT * FROM data WHERE id IN ({placeholders})", batch_ids)
        batch_data = input_cursor.fetchall()
        
        # Insert into output database
        output_cursor.executemany("INSERT INTO data VALUES (?, ?, ?, ?, ?)", batch_data)
        
        copied_count += len(batch_data)
        if copied_count % 1000 == 0 or copied_count == len(sampled_ids):
            print(f"Copied {copied_count}/{len(sampled_ids)} samples...")
    
    # Commit and close connections
    output_conn.commit()
    input_conn.close()
    output_conn.close()
    
    # Verify the output
    verify_conn = sqlite3.connect(output_db_path)
    verify_cursor = verify_conn.cursor()
    verify_cursor.execute("SELECT COUNT(*) FROM data")
    final_count = verify_cursor.fetchone()[0]
    verify_conn.close()
    
    print(f"\nSampling completed!")
    print(f"Final sample size: {final_count:,}")
    print(f"Output file size: {os.path.getsize(output_db_path) / (1024**3):.2f} GB")
    
    return final_count

def main():
    # Set paths
    input_db = "/root/25DFT/data/QH9_rawdata/QH9Stable.db"
    output_db = "/root/25DFT/data/QH9_rawdata/QH9Stable_sample_1_100.db"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_db), exist_ok=True)
    
    try:
        sample_count = sample_qh9_dataset(input_db, output_db, sample_ratio=0.01)
        print(f"\n✅ Successfully created sampled dataset with {sample_count:,} samples")
        print(f"📁 Output file: {output_db}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
