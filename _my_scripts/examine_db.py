#!/usr/bin/env python3
"""
Script to examine the QH9Stable.db database structure and sample data
"""

import sqlite3
import os

def examine_database(db_path):
    """Examine the database structure and get basic statistics"""
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return
    
    print(f"Examining database: {db_path}")
    print(f"File size: {os.path.getsize(db_path) / (1024**3):.2f} GB")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"\nFound {len(tables)} tables:")
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        print("Columns:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"Row count: {count:,}")
        
        # Show sample data (first 3 rows)
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
        sample_rows = cursor.fetchall()
        if sample_rows:
            print("Sample data:")
            for i, row in enumerate(sample_rows):
                print(f"  Row {i+1}: {row}")
    
    conn.close()

if __name__ == "__main__":
    db_path = "/root/25DFT/data/QH9_rawdata/QH9Stable.db"
    examine_database(db_path)
