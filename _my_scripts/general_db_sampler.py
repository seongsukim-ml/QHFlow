#!/usr/bin/env python3
"""
General SQLite Database Sampler
A flexible script to sample any SQLite database with customizable parameters
"""

import sqlite3
import os
import random
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

class DatabaseSampler:
    """General SQLite database sampler with flexible configuration"""
    
    def __init__(self, input_db_path: str, output_db_path: str, 
                 sample_ratio: float = 0.01, random_seed: int = 42,
                 batch_size: int = 1000, verbose: bool = True):
        """
        Initialize the database sampler
        
        Args:
            input_db_path: Path to the input SQLite database
            output_db_path: Path for the output sampled database
            sample_ratio: Fraction of data to sample (0.0 to 1.0)
            random_seed: Random seed for reproducibility
            batch_size: Batch size for processing
            verbose: Whether to print progress information
        """
        self.input_db_path = input_db_path
        self.output_db_path = output_db_path
        self.sample_ratio = sample_ratio
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Validate inputs
        if not os.path.exists(input_db_path):
            raise FileNotFoundError(f"Input database not found: {input_db_path}")
        
        if not (0.0 < sample_ratio <= 1.0):
            raise ValueError(f"Sample ratio must be between 0.0 and 1.0, got {sample_ratio}")
    
    def get_table_info(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Get information about all tables in the database"""
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        table_info = []
        for table_name, in tables:
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            
            table_info.append({
                'name': table_name,
                'columns': columns,
                'row_count': count
            })
        
        return table_info
    
    def get_primary_key_column(self, table_name: str, conn: sqlite3.Connection) -> Optional[str]:
        """Get the primary key column name for a table"""
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        for col in columns:
            if col[5]:  # pk column (6th element is 1 if primary key)
                return col[1]  # column name
        return None
    
    def get_unique_identifier_column(self, table_name: str, conn: sqlite3.Connection) -> str:
        """Get a suitable column for unique identification (primary key or first column)"""
        pk_col = self.get_primary_key_column(table_name, conn)
        if pk_col:
            return pk_col
        
        # If no primary key, use the first column
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        return columns[0][1] if columns else None
    
    def create_table_schema(self, table_name: str, conn: sqlite3.Connection, 
                           output_cursor: sqlite3.Cursor) -> None:
        """Create table schema in output database"""
        cursor = conn.cursor()
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        create_sql = cursor.fetchone()
        
        if create_sql:
            try:
                output_cursor.execute(create_sql[0])
            except sqlite3.OperationalError as e:
                if "already exists" in str(e):
                    # Table already exists, skip creation
                    pass
                else:
                    raise e
    
    def sample_table(self, table_name: str, input_conn: sqlite3.Connection, 
                    output_conn: sqlite3.Connection) -> int:
        """Sample a single table"""
        input_cursor = input_conn.cursor()
        output_cursor = output_conn.cursor()
        
        # Get table info
        table_info = self.get_table_info(input_conn)
        table_data = next((t for t in table_info if t['name'] == table_name), None)
        
        if not table_data:
            raise ValueError(f"Table '{table_name}' not found in database")
        
        total_rows = table_data['row_count']
        sample_size = int(total_rows * self.sample_ratio)
        
        if self.verbose:
            print(f"\nTable: {table_name}")
            print(f"  Total rows: {total_rows:,}")
            print(f"  Target sample size: {sample_size:,}")
        
        if sample_size == 0:
            if self.verbose:
                print(f"  Skipping table (sample size would be 0)")
            return 0
        
        # Get unique identifier column
        id_column = self.get_unique_identifier_column(table_name, input_conn)
        if not id_column:
            raise ValueError(f"No suitable identifier column found for table '{table_name}'")
        
        if self.verbose:
            print(f"  Using identifier column: {id_column}")
        
        # Get all IDs and sample
        input_cursor.execute(f"SELECT {id_column} FROM {table_name} ORDER BY {id_column}")
        all_ids = [row[0] for row in input_cursor.fetchall()]
        
        sampled_ids = random.sample(all_ids, sample_size)
        sampled_ids.sort()
        
        if self.verbose:
            print(f"  Sampled {len(sampled_ids)} unique IDs")
        
        # Copy sampled data
        copied_count = 0
        for i in range(0, len(sampled_ids), self.batch_size):
            batch_ids = sampled_ids[i:i+self.batch_size]
            placeholders = ','.join(['?' for _ in batch_ids])
            
            # Get data for this batch
            input_cursor.execute(f"SELECT * FROM {table_name} WHERE {id_column} IN ({placeholders})", batch_ids)
            batch_data = input_cursor.fetchall()
            
            # Insert into output database
            if batch_data:
                # Get column names for INSERT statement
                cursor = input_conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                placeholders = ','.join(['?' for _ in column_names])
                insert_sql = f"INSERT INTO {table_name} ({','.join(column_names)}) VALUES ({placeholders})"
                
                output_cursor.executemany(insert_sql, batch_data)
            
            copied_count += len(batch_data)
            if self.verbose and (copied_count % 1000 == 0 or copied_count == len(sampled_ids)):
                print(f"  Copied {copied_count}/{len(sampled_ids)} samples...")
        
        return copied_count
    
    def sample_database(self, table_names: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Sample the entire database or specific tables
        
        Args:
            table_names: List of table names to sample. If None, samples all tables.
        
        Returns:
            Dictionary mapping table names to sample counts
        """
        if self.verbose:
            print(f"Sampling {self.sample_ratio*100:.1f}% of data from {self.input_db_path}")
            print(f"Output will be saved to: {self.output_db_path}")
        
        # Connect to databases
        input_conn = sqlite3.connect(self.input_db_path)
        output_conn = sqlite3.connect(self.output_db_path)
        
        try:
            # Get table information
            table_info = self.get_table_info(input_conn)
            
            if table_names is None:
                table_names = [t['name'] for t in table_info]
            
            # Validate table names
            available_tables = [t['name'] for t in table_info]
            invalid_tables = [t for t in table_names if t not in available_tables]
            if invalid_tables:
                raise ValueError(f"Tables not found: {invalid_tables}")
            
            if self.verbose:
                print(f"Found {len(table_info)} tables in database")
                for table in table_info:
                    print(f"  - {table['name']}: {table['row_count']:,} rows")
            
            # Create output directory and remove existing output file
            os.makedirs(os.path.dirname(self.output_db_path), exist_ok=True)
            if os.path.exists(self.output_db_path):
                os.remove(self.output_db_path)
            
            # Sample each table
            results = {}
            for table_name in table_names:
                # Create table schema in output database
                self.create_table_schema(table_name, input_conn, output_conn.cursor())
                
                # Sample table data
                sample_count = self.sample_table(table_name, input_conn, output_conn)
                results[table_name] = sample_count
            
            # Commit changes
            output_conn.commit()
            
            if self.verbose:
                print(f"\nSampling completed!")
                print(f"Output file size: {os.path.getsize(self.output_db_path) / (1024**3):.2f} GB")
                print("\nResults:")
                for table_name, count in results.items():
                    print(f"  {table_name}: {count:,} samples")
            
            return results
            
        finally:
            input_conn.close()
            output_conn.close()

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Sample SQLite database with flexible options")
    parser.add_argument("input_db", help="Path to input SQLite database")
    parser.add_argument("output_db", help="Path to output sampled database")
    parser.add_argument("--ratio", "-r", type=float, default=0.01, 
                       help="Sample ratio (0.0 to 1.0, default: 0.01)")
    parser.add_argument("--tables", "-t", nargs="+", 
                       help="Specific tables to sample (default: all tables)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--batch-size", "-b", type=int, default=1000,
                       help="Batch size for processing (default: 1000)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    try:
        sampler = DatabaseSampler(
            input_db_path=args.input_db,
            output_db_path=args.output_db,
            sample_ratio=args.ratio,
            random_seed=args.seed,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )
        
        results = sampler.sample_database(table_names=args.tables)
        
        total_samples = sum(results.values())
        print(f"\n✅ Successfully created sampled database with {total_samples:,} total samples")
        print(f"📁 Output file: {args.output_db}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
