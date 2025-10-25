import os, json, lmdb, numpy as np
# import zlib, msgpac
import apsw
from apsw import Connection
# from tqdm.rich import tqdm
from tqdm import tqdm

from typing import Union
import concurrent.futures
from common.matrix_transforms import pack_upper_triangle, unpack_upper_triangle
from common.custom_logger import get_logger

logger = get_logger(__file__)


class LMDBShard_maker:
    def __init__(
        self,
        root_path: str,
        shard_num: int,
        save_path=None,
        processd_dir_name="processed",
        shard_dir_name="shards",
    ):
        self.root_path = root_path # .db files
        self.shard_num = shard_num
        self.processd_dir_name = processd_dir_name
        self.shard_dir_name = shard_dir_name
        if save_path is None:
            self.save_path = os.path.join(root_path, self.processd_dir_name)
        else:
            self.save_path = os.path.join(save_path, self.processd_dir_name)
        self.shard_dir_path = os.path.join(self.save_path, self.shard_dir_name)
        self.lmdb_path_list = self._lmdb_path_list()

    def _get_total_len(self):
        raise NotImplementedError("_get_total_len method is not implemented")
    
    def process_data(self, key_data_pair):
        # key, data = key_data_pair
        raise NotImplementedError("process method is not implemented")
   
    def write_shard_idx_single(self, idx):
        raise NotImplementedError("write_shard_idx_single method is not implemented")
    
    def write_shard_idx(self):
        raise NotImplementedError("write_shard_idx method is not implemented")
    
    def _process(self):
        raise NotImplementedError("_process method is not implemented")
    
    def make_new_shard(self, idx, max_workers=8, get_keys_list=True):
        raise NotImplementedError("make_new_shard method is not implemented")
    
    def process(self, idx = None):        
        assert isinstance(idx, Union[int, list, None]), f"Index {idx} is not an integer or list"
        logger.info(f"Loading the database from {self.root_path}")
        logger.info(f"Saving processed data to {self.save_path}")
        logger.info(f"Saving shards to {self.shard_dir_path}")
        logger.info(f"Number of shards: {self.shard_num}")

        self._process()

        if idx == -1 or idx is None:
            if idx == -1:
                logger.info(f"idx is -1, making all shards")
            else:
                logger.info(f"idx is None, making all shards")
            for idx in range(self.shard_num):
                logger.info(f"Making shard {idx} / {self.shard_num}")
                if os.path.exists(self.lmdb_path_list[idx]):
                    if self.overwrite:
                        logger.info(f"Shard {idx} already exists, but overwriting...")
                        self.make_new_shard(idx)
                    else:
                        logger.info(f"Shard {idx} already exists, skipping...")
                        continue
                else:
                    self.make_new_shard(idx)
                self._check_shard_completion(idx)
        else:
            if type(idx) == list:
                for id in idx:
                    assert type(id) == int, f"Index {id} is not an integer"
                    assert id >= 0 and id < self.shard_num, f"Index {id} is out of range"
                    logger.info(f"Making shard {id} / {self.shard_num}")
                    if os.path.exists(self.lmdb_path_list[id]):
                        if self.overwrite:
                            logger.info(f"Shard {id} already exists, but overwriting...")
                            self.make_new_shard(id)
                        else:
                            logger.info(f"Shard {id} already exists, skipping...")
                            continue
                    else:
                        self.make_new_shard(id)
                    self._check_shard_completion(id)
            else:
                assert type(idx) == int, f"Index {idx} is not an integer"
                assert idx >= 0 and idx < self.shard_num, f"Index {idx} is out of range"
                logger.info(f"Making new shard {idx} / {self.shard_num}")
                if os.path.exists(self.lmdb_path_list[idx]):
                    if self.overwrite:
                        logger.info(f"Shard {idx} already exists, but overwriting...")
                        self.make_new_shard(idx)
                    else:
                        logger.info(f"Shard {idx} already exists, skipping...")
                else:
                    self.make_new_shard(idx)
                # Check if this specific shard is completed
                self._check_shard_completion(idx)

        # Check if all shards are completed
        all_completed = self._check_all_shards_completion()
        
        if all_completed:
            # Write completion status file
            self._write_completion_status()

    @property
    def total_len(self):
        return self._get_total_len()
    
    def _lmdb_path_list(self):
        if not os.path.exists(os.path.join(self.shard_dir_path)):
            os.makedirs(os.path.join(self.shard_dir_path))
        return [os.path.join(self.shard_dir_path, f"shard_{i:03d}.lmdb") for i in range(self.shard_num)]
    
    def _check_shard_completion(self, idx):
        """Check if a specific shard is completed"""
        shard_path = self.lmdb_path_list[idx]
        if os.path.exists(shard_path):
            logger.info(f"Shard {idx} completed successfully")
            return True
        else:
            logger.warning(f"Shard {idx} not found or incomplete")
            return False

    def _check_all_shards_completion(self):
        """Check if all shards are completed"""
        completed_shards = []
        missing_shards = []
        
        for idx in range(self.shard_num):
            if self._check_shard_completion(idx):
                completed_shards.append(idx)
            else:
                missing_shards.append(idx)
        
        logger.info(f"Completed shards: {len(completed_shards)}/{self.shard_num}")
        if missing_shards:
            logger.warning(f"Missing shards: {missing_shards}")
        else:
            logger.info("All shards completed successfully!")
        
        return len(missing_shards) == 0

    def _write_completion_status(self):
        """Write completion status to a file"""
        status_file = os.path.join(self.save_path, "shard_completion_status.json")
        
        completion_info = {
            "total_shards": self.shard_num,
            "completed_shards": [],
            "missing_shards": [],
            "completion_time": None,
            "all_completed": False
        }
        
        # Check each shard
        for idx in range(self.shard_num):
            if self._check_shard_completion(idx):
                completion_info["completed_shards"].append(idx)
            else:
                completion_info["missing_shards"].append(idx)
        
        # Set completion status
        completion_info["all_completed"] = len(completion_info["missing_shards"]) == 0
        
        if completion_info["all_completed"]:
            import datetime
            completion_info["completion_time"] = datetime.datetime.now().isoformat()
            logger.info("All shards processing completed!")
        
        # Write status to JSON file
        with open(status_file, "w") as f:
            json.dump(completion_info, f, indent=2)
        
        logger.info(f"Completion status written to: {status_file}")
        
        # Also write a simple completion marker file if all completed
        if completion_info["all_completed"]:
            completion_marker = os.path.join(self.save_path, "ALL_SHARDS_COMPLETED.txt")
            with open(completion_marker, "w") as f:
                f.write(f"All {self.shard_num} shards completed at {completion_info['completion_time']}\n")
            logger.info(f"Completion marker file created: {completion_marker}")

    @staticmethod        
    def pack_upper_triangle(M: np.ndarray):
        return pack_upper_triangle(M)

    @ staticmethod
    def unpack_upper_triangle(packed: np.ndarray, n: int):
        return unpack_upper_triangle(packed, n)
    
    @staticmethod
    def test_pack_and_unpack(M: np.ndarray):
        packed, n = LMDBShard_maker.pack_upper_triangle(M)
        assert packed.shape == (n * (n + 1) // 2,), f"Packed shape {packed.shape} is not equal to {n * (n + 1) // 2}"
        assert LMDBShard_maker.unpack_upper_triangle(packed, n).shape == M.shape, f"Unpacked shape {LMDBShard_maker.unpack_upper_triangle(packed, n).shape} is not equal to {M.shape}"
        logger.info(f"Pack and unpack test passed max diff is {np.abs(LMDBShard_maker.unpack_upper_triangle(packed, n) - M).max()}")

class LMDBShard_maker_npz(LMDBShard_maker):
    # MD17 revised dataset
    def __init__(
        self,
        root_path: str,
        shard_num: int,
        save_path=None,
        max_workers=8,
        use_parallel=True,
        processd_dir_name="processed",
        shard_dir_name="lmdbs",
        overwrite=False,
        make_split_info=True,
    ):
        super().__init__(
            root_path=root_path,
            shard_num=shard_num,
            save_path=save_path,
            processd_dir_name=processd_dir_name,
            shard_dir_name=shard_dir_name,
        )
        self.max_workers = max_workers
        self.use_parallel = use_parallel
        self.overwrite = overwrite
        self.make_split_info = make_split_info

    def open_npz(self, npz_path: str):
        with np.load(npz_path) as data:
            return data
    
    def _get_total_len(self):
        if self.data is None:
            self.data = self.open_npz(self.root_path)
        return len(self.data)
        
    def _process(self, idx = None):
        self.data = self.open_npz(self.root_path)
        if self.use_parallel:
            logger.info(f"Using parallel processing: {self.use_parallel} with {self.max_workers} workers")
        else:
            logger.info(f"Using sequential processing (Not parallel)")
            
    def make_new_shard(self, idx, max_workers=8, get_keys_list=True):
        logger.info(f"Loading data from database for shard {idx}")
        data_idx = np.arange(self.start_idx_list[idx], self.end_idx_list[idx])
        
        atoms_list = self.data["z"]  # (N)
        atoms_list = atoms_list.reshape(1, -1).repeat(data_idx.shape[0], axis=0) # (B, N)
        pos_list = self.data["R"][data_idx] # (B, N, 3)
        energy_list = self.data["E"][data_idx] # (B, 1)
        force_list = self.data["F"][data_idx] # (B , N, 3)
        
        data_chunk = list(zip(atoms_list, pos_list, energy_list, force_list))
        data_chunk_idx = list(zip(data_chunk, data_idx))
        
        logger.info(f"Loaded {len(data_chunk)} data from database for shard {idx}")

        # Calculate optimal map_size based on shard size
        samples_per_shard = len(data_chunk)
        avg_size_per_sample = 5 * 1024 * 1024  # 5MB per sample (MD17 revised average)
        shard_size = samples_per_shard * avg_size_per_sample

        map_size = max(shard_size * 3, 1024**3)  # 3x safety margin, minimum 1GB
        logger.info(f"Making shard {idx} with map_size {map_size / (1024**3):.2f} GB")
        db_env = lmdb.open(self.lmdb_path_list[idx]+"_in_process", map_size=map_size)
        data_chunk_idx = list(zip(data_chunk, data_idx)) # use key for data index
        if self.use_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    tqdm(
                        executor.map(self.process_data, data_chunk_idx),
                        total=len(data_chunk),
                    )
                )
            with db_env.begin(write=True) as txn:
                for key, value in results:
                    txn.put(key, value)
        else:
            with db_env.begin(write=True) as txn:
                for data in tqdm(data_chunk_idx, desc="Processing data"):
                    key, value = self.process_data(data)
                    txn.put(key, value)
        db_env.close()
        os.rename(self.lmdb_path_list[idx]+"_in_process", self.lmdb_path_list[idx])
        logger.info(f'The size of the shard {idx} is {os.path.getsize(self.lmdb_path_list[idx]) / (1024**3):.2f} GB')
        self.wirte_shard_idx_single(idx)

    def process_data(self, key_data_pair):
        # key, data = key_data_pair
        raise NotImplementedError("process method is not implemented")
    
    def wirte_shard_idx_single(self, idx):
        index = []
        data_idx = np.arange(self.start_idx_list[idx], self.end_idx_list[idx])
        shard_data_idx = 0
        for key in data_idx:
            index.append((idx, key.item(), shard_data_idx))
            shard_data_idx += 1
        with open(os.path.join(self.lmdb_path_list[idx],"single_index.json"), "w") as f:
            json.dump({"index": index}, f)
    
    def _write_shard_idx(self):
        """
        Write the shard index to a json file
        This file is used to load the shard index for the dataset.
        The index is a list of tuples, each containing the shard index, the start index, and the end index.
        """

        if not os.path.exists(os.path.join(self.save_path, "index.json")):
            index = []
            shard_idx = 0
            shard_data_idx = 0
            for key in range(self.total_len):
                if key >= self.end_idx_list[shard_idx]:
                    shard_idx += 1
                    shard_data_idx = 0
                index.append((shard_idx, key, shard_data_idx))
                shard_data_idx += 1
            assert len(index) == self.total_len
            assert index[-1][1] == self.total_len - 1
            assert index[-1][0] == self.shard_num - 1

            with open(os.path.join(self.save_path, "index.json"), "w") as f:
                json.dump({"index": index}, f)

class LMDBShard_maker_db(LMDBShard_maker):
    def __init__(
        self, 
        root_path: str, 
        shard_num: int, 
        save_path=None, 
        max_workers=8, 
        use_parallel=True,
        processd_dir_name="processed",
        shard_dir_name="lmdbs",
        overwrite=False,
        make_split_info=True,
        table_name="data",
    ):
        super().__init__(
            root_path=root_path,
            shard_num=shard_num,
            save_path=save_path,
            processd_dir_name=processd_dir_name,
            shard_dir_name=shard_dir_name,
        )
        # parallel processing
        self.use_parallel = use_parallel
        self.max_workers = max_workers
    
        # overwrite the shard if it already exists
        self.overwrite = overwrite
        self.table_name = table_name

        self.db = None
        self.cursor = None
        # self.all_data = None
        self.make_split_info = make_split_info
        
    def _process(self):
        self._get_db_env(read_only=True)
        self._db_info()
        self._set_shards_info()
        self._write_shard_idx()
        
        self.cursor = self.db.cursor()
        # logger.info(f"Loading all data from database")
        # all_data = self.cursor.execute("select * from data").fetchall()
        # self.all_data = all_data
        # logger.info(f"Loaded {len(all_data)} data from database")
        # self._close_db_env()

        if self.make_split_info:
            self._make_split_info() # Making Train, Val, Test indices

        if self.use_parallel:
            logger.info(f"Using parallel processing: {self.use_parallel} with {self.max_workers} workers")
        else:
            logger.info(f"Using sequential processing (Not parallel)")            

    def _get_db_env(self, read_only=True):
        if self.db is None:
            if not os.path.exists(self.root_path):
                raise FileNotFoundError(f"Database file not found: {self.root_path}")
            logger.info(f"Loading the database from {self.root_path}")
            logger.info(f"File size: {os.path.getsize(self.root_path) / (1024**3):.2f} GB")
            if read_only:
                self.db = Connection(self.root_path, flags=apsw.SQLITE_OPEN_READONLY)
            else:
                self.db = Connection(self.root_path)
        return self.db
 
    def _close_db_env(self):
        if self.db is not None:
            self.db.close()
            self.db = None
            self.cursor = None
            
    def _db_info(self, save_info=True):
        db = self._get_db_env() if self.db is None else self.db
        cursor = db.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        
        # Prepare info string for both printing and saving
        info_lines = []
        
        for table in tables:
            table_name = table[0]
            columns = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
            
            table_info = f"Table: {table_name}"
            logger.info(table_info)
            info_lines.append(table_info)
            
            for column in columns:
                col_id, col_name, col_type, not_null, default_val, pk = column
                column_info = f"  Column {col_id}: {col_name} ({col_type}) {'NOT NULL' if not_null else 'NULL'} {'PRIMARY KEY' if pk else ''}"
                logger.info(column_info)
                info_lines.append(column_info)
            
            row_count = cursor.execute(f'SELECT COUNT(*) FROM {table_name};').fetchone()[0]
            row_info = f"  Total rows: {row_count}"
            logger.info(row_info)
            info_lines.append(row_info)
            info_lines.append("")  # Add empty line between tables
        
        # Save to txt file
        if save_info:
            db_info_path = os.path.join(self.save_path, "db_info.txt")
            if not os.path.exists(db_info_path):
                with open(db_info_path, "w") as f:
                    f.write("\n".join(info_lines))        
                logger.info(f"Database info saved to {db_info_path}")
            else:
                logger.info(f"Database info already exists at {db_info_path}")

    def _get_total_len(self):
        if self.db is None:
            self._get_db_env()
        cursor = self.db.cursor()
        # Get total number of rows from the first table
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        if not tables:
            raise ValueError("No tables found in database")
        table_name = tables[0][0]
        total_len = cursor.execute(f'SELECT COUNT(*) FROM {table_name};').fetchone()[0]
        return total_len

    def _set_shards_info(self):
        self.shard_size = (self.total_len + self.shard_num - 1) // self.shard_num
        self.last_shard_size = self.total_len - (self.shard_num - 1) * self.shard_size
        self.shard_size_list = [self.shard_size] * (self.shard_num - 1) + [self.last_shard_size]
        assert sum(self.shard_size_list) == self.total_len
        self.start_idx_list = np.cumsum([0] + self.shard_size_list)[:-1]
        self.end_idx_list = np.cumsum(self.shard_size_list)
        assert self.end_idx_list[-1] == self.total_len

    def _write_shard_idx(self):
        """
        Write the shard index to a json file
        This file is used to load the shard index for the dataset.
        The index is a list of tuples, each containing the shard index, the start index, and the end index.
        """

        if not os.path.exists(os.path.join(self.save_path, "index.json")):
            index = []
            shard_idx = 0
            shard_data_idx = 0
            for key in range(self.total_len):
                if key >= self.end_idx_list[shard_idx]:
                    shard_idx += 1
                    shard_data_idx = 0
                index.append((shard_idx, key, shard_data_idx))
                shard_data_idx += 1
            assert len(index) == self.total_len
            assert index[-1][1] == self.total_len - 1
            assert index[-1][0] == self.shard_num - 1

            with open(os.path.join(self.save_path, "index.json"), "w") as f:
                json.dump({"index": index}, f)
    
    def _merge_shard_idx(self):
        if not os.path.exists(os.path.join(self.save_path, "index.json")):
            merge_index = []
            for idx in range(self.shard_num):
                with open(os.path.join(self.lmdb_path_list[idx], "single_index.json"), "r") as f:
                    single_index = json.load(f)
                single_index = single_index["index"]
                merge_index.extend(single_index)
            assert len(merge_index) == self.total_len
            assert merge_index[-1][1] == self.total_len - 1
            assert merge_index[-1][0] == self.shard_num - 1
            with open(os.path.join(self.save_path, "index.json"), "w") as f:
                json.dump({"index": merge_index}, f)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_db_env()
    
    def make_new_shard(self, idx, max_workers=8, get_keys_list=True):
        # assert self.all_data is not None, "all_data is not set"
        # data_chunk = self.all_data[self.start_idx_list[idx]:self.end_idx_list[idx]]

        # Loading the batch data from database (Loading all data from database is too slow)
        logger.info(f"Loading data from database for shard {idx}")
        data_chunk = self.cursor.execute(
            f"SELECT * FROM {self.table_name} LIMIT {self.end_idx_list[idx] - self.start_idx_list[idx]} OFFSET {self.start_idx_list[idx]}"
        ).fetchall()

        if get_keys_list:
            try:
                keys_list = self.cursor.execute(
                    f"SELECT id FROM data LIMIT {self.end_idx_list[idx] - self.start_idx_list[idx]} OFFSET {self.start_idx_list[idx]}"
                ).fetchall()
                os.makedirs(self.lmdb_path_list[idx]+"_in_process", exist_ok=True)
                with open(os.path.join(self.lmdb_path_list[idx]+"_in_process", "keys_list.json"), "w") as f:
                    json.dump({"keys_list": keys_list}, f)
            except Exception as e:
                logger.warning(f"Error getting keys list for shard {idx}: {e}")

        logger.info(f"Loaded {len(data_chunk)} data from database for shard {idx}")
        data_idx = np.arange(self.start_idx_list[idx], self.end_idx_list[idx])

        # Calculate optimal map_size based on shard size
        samples_per_shard = len(data_chunk)
        avg_size_per_sample = 30 * 1024 * 1024  # 30MB per sample (QH9 average)
        shard_size = samples_per_shard * avg_size_per_sample

        map_size = max(shard_size * 3, 1024**3)  # 3x safety margin, minimum 1GB
        logger.info(f"Making shard {idx} with map_size {map_size / (1024**3):.2f} GB")
        db_env = lmdb.open(self.lmdb_path_list[idx]+"_in_process", map_size=map_size)
        data_chunk_idx = list(zip(data_chunk, data_idx)) # use key for data index
        if self.use_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    tqdm(
                        executor.map(self.process_data, data_chunk_idx),
                        total=len(data_chunk),
                    )
                )
            with db_env.begin(write=True) as txn:
                for key, value in results:
                    txn.put(key, value)
        else:
            with db_env.begin(write=True) as txn:
                for data in tqdm(data_chunk_idx, desc="Processing data"):
                    key, value = self.process_data(data)
                    txn.put(key, value)
        db_env.close()
        os.rename(self.lmdb_path_list[idx]+"_in_process", self.lmdb_path_list[idx])
        logger.info(f'The size of the shard {idx} is {os.path.getsize(os.path.join(self.lmdb_path_list[idx],"data.mdb")) / (1024**3):.2f} GB')
        self.wirte_shard_idx_single(idx)
    
    def wirte_shard_idx_single(self, idx):
        index = []
        data_idx = np.arange(self.start_idx_list[idx], self.end_idx_list[idx])
        shard_data_idx = 0
        for key in data_idx:
            index.append((idx, key.item(), shard_data_idx))
            shard_data_idx += 1
        with open(os.path.join(self.lmdb_path_list[idx],"single_index.json"), "w") as f:
            json.dump({"index": index}, f)