import datetime
import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import Optional, Dict, List, Tuple, Any

import pyBigWig as bw
import pickle

import pandas as pd
import os
import numpy as np
import hashlib

import logging
from pysam import FastaFile
import re
import h5py
import tqdm
import sys
import json
from transformers import AutoTokenizer
from multiprocessing import Pool
from downstream_tasks.expression_prediction.datasets.src.utils import convert_fm_relative_path_to_absolute_path
from pathlib import Path



class ExpressionDataset(Dataset):
    def __init__(
        self,
        gen_tokenizer,
        targets_path: str,
        genome: str,
        forward_intervals_path: str = None,
        reverse_intervals_path: str = None,
        loglevel: int = logging.WARNING,
        seed: int = 42,
        num_before: int = 512,
        gen_max_seq_len: int = 1024,
        transform_targets_bw=None,
        transform_targets_tpm=None,
        bw : str = "",
        tpm : str = "",
        hash_prefix = None,
        n_keys: Optional[int] = None,
        token_len_for_fetch: int = 10,
        norm_bw = False,
        tf_tokenizer: str = None,
        panel_genes_path: str = None,
        bin_edges_path: str = None,
    ):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=loglevel)

        assert sys.version_info >= (3, 8), "Python 3.8+ required"

        if isinstance(gen_tokenizer, str):
            self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_tokenizer, trust_remote_code=True)
        else:
            self.gen_tokenizer = gen_tokenizer

        self.token_len_for_fetch = token_len_for_fetch

        self.gen_max_seq_len = gen_max_seq_len
        self.genome = genome
        self._genome_sizes_map = {}
        self._genome_sizes_base_dir = None
        genome_sizes_path = Path(self.genome).expanduser().parent.parent / "genome_sizes.tsv"
        if genome_sizes_path.exists():
            self._genome_sizes_base_dir = genome_sizes_path.parent
            df_sizes = pd.read_csv(genome_sizes_path, sep="\t")
            for p, size in zip(df_sizes["path"], df_sizes["size"]):
                if pd.isna(p) or pd.isna(size):
                    continue
                genome_path = Path(str(p)).expanduser()
                if not genome_path.is_absolute():
                    genome_path = self._genome_sizes_base_dir / genome_path
                self._genome_sizes_map[str(genome_path.resolve())] = size

        self.seed = seed
        np.random.seed(self.seed)
        
        self.bw = bw
        self.tpm = tpm
        self.norm_bw = norm_bw 
        
        self.targets_path = targets_path

        self.num_before = num_before
        self.transform_targets_bw = transform_targets_bw
        self.transform_targets_tpm = transform_targets_tpm

        assert forward_intervals_path is not None or reverse_intervals_path is not None, "Either forward_intervals_path or reverse_intervals_path must be provided"
        self.intervals_hash = self._name_and_size(forward_intervals_path) + "|" + self._name_and_size(reverse_intervals_path)
        if hash_prefix is None:
            self.hash_prefix = os.path.dirname(forward_intervals_path) if forward_intervals_path is not None else os.path.dirname(reverse_intervals_path)
            self.hash_prefix = os.path.join(self.hash_prefix, "dataset_hash")
        else:
            self.hash_prefix = hash_prefix

        self.read_paths()
        self._bw_key_to_col = {k: i for i, k in enumerate(self.paths.keys())}
        
        forward_genes = pd.read_csv(forward_intervals_path, sep=None, engine="python") if forward_intervals_path is not None else pd.DataFrame()
        forward_genes["strand"] = "+"
        reverse_genes = pd.read_csv(reverse_intervals_path, sep=None, engine="python") if reverse_intervals_path is not None else pd.DataFrame()
        reverse_genes["strand"] = "-"
        self.genes = pd.concat([forward_genes, reverse_genes], ignore_index=True)

        if n_keys is None:
            n_keys = len(self.track_keys)
        self.n_keys = n_keys

        self.all_keys = list(self.paths.keys())
        self.epoch = 0

        self.files_opened = False
        self.sequences = None
        self.h5_cache_path = self.get_hash_path() + ".h5"

        if os.path.exists(self.h5_cache_path):
            self.h5_cache = h5py.File(self.h5_cache_path, "r")
        else:
            self._ensure_sequences_open("token cache precomputation")
            self.precompute_tokenization()


        if self.bw:
            self.signals_cache_path = self.get_signals_hash_path() + ".h5"
            if os.path.exists(self.signals_cache_path):
                self.signals_cache = h5py.File(self.signals_cache_path, "r")
            else:
                self.precompute_signals()
                   
        if self.tpm:
            assert all(self.paths[k][1] is not None for k in self.paths), "TPM paths are not set for some of the keys"
            tpm_hash_path = self.get_tpm_hash_path()
            if os.path.exists(tpm_hash_path):
                self.logger.debug(f"Loading tpm cache from {tpm_hash_path}")
                self.tpm_lookup = pickle.load(open(tpm_hash_path, "rb"))
                assert len(self.tpm_lookup) == len(self.paths), "Number of tpm cache and paths are not the same"
                assert all(key in self.tpm_lookup for key in self.paths.keys()), "All keys in paths must be in tpm cache"
                # Empty/NaN/inf values are allowed: they are masked out of the targets and
                # become [UNK] in the tf panel.
                n_bad = sum(int((~np.isfinite(v.to_numpy(dtype=np.float32))).sum()) for v in self.tpm_lookup.values())
                if n_bad > 0:
                    self.logger.warning(f"TPM cache contains {n_bad} empty/NaN/inf values, they are treated as unknown")
            else:
                self.tpm_cache = {}
                for key, (bw_paths, tpm_path) in tqdm.tqdm(self.paths.items()):
                    self.logger.debug(f"Reading tpm from {tpm_path}")
                    tpm_df = pd.read_csv(tpm_path, dtype=np.float32)
                    self.tpm_cache[key] = tpm_df
                self.tpm_lookup = {}
                for key, tpm_df in self.tpm_cache.items():
                    self.tpm_lookup[key] = tpm_df.T.set_index(tpm_df.columns)
                pickle.dump(self.tpm_lookup, open(tpm_hash_path, "wb"))

        if not self.tpm:
            raise ValueError("The tf panel is built from the tpm matrix, so tpm must be set")
        if tf_tokenizer is None or panel_genes_path is None or bin_edges_path is None:
            raise ValueError("tf_tokenizer, panel_genes_path and bin_edges_path must be provided")
        for key in self.all_keys:
            if not self.tpm_lookup[key].index.is_unique:
                raise ValueError(f"Duplicated gene ids in the tpm profile of {key}")

        # gene order and bins are the ones the panel model was trained with (make_panel_files.py)
        self.tf_genes = self.load_panel_genes(panel_genes_path)
        self.tf_gene_pos = {gene_id: i + 1 for i, gene_id in enumerate(self.tf_genes)}  # +1 for [BOS]
        self.tf_tokenizer = tf_tokenizer
        self._init_tf_tokens(tf_tokenizer)
        self.tf_bin_edges = self.load_tf_bin_edges(bin_edges_path)
        self.control_panels = self.build_control_panels()
        self.build_queue()

    def _name_and_size(self, p: str) -> str:
        if p is None:
            return "None|0"
        path = str(p)
        name = Path(path).name
        try:
            size = os.path.getsize(path)  # bytes
        except OSError:
            if path == self.genome and self._genome_sizes_base_dir is not None:
                genome_path = Path(path).expanduser()
                if not genome_path.is_absolute():
                    genome_path = self._genome_sizes_base_dir / genome_path
                genome_size = self._genome_sizes_map.get(str(genome_path.resolve()))
                if genome_size is not None:
                    return f"{name}|{genome_size}"
            raise FileNotFoundError(f"File not found for hashing: {path}")
        return f"{name}|{size}"

    def _ensure_sequences_open(self, required_for: str = "sequence access"):
        if self.sequences is not None:
            return
        if not self.genome or not os.path.exists(self.genome):
            raise FileNotFoundError(
                f"Genome fasta is required for {required_for}, but was not found: {self.genome}"
            )
        self.sequences = FastaFile(self.genome)
        
    def get_hash_path(self):
        m = hashlib.blake2b(digest_size=8)
        input_strings = []
        
        input_str = str('tokens')
        m.update(input_str.encode("utf-8"))
        input_strings.append(input_str)
        
        input_str = str(self.intervals_hash)
        m.update(input_str.encode("utf-8"))
        input_strings.append(input_str)
        
        input_str = self._name_and_size(self.genome)
        m.update(input_str.encode("utf-8"))
        input_strings.append(input_str)
        
        input_str = str(self.num_before)
        m.update(input_str.encode("utf-8"))
        input_strings.append(input_str)
        
        if self.token_len_for_fetch != 10: # 8 was default in first version of the dataset; TODO: remove at some point
            input_str = str(self.token_len_for_fetch)
            m.update(input_str.encode("utf-8"))
            input_strings.append(input_str)
            
        self.logger.debug(f"Hash inputs: {input_strings}")
        self.logger.debug(f"constructed hash suffix: {m.hexdigest()}")
        hash_suffix = m.hexdigest()
        hash_path = str(self.hash_prefix) + "." + hash_suffix
        return hash_path

    def get_signals_hash_path(self):
        m = hashlib.blake2b(digest_size=8)
        m.update(str('signals').encode("utf-8"))
        m.update(str(self.intervals_hash).encode("utf-8"))
        m.update(self._name_and_size(self.targets_path).encode("utf-8"))
        m.update(self._name_and_size(self.genome).encode("utf-8"))
        m.update(str(self.num_before).encode("utf-8"))
        m.update(str(self.gen_max_seq_len).encode("utf-8"))
        if self.norm_bw:
            m.update(str("norm_bw").encode("utf-8"))
        target_ids = "".join(sorted(list(self.paths.keys())))
        m.update(str(target_ids).encode("utf-8"))
        hash_suffix = m.hexdigest()
        return str(self.hash_prefix) + ".signal." + hash_suffix
    
    def get_tpm_hash_path(self):
        signals_hash_path = self.get_signals_hash_path()
        return self.hash_prefix + ".tpm." + signals_hash_path[len(self.hash_prefix) + len(".signal."):]
    
    def get_desc_hash_path(self, targets_path, text_tokenizer, text_max_seq_len):
        tokenizer_tag = text_tokenizer.replace("/", "_")
        descriptions_dir = Path(__file__).resolve().parent / "descriptions"
        descriptions_dir.mkdir(parents=True, exist_ok=True)
        targets_tag = hashlib.blake2b(
            self._name_and_size(targets_path).encode("utf-8"),
            digest_size=8,
        ).hexdigest()
        desc_cache_name = (
            f"{Path(targets_path).name}.{targets_tag}.{tokenizer_tag}.{text_max_seq_len}.description.h5"
        )
        return str(descriptions_dir / desc_cache_name)

    def get_num_keys(self):
        return len(self.paths.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _stable_gene_seed(self, gene_id: str) -> int:
        h = hashlib.blake2b(
            f"{self.seed}|{self.epoch}|{gene_id}".encode("utf-8"),
            digest_size=8
        )
        return int.from_bytes(h.digest(), "little") % (2**32)

    def _get_gene_key_order(self, gene_id: str, keys: List[str]) -> List[str]:
        rng = np.random.default_rng(self._stable_gene_seed(gene_id))
        perm = rng.permutation(len(keys))
        return [keys[i] for i in perm]

    def _get_selected_keys_for_gene(self, gene_id: str, keys: List[str], chunk_idx: int) -> List[str]:
        ordered_keys = self._get_gene_key_order(gene_id, keys)
        start_idx = chunk_idx * self.n_keys
        end_idx = min(start_idx + self.n_keys, len(ordered_keys))
        return ordered_keys[start_idx:end_idx]

        
    def read_paths(self):
        self.paths: Dict[str, List[Any]] = {}
        self.logger.info(f"Reading paths from {self.targets_path}")
        df = pd.read_csv(self.targets_path)

        assert not df["id"].duplicated().any(), "Found duplicated id in targets_path"
        for column in ("dataset", "perturbation", "dataset_description"):
            if column not in df.columns:
                raise ValueError(f"Column '{column}' is missing in {self.targets_path}")
        self.dataset_description = ", ".join(sorted(df["dataset_description"].unique()))
        self.key_dataset = dict(zip(df["id"], df["dataset"]))
        self.key_description = dict(zip(df["id"], df["dataset_description"]))
        self.key_perturbation = dict(zip(df["id"], df["perturbation"]))

        # the non-targeting profile of a dataset is the panel of its tracks; it is never a track itself
        is_control = df["perturbation"].str.lower() == "non-targeting"
        self.control_keys = {}
        for key, dataset in zip(df.loc[is_control, "id"], df.loc[is_control, "dataset"]):
            if dataset in self.control_keys:
                raise ValueError(f"Dataset {dataset} has more than one non-targeting profile in {self.targets_path}")
            self.control_keys[dataset] = key
        missing = set(df["dataset"]) - set(self.control_keys)
        if missing:
            raise ValueError(f"No non-targeting profile for datasets {sorted(missing)} in {self.targets_path}")
        self.track_keys = list(df.loc[~is_control, "id"])

        if self.bw:
            forward_colname = "forward_"+self.bw
            reverse_colname = "reverse_"+self.bw
            assert not pd.isna(df[forward_colname]).any(), f"{forward_colname} is NaN for some targets"
            assert not pd.isna(df[reverse_colname]).any(), f"{reverse_colname} is NaN for some targets"
            forward_paths = df[forward_colname].apply(lambda x: convert_fm_relative_path_to_absolute_path(x, self.targets_path)).values
            reverse_paths = df[reverse_colname].apply(lambda x: convert_fm_relative_path_to_absolute_path(x, self.targets_path)).values
            self.paths = {k:[{"+": forward_paths[ind], "-": reverse_paths[ind]}] for ind,k in enumerate(df["id"])}
            
            if self.norm_bw:
                self.logger.info("Reading metadata for normalization of bigwig tracks")
                meta_list = []
                dir_path = os.path.dirname(self.targets_path)
                for _, row in df.iterrows():
                    json_path = os.path.abspath(os.path.join(dir_path, row['metadata']))
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                        meta_list.append(metadata)
                meta_df = pd.DataFrame(meta_list, index=df['id'])
                self.coverage_norm = {
                    k: {
                        "+": meta_df.loc[k]["forward_total_coverage"],
                        "-": meta_df.loc[k]["reverse_total_coverage"]
                    }
                    for k in df['id']
                }
        else:
            self.paths = {k:[{"+": None, "-": None}] for ind,k in enumerate(df["id"])}

        if self.tpm:
            tpm_colname = self.tpm
            assert not pd.isna(df[tpm_colname]).any(), f"{tpm_colname} is NaN for some targets"
            tpm_paths = df[tpm_colname].apply(lambda x: convert_fm_relative_path_to_absolute_path(x, self.targets_path)).values
            for ind,k in enumerate(df["id"]):
                self.paths[k].append(tpm_paths[ind])
        else:
            for ind,k in enumerate(df["id"]):
                self.paths[k].append(None)

        self.files_opened = False

    def precompute_tokenization(self):
        self.logger.info(f"Precomputing tokenization to {self.h5_cache_path}")
        temp_path = f"{self.h5_cache_path}.{os.getpid()}.temp"
        
        try:
            with h5py.File(temp_path, "w") as h5f:
                pbar = tqdm.tqdm(total=len(self.genes), desc="Tokenizing sequences")
                for idx in range(len(self.genes)):
                    gene_id = self.genes.iloc[idx]['gene_id']
                    _, tokens_df = self.tokenize_genome(idx)
                    
                    gene_group = h5f.create_group(gene_id)
                    gene_group.create_dataset('input_ids', data=tokens_df["token_id"].values.astype(np.int32))
                    gene_group.create_dataset('starts', data=tokens_df["start"].values.astype(np.int64))
                    gene_group.create_dataset('ends', data=tokens_df["end"].values.astype(np.int64))
                    gene_group.attrs['strand'] = self.genes.iloc[idx]['strand']
                    gene_group.attrs['chrom'] = tokens_df["chrom"].iloc[0]
                    
                    if idx % 100 == 0:  
                        h5f.flush()
                    
                    pbar.update(1)
                    
                pbar.close()
                h5f.flush()
            
            os.rename(temp_path, self.h5_cache_path)
            self.h5_cache = h5py.File(self.h5_cache_path, "r")
            
        except Exception as e:
            self.logger.error(f"Error creating cache: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    def check_bw_genome_consistency(self, bw_handler, sequences):
        fasta_ref_lengths = {k:v for k, v in zip(sequences.references, sequences.lengths)}
        bw_ref_lengths = {k:v for k, v in bw_handler.chroms().items()}
        common_refs = set(fasta_ref_lengths.keys()) & set(bw_ref_lengths.keys())
        assert len(common_refs) > 0, f"No common references found in genome and bigwig file. Genome: {list(fasta_ref_lengths.keys())}, Bigwig: {list(bw_ref_lengths.keys())}. Genome mismatch?"
        for ref in common_refs:
            if fasta_ref_lengths[ref] != bw_ref_lengths[ref]:
                raise ValueError(f"Length of {ref} in genome and bigwig file are different. Ref: {ref}, Genome: {fasta_ref_lengths[ref]}, Bigwig: {bw_ref_lengths[ref]}")

    def open_files(self):
        """Open the bigWigs and build a fast key->col mapping."""
        if not self.bw or self.files_opened:
            return
        if getattr(self, "signals_cache", None) is not None:
            self.files_opened = True
            return
        self._ensure_sequences_open("bigWig/genome consistency checks")
        self.bigWigHandlers: Dict[str, Dict[str, Any]] = {}
        for k, (v1, v2) in self.paths.items():
            try:
                self.bigWigHandlers[k] = {strand: bw.open(path) for strand, path in v1.items()}
                for bw_handler in self.bigWigHandlers[k].values():
                    self.check_bw_genome_consistency(bw_handler, self.sequences)
            except Exception:
                self.logger.exception(f"Error opening bigwig file for key={k}, v1={v1}")

        self.files_opened = True

    def reverse_complement(self, sequence):
        complement = str.maketrans('ACGTN', 'TGCAN')
        return sequence.translate(complement)[::-1]

    def tokenize_genome(self, i):
        self._ensure_sequences_open("genome tokenization")
        row = self.genes.iloc[i]  
        chrom = row["chromosome"] 
        start = row["TSS"]
        end = row["TES"] 
        strand = row["strand"]
        reverse = 0 if strand == "+" else 1
        token_lengths = []
        
        if self.num_before > 0: 
            if (reverse == 0): # forward strand
                try:
                    sequence = self.sequences.fetch(chrom, max(start - self.num_before * self.token_len_for_fetch, 0), start).upper()
                except ValueError as e:
                    self.logger.error(f"Error sequence {i}")
            else: # reverse strand
                chrom_length = self.sequences.get_reference_length(chrom)
                try:
                    sequence = self.sequences.fetch(chrom, start, min(start + self.num_before * self.token_len_for_fetch, chrom_length)).upper()
                    sequence = self.reverse_complement(sequence)
                except ValueError as e:
                    self.logger.error(f"Error sequence {i}")
                
            encoded_sequence = self.gen_tokenizer.encode_plus(sequence, return_offsets_mapping=True)
            encoded_sequence['input_ids'] = encoded_sequence['input_ids'][1:-1]
            encoded_sequence['offset_mapping'] = encoded_sequence['offset_mapping'][1:-1]
            if len(encoded_sequence['input_ids']) < self.num_before:
                self.logger.warning(f"Trying to tokenize seq before TSS, but it's too short: {len(encoded_sequence['input_ids'])} < {self.num_before}; {chrom}: {start}-{end} ({strand})")
            tokens_before = encoded_sequence['input_ids'][-self.num_before:]
            mapping = encoded_sequence['offset_mapping'][-self.num_before:]
            
            for i, (start_i, end_i) in enumerate(mapping):
                token_id = tokens_before[i]
                if (token_id == 5):
                    if i > 0:
                        length = end_i - mapping[i-1][1] 
                    else:
                        length = end_i
                else:
                    length = end_i - start_i  
                # Only token ids and genomic offsets are persisted in the HDF5
                # cache. Decoding every token here is unused and makes the
                # first cache build needlessly expensive.
                token_lengths.append((token_id, length))
    
        if reverse == 0:
            start_gene = start - sum(t[1] for t in token_lengths)
        else:
            start_gene = end
    
        if reverse == 0:
            try:
                sequence = self.sequences.fetch(chrom, start, end).upper()
            except ValueError as e:
                self.logger.error(f"Error sequence {i}")
        else:
            try:
                sequence = self.sequences.fetch(chrom, end, start).upper()
                sequence = self.reverse_complement(sequence)
            except ValueError as e:
                self.logger.error(f"Error sequence {i}")
        
        encoded_sequence = self.gen_tokenizer.encode_plus(sequence, return_offsets_mapping=True)
        tokens_before = encoded_sequence['input_ids'][1:-1]
        mapping = encoded_sequence['offset_mapping'][1:-1]
       
        for i, (start_i, end_i) in enumerate(mapping):
            token_id = tokens_before[i]
            if (token_id == 5):
                if i > 0:
                    length = end_i - mapping[i-1][1] 
                else:
                    length = end_i
            else:
                length = end_i - start_i 
            token_lengths.append((token_id, length))
            
        if reverse == 1: 
            token_lengths.reverse()
        token_lengths_df = pd.DataFrame(token_lengths, columns=['token_id', 'length'])
        token_lengths_df['start'] = token_lengths_df['length'].cumsum().shift(fill_value=0) + start_gene 
        token_lengths_df['end'] = token_lengths_df['start'] + token_lengths_df['length']
        token_lengths_df['chrom'] = chrom
        if reverse == 1: 
            token_lengths_df = token_lengths_df[::-1].reset_index(drop=True)
        return start_gene, token_lengths_df

    # def process_region_signals(self, bw_handler, chrom, starts, ends, l, strand):
    #     reverse = 0 if strand == "+" else 1

    #     signals = np.zeros(l, dtype=np.float32)
        
    #     if reverse == 0:
    #         region_start = int(starts[0])
    #         region_end = int(ends[-1])
    #     else:
    #         region_start = int(starts[-1])
    #         region_end = int(ends[0])
        
    #     if region_start >= region_end:
    #         return signals
            
    #     try:
    #         intervals = bw_handler.intervals(chrom, region_start, region_end)
    #         if not intervals:
    #             return signals
                
    #         region_size = region_end - region_start
    #         position_values = np.zeros(region_size, dtype=np.float32)
    #         for interval_start, interval_end, value in intervals:
    #             rel_start = max(0, interval_start - region_start)
    #             rel_end = min(region_size, interval_end - region_start)
    #             if rel_start < rel_end:
    #                 position_values[rel_start:rel_end] = value
            
    #         for j in range(l):
    #             token_start = max(0, int(starts[j]) - region_start)
    #             token_end = min(region_size, int(ends[j]) - region_start)
    #             if token_start < token_end:
    #                 signals[j] = np.sum(position_values[token_start:token_end])
    #     except Exception as e:
    #         self.logger.error(f"Error processing signals for {chrom}:{region_start}-{region_end}: {e}")
    #     return signals

    def process_region_signals(self, bw_handler, chrom, starts, ends, l, strand):
        reverse = 0 if strand == "+" else 1
        signals = np.zeros(l, dtype=np.float32)

        if reverse == 0:
            region_start, region_end = int(starts[0]), int(ends[-1])
        else:
            region_start, region_end = int(starts[-1]), int(ends[0])

        if region_start >= region_end:
            return signals

        intervals = bw_handler.intervals(chrom, region_start, region_end)
        if not intervals:
            return signals

        region_size = region_end - region_start
        pos = np.zeros(region_size, dtype=np.float32)

        for s, e, v in intervals:
            rs = max(0, s - region_start)
            re = min(region_size, e - region_start)
            if rs < re:
                pos[rs:re] = v

        pref = np.empty(region_size + 1, dtype=np.float32)
        pref[0] = 0.0
        np.cumsum(pos, out=pref[1:])   # pref[i] = sum(pos[:i])

        ts = np.clip(starts[:l].astype(np.int64) - region_start, 0, region_size)
        te = np.clip(ends[:l].astype(np.int64)   - region_start, 0, region_size)
        return pref[te] - pref[ts]

    def precompute_signals(self):
        self.logger.info(f"Precomputing signals to {self.signals_cache_path}")
        temp_path = f"{self.signals_cache_path}.{os.getpid()}.temp"
        
        try:
            if not self.files_opened:
                self.open_files()

            with h5py.File(temp_path, "w") as h5f:
                pbar = tqdm.tqdm(total=len(self.genes), desc="Computing signals")
                
                for idx in range(len(self.genes)):
                    gene_id = self.genes.iloc[idx]['gene_id']
                    gene_group = self.h5_cache[gene_id]
                    
                    input_ids = np.array(gene_group['input_ids'])
                    starts = np.array(gene_group['starts'])
                    ends = np.array(gene_group['ends'])
                    chrom  = gene_group.attrs['chrom']
                    strand = gene_group.attrs['strand']

                    assert strand == self.genes.iloc[idx]['strand']
                    assert chrom == self.genes.iloc[idx]['chromosome']

                    l = min(len(input_ids), self.gen_max_seq_len)
                    
                    bigwig_signals = np.zeros((l, len(self.bigWigHandlers)), dtype=np.float32)

                    for i_key, (key, bw_pair) in enumerate(self.bigWigHandlers.items()):
                        track_signals = self.process_region_signals(bw_pair[strand], chrom, starts[:l], ends[:l], l, strand)
                    
                        if self.norm_bw:
                            try:
                                norm_factor = self.coverage_norm[key][strand]
                                if norm_factor > 0:
                                    track_signals /= norm_factor
                                else:
                                    self.logger.warning(f"Zero normalization factor for {key}, strand {strand}")
                            except KeyError:
                                self.logger.warning(f"Missing normalization factor for {key}, strand {strand}")
                    
                        bigwig_signals[:, i_key] = track_signals

                    
                    signals_group = h5f.create_group(gene_id)
                    signals_group.create_dataset('signals', data=bigwig_signals)
                    
                    if idx % 100 == 0: 
                        h5f.flush()
                    
                    pbar.update(1)
                
                pbar.close()
                h5f.flush()
            
            os.rename(temp_path, self.signals_cache_path)
            self.signals_cache = h5py.File(self.signals_cache_path, "r")
            
        except Exception as e:
            self.logger.error(f"Error creating signals cache: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def load_panel_genes(self, panel_genes_path):
        """Gene order of the tf sequence: one gene id per line.

        The position of a gene in the tf sequence is its only identity for the model,
        so it must be the order the panel model was trained with.
        """
        with open(panel_genes_path, "r", encoding="utf-8") as f:
            genes = [line.strip() for line in f if line.strip()]
        if len(set(genes)) != len(genes):
            raise ValueError(f"Duplicated gene ids in {panel_genes_path}")
        self.logger.info(f"tf panel: {len(genes)} genes from {panel_genes_path}")
        return genes

    def _init_tf_tokens(self, tf_tokenizer):
        tokenizer = AutoTokenizer.from_pretrained(tf_tokenizer)
        vocab = tokenizer.get_vocab()
        bin_ids = []
        while f"[BIN_{len(bin_ids)}]" in vocab:
            bin_ids.append(vocab[f"[BIN_{len(bin_ids)}]"])
        if len(bin_ids) < 2:
            raise ValueError(f"No [BIN_i] tokens found in the tf tokenizer at {tf_tokenizer}")
        for name in ("bos", "cls", "unk", "pad", "mask"):
            if getattr(tokenizer, f"{name}_token_id") is None:
                raise ValueError(f"tf tokenizer at {tf_tokenizer} has no {name} token")

        self.tf_bin_ids = np.asarray(bin_ids, dtype=np.int16)
        self.tf_n_bins = len(bin_ids)
        self.tf_bos_id = tokenizer.bos_token_id
        self.tf_cls_id = tokenizer.cls_token_id
        self.tf_unk_id = tokenizer.unk_token_id
        self.tf_pad_id = tokenizer.pad_token_id
        self.tf_mask_id = tokenizer.mask_token_id

    def _transform_tpm(self, values):
        values = np.asarray(values, dtype=np.float32)
        if self.transform_targets_tpm is None:
            return values
        # log of negative values (normalized datasets) gives NaN: such values become unknown
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.asarray(self.transform_targets_tpm(values), dtype=np.float32)

    def load_tf_bin_edges(self, bin_edges_path):
        """Quantile bin edges of the panel model (make_panel_files.py), in the transformed scale."""
        edges = np.load(bin_edges_path)
        if edges.shape != (self.tf_n_bins - 1,):
            raise ValueError(f"{bin_edges_path} holds {edges.shape} edges, expected {self.tf_n_bins - 1}")
        return edges

    def build_control_panels(self):
        """Per dataset: [BOS] + one token per panel gene (in tf_genes order) + [CLS], from its non-targeting profile.

        A gene gets [UNK] when the profile has no such gene or its value is empty/NaN/inf
        (also after the transform), otherwise the token of its bin. The knocked-down gene gets
        [BIN_0] and the predicted gene [MASK] later, per sample.
        """
        n_genes = len(self.tf_genes)
        panels = {}
        for dataset, key in self.control_keys.items():
            profile = self.tpm_lookup[key].iloc[:, 0]
            values = self._transform_tpm(profile.reindex(self.tf_genes).to_numpy(dtype=np.float32))
            known = np.isfinite(values)
            bins = np.searchsorted(self.tf_bin_edges, values[known], side="right")

            ids = np.full(n_genes + 2, self.tf_unk_id, dtype=np.int16)
            ids[0] = self.tf_bos_id
            ids[-1] = self.tf_cls_id
            ids[1:-1][known] = self.tf_bin_ids[bins]
            panels[dataset] = torch.from_numpy(ids)
            outside = len(set(profile.index) - set(self.tf_gene_pos))
            self.logger.info(
                f"tf panel of {dataset} ({key}): {int(known.sum())}/{n_genes} genes known, "
                f"{outside} genes of the profile are not in the panel"
            )

        self.key_knockdown_pos = {key: self.tf_gene_pos.get(self.key_perturbation[key]) for key in self.track_keys}
        absent = [self.key_perturbation[k] for k, pos in self.key_knockdown_pos.items() if pos is None]
        if absent:
            self.logger.warning(f"{len(absent)} knocked-down genes are not in the panel, their tracks get no [BIN_0]: {absent[:5]}")

        # rows that pad an incomplete chunk of tracks: their labels are masked out
        self.tf_unknown_panel = torch.full((n_genes + 2,), self.tf_unk_id, dtype=torch.int16)
        self.tf_unknown_panel[0] = self.tf_bos_id
        self.tf_unknown_panel[-1] = self.tf_cls_id
        return panels

    def track_panel(self, key):
        """The panel of the track: the non-targeting panel of its dataset, its knocked-down gene in [BIN_0]."""
        panel = self.control_panels[self.key_dataset[key]].clone()
        pos = self.key_knockdown_pos[key]
        if pos is not None:
            panel[pos] = int(self.tf_bin_ids[0])
        return panel

    def build_queue(self):
        """Samples: (gene of the intervals, chunk of its tracks).

        A gene is predicted only in the tracks whose matrix has it with a finite value, so a
        gene absent from a matrix is never queued for that track, and a gene absent from every
        matrix is not queued at all. The knocked-down gene is not predicted in its own track.
        """
        track_genes = {}
        for key in self.track_keys:
            profile = self.tpm_lookup[key].iloc[:, 0]
            known = np.isfinite(self._transform_tpm(profile.to_numpy(dtype=np.float32)))
            track_genes[key] = set(profile.index[known]) - {self.key_perturbation[key]}

        self.gene_tracks = []  # (row in self.genes, tracks where the gene is predicted)
        self.queue = []        # (index in self.gene_tracks, chunk of its tracks)
        for idx, gene_id in enumerate(self.genes["gene_id"]):
            keys = [key for key in self.track_keys if gene_id in track_genes[key]]
            if not keys:
                continue
            n_chunks = (len(keys) - 1) // self.n_keys + 1
            self.queue.extend((len(self.gene_tracks), chunk) for chunk in range(n_chunks))
            self.gene_tracks.append((idx, keys))
        self.logger.info(
            f"queue: {len(self.queue)} samples, {len(self.gene_tracks)} of {len(self.genes)} interval genes, "
            f"{len(self.track_keys)} tracks"
        )

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, idx):
        gene_tracks_idx, chunk_idx = self.queue[idx]
        original_idx, gene_keys = self.gene_tracks[gene_tracks_idx]
        gene_id = self.genes.iloc[original_idx]['gene_id']
        gene_group = self.h5_cache[gene_id]
        selected_keys = self._get_selected_keys_for_gene(gene_id, gene_keys, chunk_idx)
        n_real = len(selected_keys)

        if self.bw and not self.files_opened:
            self.open_files()

        n_tokens = gene_group["input_ids"].shape[0]
        L = min(n_tokens, self.gen_max_seq_len - 2)
        assert L>0, f"Empty token sequence for gene_id={gene_id}"
        
        input_ids = gene_group["input_ids"][:L]
        starts    = gene_group["starts"][:L]
        ends      = gene_group["ends"][:L]
        chrom  = gene_group.attrs['chrom']
        strand    = gene_group.attrs['strand']
        assert strand == self.genes.iloc[original_idx]['strand']
        assert chrom == self.genes.iloc[original_idx]['chromosome']

        cls_id = self.gen_tokenizer.cls_token_id
        sep_id = self.gen_tokenizer.sep_token_id
        assert (cls_id is not None) and (sep_id is not None), "Tokenizer must have CLS/SEP"

        tok = torch.as_tensor(input_ids, dtype=torch.long)
        seq_input_ids = torch.cat([tok.new_tensor([cls_id]), tok, tok.new_tensor([sep_id])], dim=0)
        seq_attn_mask = torch.ones(seq_input_ids.size(0), dtype=torch.long)
        seq_token_types = torch.zeros(seq_input_ids.size(0), dtype=torch.long)

        batch_input_ids   = seq_input_ids.unsqueeze(0).expand(self.n_keys, -1)
        batch_attn_mask   = seq_attn_mask.unsqueeze(0).expand(self.n_keys, -1)
        batch_token_types = seq_token_types.unsqueeze(0).expand(self.n_keys, -1)

        labels = torch.zeros((self.n_keys, L + 2, 1), dtype=torch.float32)
        labels_mask = torch.zeros((self.n_keys, L + 2, 1), dtype=torch.bool)

        if self.bw:
            if self.signals_cache is not None:
                bigwig_signals = np.array(self.signals_cache[gene_id]['signals'])  
            else:
                N_tracks = len(self.bigWigHandlers)
                bigwig_signals = np.zeros((L, N_tracks), dtype=np.float32)
                for i_key, (key, bw_pair) in enumerate(self.bigWigHandlers.items()):
                    bigwig_signals[:, i_key] = self.process_region_signals(
                        bw_pair[strand], chrom, starts, ends, L, strand
                    )
            cols = [self._bw_key_to_col[k] for k in selected_keys]
            bw_np = bigwig_signals[:L, cols].T  # (n_keys, L)
            if self.transform_targets_bw is not None:
                bw_np = self.transform_targets_bw(bw_np)
            labels[:n_real, 1:1+L, 0] = torch.from_numpy(bw_np)
            labels_mask[:n_real, 1:1+L, 0] = True

        tpm_values = np.full(self.n_keys, np.nan, dtype=np.float32)
        if self.tpm:
            for i_key, key in enumerate(selected_keys):
                if gene_id in self.tpm_lookup[key].index:
                    tpm_values[i_key] = float(self.tpm_lookup[key].loc[gene_id].iloc[0])
            if self.transform_targets_tpm is not None:
                tpm_values = self.transform_targets_tpm(tpm_values)

        tpm_mask = np.isfinite(tpm_values)                                 # (n_keys,)
        tpm_filled = np.where(tpm_mask, tpm_values, 0.0).astype(np.float32)

        filtered_keys = [k for k, m in zip(selected_keys, tpm_mask) if m]

        labels[:, 0, 0] = torch.from_numpy(tpm_filled)
        labels_mask[:, 0, 0] = torch.from_numpy(tpm_mask).bool()

        reverse = 0 if strand == "+" else 1
        if reverse == 0 :
            start_coord = starts[0]
            end_coord   = ends[L-1]
        else:
            start_coord = starts[L-1]
            end_coord   = ends[0]

        tf_ids = [self.track_panel(key) for key in selected_keys]
        tf_ids += [self.tf_unknown_panel] * (self.n_keys - len(tf_ids))
        tf_ids = torch.stack(tf_ids, dim=0).long()              # (n_keys, T)
        # the predicted gene must not see its own value in the panel
        target_pos = self.tf_gene_pos.get(gene_id)
        if target_pos is not None:
            tf_ids[:, target_pos] = self.tf_mask_id
        tf_attention_mask = torch.ones_like(tf_ids)             # (n_keys, T)

        if self.bw and not self.tpm: 
            features_selected_keys = []
        else:
            features_selected_keys = filtered_keys

        features = {
            "input_ids": batch_input_ids,          
            "attention_mask": batch_attn_mask,    
            "token_type_ids": batch_token_types,  
            "labels": labels,                    
            "labels_mask": labels_mask,                    
            "selected_keys": filtered_keys,      
            "gene_id": [gene_id] * len(filtered_keys),       
            # "name": self.genes.iloc[original_idx]['gene_name'],
            "chrom": chrom,
            "reverse": reverse,
            "start": start_coord,
            "end": end_coord,
            # per track, aligned with selected_keys: one chunk may hold tracks of several datasets
            "dataset_description": [self.key_description[k] for k in filtered_keys],
            "dataset_flag": torch.ones(self.n_keys, dtype=torch.float32),
            "tf_ids": tf_ids,
            "tf_attention_mask": tf_attention_mask,
        }

        return features

    def __del__(self):
        try:
            if hasattr(self, 'sequences') and self.sequences is not None:
                self.sequences.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'h5_cache') and self.h5_cache is not None:
                self.h5_cache.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'signals_cache') and self.signals_cache is not None:
                self.signals_cache.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'bigWigHandlers'):
                for d in self.bigWigHandlers.values():
                    for h in d.values():
                        try:
                            h.close()
                        except Exception:
                            pass
        except Exception:
            pass

    def describe(self):
        result = f"ExpressionDataset(n_genes={len(self.gene_tracks)}, n_tracks={len(self.track_keys)}, n_samples={len(self.queue)}, bw={self.bw}, tpm={self.tpm}"
        if hasattr(self, 'dataset_description'):
            result += f", dataset_description={self.dataset_description}"
        result += ")"
        return result

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            if hasattr(ds, 'open_files'):
                ds.open_files()
    else:
        if hasattr(dataset, 'open_files'):
            dataset.open_files()

class logtransform():
    def __init__(self, pseudocount=0.01):
        self.pseudocount = pseudocount
        self.rounddigits = int(abs(np.log(pseudocount)))

    def __call__(self, x):
        return np.log(x + self.pseudocount)

    def reverse(self, x):
        return np.round(np.exp(x) - self.pseudocount, self.rounddigits)


class multiplytransform():
    def __init__(self, coefficient=1.0):
        self.coefficient = float(coefficient)

    def __call__(self, x):
        return x * self.coefficient

    def reverse(self, x):
        if self.coefficient == 0:
            raise ValueError("Cannot reverse multiplytransform with coefficient=0")
        return x / self.coefficient
