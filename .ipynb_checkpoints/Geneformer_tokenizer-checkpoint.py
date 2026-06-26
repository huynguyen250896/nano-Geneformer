import pickle
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch


# gc = Genecorpus
# Geneformer-V1-10M
# uses:
# - token_dictionary_gc30M.pkl
# - gene_median_dictionary_gc30M.pkl
# - gene_name_id_dict_gc30M.pkl
# - ensembl_mapping_dict_gc30M.pkl
# ...
# Geneformer-V2-104M
# Geneformer-V2-104M_CLcancer
# Geneformer-V2-316M
# use:
# - token_dictionary_gc104M.pkl
# - gene_median_dictionary_gc104M.pkl
# - gene_name_id_dict_gc104M.pkl
# - ensembl_mapping_dict_gc104M.pkl
TOKENIZER_REGISTRY = {
    "Geneformer-V1": {
        "token_dictionary": "assets/token_dictionary_gc30M.pkl",
        "gene_median": "assets/gene_median_dictionary_gc30M.pkl",
        "gene_name_id": "assets/gene_name_id_dict_gc30M.pkl",
        "ensembl_mapping": "assets/ensembl_mapping_dict_gc30M.pkl",
        "model_input_size": 2048,
        "special_token": False,
    },
    "Geneformer-V2-104M": {
        "token_dictionary": "assets/token_dictionary_gc104M.pkl",
        "gene_median": "assets/gene_median_dictionary_gc104M.pkl",
        "gene_name_id": "assets/gene_name_id_dict_gc104M.pkl",
        "ensembl_mapping": "assets/ensembl_mapping_dict_gc104M.pkl",
        "model_input_size": 4096,
        "special_token": True,
    },
    "Geneformer-V2-104M_CLcancer": {
        "token_dictionary": "assets/token_dictionary_gc104M.pkl",
        "gene_median": "assets/gene_median_dictionary_gc104M.pkl",
        "gene_name_id": "assets/gene_name_id_dict_gc104M.pkl",
        "ensembl_mapping": "assets/ensembl_mapping_dict_gc104M.pkl",
        "model_input_size": 4096,
        "special_token": True,
    },
    "Geneformer-V2-316M": {
        "token_dictionary": "assets/token_dictionary_gc104M.pkl",
        "gene_median": "assets/gene_median_dictionary_gc104M.pkl",
        "gene_name_id": "assets/gene_name_id_dict_gc104M.pkl",
        "ensembl_mapping": "assets/ensembl_mapping_dict_gc104M.pkl",
        "model_input_size": 4096,
        "special_token": True,
    },
}


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _rank_genes(gene_values, gene_tokens):
    order = np.argsort(-gene_values)
    return gene_tokens[order]


class GeneformerTokenizer:
    def __init__(
        self,
        token_dictionary,
        gene_median,
        gene_name_id=None,
        ensembl_mapping=None,
        model_input_size=2048,
        special_token=False,
        target_sum=10_000,
    ):
        self.gene_token_dict = token_dictionary
        self.gene_median_dict = gene_median
        self.gene_name_id_dict = gene_name_id or {}
        self.ensembl_mapping_dict = ensembl_mapping or {}
        
        self.model_input_size = model_input_size
        self.special_token = special_token
        self.target_sum = target_sum
        
        self.pad_token_id = self.gene_token_dict.get("<pad>", 0)
        self.cls_token_id = self.gene_token_dict.get("<cls>")
        self.eos_token_id = self.gene_token_dict.get("<eos>")

        if self.special_token:
            if self.cls_token_id is None or self.eos_token_id is None:
                raise ValueError(
                    "special_token=True requires '<cls>' and '<eos>' in token dictionary."
                )

    @classmethod
    def from_pretrained(cls, model_name, asset_dir=None):
        if model_name not in TOKENIZER_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(TOKENIZER_REGISTRY.keys())}"
            )
    
        if asset_dir is None:
            repo_root = Path(__file__).resolve().parent
        else:
            repo_root = Path(asset_dir).resolve()
    
        cfg = TOKENIZER_REGISTRY[model_name]
    
        token_dictionary = _load_pickle(repo_root / cfg["token_dictionary"])
        gene_median = _load_pickle(repo_root / cfg["gene_median"])
        gene_name_id = _load_pickle(repo_root / cfg["gene_name_id"])
        ensembl_mapping = _load_pickle(repo_root / cfg["ensembl_mapping"])
    
        return cls(
            token_dictionary=token_dictionary,
            gene_median=gene_median,
            gene_name_id=gene_name_id,
            ensembl_mapping=ensembl_mapping,
            model_input_size=cfg["model_input_size"],
            special_token=cfg["special_token"],
        )

    def encode_h5ad(
        self,
        path,
        chunk_size=512,
        return_tensors="pt",
    ):
        adata = sc.read_h5ad(path)

        if "ensembl_id" not in adata.var.columns:
            adata.var["ensembl_id"] = [
                self.gene_name_id_dict.get(str(gene).upper(), None)
                for gene in adata.var_names
            ]

        if "n_counts" not in adata.obs.columns:
            adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).reshape(-1)

        adata.var["ensembl_id_collapsed"] = [
            self.ensembl_mapping_dict.get(str(gene).upper(), None)
            for gene in adata.var["ensembl_id"]
        ]
        
        adata = adata[:, adata.var["ensembl_id_collapsed"].notna()].copy()
        
        if adata.n_vars == 0:
            raise ValueError("No genes remained after Ensembl ID mapping.")
        
        adata.var_names = adata.var["ensembl_id_collapsed"].astype(str)
        
        if not adata.var_names.is_unique:
            if sp.issparse(adata.X):
                X_df = None
                X = adata.X.tocsr()
            else:
                X_df = None
                X = sp.csr_matrix(adata.X)
        
            unique_genes = np.array(sorted(set(adata.var_names)))
            gene_to_col = {g: i for i, g in enumerate(unique_genes)}
        
            rows = []
            cols = []
            data = []
        
            X_csc = X.tocsc()
            for j, gene in enumerate(adata.var_names):
                col = X_csc[:, j]
                rows.extend(col.indices)
                cols.extend([gene_to_col[gene]] * len(col.data))
                data.extend(col.data)
        
            X_collapsed = sp.csr_matrix(
                (data, (rows, cols)),
                shape=(adata.n_obs, len(unique_genes)),
            )
        
            adata = sc.AnnData(
                X=X_collapsed,
                obs=adata.obs.copy(),
                var={"ensembl_id_collapsed": unique_genes},
            )
        
        ensembl_ids = adata.var["ensembl_id_collapsed"].astype(str).str.upper().to_numpy()

        keep = np.array(
            [
                (gene in self.gene_token_dict) and (gene in self.gene_median_dict)
                for gene in ensembl_ids
            ],
            dtype=bool,
        )

        if keep.sum() == 0:
            raise ValueError(
                "No genes matched the Geneformer token dictionary."
            )

        # Print some results after tokenization
        n_input_genes = len(ensembl_ids)
        n_reference_genes = len(self.gene_token_dict)
        
        # bỏ special tokens khỏi reference count
        if self.special_token:
            n_reference_genes -= 3   # <pad>, <cls>, <eos>
        else:
            n_reference_genes -= 1   # <pad>
        
        n_matched_genes = int(keep.sum())
        match_pct = 100 * n_matched_genes / n_reference_genes
        
        print(
            f"Original genes: {n_input_genes:,} | "
            f"Reference genes: {n_reference_genes:,} | "
            f"Matched genes: {n_matched_genes:,} ({match_pct:.1f}%)"
        )

        gene_tokens = np.array(
            [self.gene_token_dict[g] for g in ensembl_ids[keep]],
            dtype=np.int64,
        )

        gene_medians = np.array(
            [self.gene_median_dict[g] for g in ensembl_ids[keep]],
            dtype=np.float32,
        )

        X = adata[:, keep].X
        # Print some results after tokenization
        print(
            f"Cells after filtering: {adata.n_obs:,} | "
            f"Genes: {keep.sum():,}"
        )
        
        print((adata.n_obs, int(keep.sum())))
        n_counts_all = adata.obs["n_counts"].to_numpy().astype(np.float32)

        input_ids = []
        lengths = []

        for start in range(0, adata.n_obs, chunk_size):
            end = min(start + chunk_size, adata.n_obs)

            X_chunk = X[start:end]
            n_counts = n_counts_all[start:end, None]

            X_norm = X_chunk / n_counts * self.target_sum
            X_norm = X_norm / gene_medians

            if sp.issparse(X_norm):
                X_norm = X_norm.tocsr()
            else:
                X_norm = sp.csr_matrix(X_norm)

            for i in range(X_norm.shape[0]):
                row = X_norm[i]

                values = row.data
                indices = row.indices

                if len(values) == 0:
                    continue

                tokens = _rank_genes(values, gene_tokens[indices])

                if self.special_token:
                    tokens = tokens[: self.model_input_size - 2]
                    tokens = np.concatenate(
                        [
                            np.array([self.cls_token_id], dtype=np.int64),
                            tokens.astype(np.int64),
                            np.array([self.eos_token_id], dtype=np.int64),
                        ]
                    )
                else:
                    tokens = tokens[: self.model_input_size].astype(np.int64)

                input_ids.append(tokens)
                lengths.append(len(tokens))

        if len(input_ids) == 0:
            raise ValueError(
                "No valid cells were tokenized. Check raw counts and gene IDs."
            )

        max_len = min(max(lengths), self.model_input_size)

        padded = np.full(
            (len(input_ids), max_len),
            fill_value=self.pad_token_id,
            dtype=np.int64,
        )

        attention_mask = np.zeros(
            (len(input_ids), max_len),
            dtype=np.int64,
        )

        for i, ids in enumerate(input_ids):
            ids = ids[:max_len]
            padded[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "length": torch.tensor(lengths, dtype=torch.long),
            }

        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "length": np.asarray(lengths, dtype=np.int64),
        }

    def encode(
        self,
        exprs,
        ensembl_ids,
        n_counts=None,
        return_tensors="pt",
    ):
        ensembl_ids = np.asarray([str(g).upper() for g in ensembl_ids])

        if len(ensembl_ids) != len(set(ensembl_ids)):
            raise ValueError(
                "Duplicate Ensembl IDs detected. nano-Geneformer expects unique Ensembl IDs."
            )

        if sp.issparse(exprs):
            X = exprs.tocsr()
        else:
            X = sp.csr_matrix(exprs)

        if n_counts is None:
            n_counts = np.asarray(X.sum(axis=1)).reshape(-1).astype(np.float32)
        else:
            n_counts = np.asarray(n_counts).reshape(-1).astype(np.float32)

        keep = np.array(
            [
                (gene in self.gene_token_dict) and (gene in self.gene_median_dict)
                for gene in ensembl_ids
            ],
            dtype=bool,
        )

        if keep.sum() == 0:
            raise ValueError(
                "No genes matched the Geneformer token dictionary."
            )

        X = X[:, keep]
        kept_genes = ensembl_ids[keep]

        gene_tokens = np.array(
            [self.gene_token_dict[g] for g in kept_genes],
            dtype=np.int64,
        )
        gene_medians = np.array(
            [self.gene_median_dict[g] for g in kept_genes],
            dtype=np.float32,
        )

        input_ids = []
        lengths = []

        for i in range(X.shape[0]):
            row = X[i]
            values = row.data
            indices = row.indices

            if len(values) == 0:
                continue

            values = values / n_counts[i] * self.target_sum
            values = values / gene_medians[indices]

            tokens = _rank_genes(values, gene_tokens[indices])

            if self.special_token:
                tokens = tokens[: self.model_input_size - 2]
                tokens = np.concatenate(
                    [
                        np.array([self.cls_token_id], dtype=np.int64),
                        tokens.astype(np.int64),
                        np.array([self.eos_token_id], dtype=np.int64),
                    ]
                )
            else:
                tokens = tokens[: self.model_input_size].astype(np.int64)

            input_ids.append(tokens)
            lengths.append(len(tokens))

        max_len = min(max(lengths), self.model_input_size)

        padded = np.full(
            (len(input_ids), max_len),
            fill_value=self.pad_token_id,
            dtype=np.int64,
        )
        attention_mask = np.zeros(
            (len(input_ids), max_len),
            dtype=np.int64,
        )

        for i, ids in enumerate(input_ids):
            ids = ids[:max_len]
            padded[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "length": torch.tensor(lengths, dtype=torch.long),
            }

        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "length": np.asarray(lengths, dtype=np.int64),
        }