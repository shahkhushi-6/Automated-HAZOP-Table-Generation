"""
dataset.py — HAZOP Dataset loader
==================================
Reads the folder-per-system layout:

    <data_dir>/
        system_0001/
            nodes.csv       — equipment items and their features
            adjacency.csv   — which node connects to which (supports both
                              list format: source,target  OR  matrix format)
            edges.csv       — pipe / edge features (rows must align with edges)
            hazop.csv       — HAZOP rows (node, guideword, deviation, ...)
        system_0002/
            ...

One training INSTANCE = one row from hazop.csv + the full graph it belongs to.
Graphs are identical across instances from the same system; only the target
node, guideword, and output texts change.

Expected CSV column names
─────────────────────────
nodes.csv:
    node_id, equipment_type, hazard_class, has_flow, has_pressure,
    has_temp, has_level, relief_device, has_spare
    
    (Alternative names supported:)
    - has_flow_instrument → has_flow
    - has_pressure_instrument → has_pressure
    - has_temp_instrument → has_temp
    - has_level_instrument → has_level
    - equipment_type can be: centrifugal_pump, shell_tube_exchanger, etc.

adjacency.csv (LIST FORMAT):
    source, target
    Pump, HEX
    HEX, Vessel

adjacency.csv (MATRIX FORMAT):
    node_id, Pump, HEX, Vessel
    Pump,    0,    1,   0
    HEX,     0,    0,   1
    Vessel,  0,    0,   0

edges.csv (rows must align with edges from adjacency, one per edge):
    fluid_type, has_control_valve, has_check_valve
    
    (Alternative names supported:)
    - phase → fluid_type (liquid, gas, steam, two_phase)

hazop.csv:
    node_id, guideword, deviation, causes, consequences,
    safeguards, recommendations
"""

import os
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Categorical encodings
# ──────────────────────────────────────────────────────────────────────────────

# Equipment types - normalize various naming conventions
EQUIPMENT_TYPES = {
    "pump": 1, "centrifugal_pump": 1, "positive_displacement_pump": 1,
    "heat_exchanger": 2, "shell_tube_exchanger": 2, "plate_fin_exchanger": 2,
    "storage": 3, "tank": 3, "vessel": 3, "evaporator": 3,
    "compressor": 4, "control_valve": 5, "valve": 5,
    "reactor": 7, "separator": 8, "cooler": 9, "heater": 10, "filter": 11,
    "column": 12, "distillation_column": 12,
    "unknown": 0,
}

HAZARD_CLASSES = {
    "inert": 1, "flammable": 2, "corrosive": 3, "toxic": 4,
    "oxidising": 5, "cryogenic": 6, "unknown": 0,
}

# Fluid types - normalize phase naming
FLUID_TYPES = {
    "liquid": 1, "gas": 2, "steam": 3, "two_phase": 4,
    "unknown": 0,
}

# All possible guidewords
GUIDEWORDS = [
    "NO_FLOW", "MORE_FLOW", "LESS_FLOW", "REVERSE_FLOW",
    "MORE_PRESSURE", "LESS_PRESSURE",
    "MORE_TEMP", "LESS_TEMP",
    "MORE_LEVEL", "LESS_LEVEL",
    "MORE_CONCENTRATION", "LESS_CONCENTRATION",
    "OTHER_THAN",
]
GUIDEWORD2IDX = {gw: i for i, gw in enumerate(GUIDEWORDS)}

# Output text fields — in order they will be generated
OUTPUT_KEYS = ["deviation", "causes", "consequences",
               "safeguards", "recommendations"]

# Node feature vector length:
# [equip_type(norm), hazard_class(norm), has_flow, has_pressure, has_temp,
#  has_level, relief_device, has_spare]  → 8 features
NODE_FEAT_DIM = 8

# Edge feature vector length:
# [fluid_type(norm), has_control_valve, has_check_valve]  → 3 features
EDGE_FEAT_DIM = 3


def _safe_int(val, default: int = 0) -> int:
    """Safely convert value to int, returning default on failure."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _norm(val: int, max_val: int) -> float:
    """Normalise a categorical int to [0, 1]."""
    return val / max(max_val, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Low-level CSV readers
# ──────────────────────────────────────────────────────────────────────────────

def read_nodes(path: str) -> Tuple[List[str], torch.Tensor]:
    """
    Reads nodes.csv and returns node IDs and feature tensor.
    
    Handles flexible column naming for equipment types and instrument flags.

    Returns:
        node_ids  : list of string node identifiers, length N
        feat      : float tensor of shape (N, NODE_FEAT_DIM)
    """
    node_ids, rows = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"nodes.csv is empty: {path}")
        if "node_id" not in reader.fieldnames:
            raise ValueError(f"nodes.csv must have 'node_id' column: {path}")

        for row in reader:
            node_ids.append(row["node_id"].strip())
            
            # Equipment type - normalize to standard format
            et_str = row.get("equipment_type", "unknown").strip().lower()
            et = EQUIPMENT_TYPES.get(et_str, 0)
            
            # Hazard class
            hc_str = row.get("hazard_class", "unknown").strip().lower()
            hc = HAZARD_CLASSES.get(hc_str, 0)
            
            # Instrument flags - check multiple naming conventions
            has_flow = _safe_int(
                row.get("has_flow") or row.get("has_flow_instrument", 0))
            has_pressure = _safe_int(
                row.get("has_pressure") or row.get("has_pressure_instrument", 0))
            has_temp = _safe_int(
                row.get("has_temp") or row.get("has_temp_instrument", 0))
            has_level = _safe_int(
                row.get("has_level") or row.get("has_level_instrument", 0))
            
            feat_vec = [
                _norm(et, max(EQUIPMENT_TYPES.values())),
                _norm(hc, max(HAZARD_CLASSES.values())),
                float(has_flow),
                float(has_pressure),
                float(has_temp),
                float(has_level),
                float(_safe_int(row.get("relief_device", 0))),
                float(_safe_int(row.get("has_spare", 0))),
            ]
            rows.append(feat_vec)

    if not rows:
        raise ValueError(f"No node data found in {path}")

    feat = torch.tensor(rows, dtype=torch.float)
    return node_ids, feat


def read_adjacency(path: str,
                   node2idx: Dict[str, int]) -> List[Tuple[int, int]]:
    """
    Reads adjacency.csv and converts it to list of (src_idx, dst_idx) tuples.
    
    Supports two formats:
    1. LIST format: source, target (standard edge list)
    2. MATRIX format: node_id, <nodes...> (adjacency matrix with 0/1 values)
    
    Returns:
        list of (src_idx, dst_idx) tuples
    """
    edges = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"adjacency.csv is empty: {path}")

        fieldnames = reader.fieldnames
        if "node_id" not in fieldnames:
            raise ValueError(f"adjacency.csv must have 'node_id' column: {path}")

        # Detect format: LIST has 'source' and 'target', MATRIX doesn't
        if "source" in fieldnames and "target" in fieldnames:
            # LIST format: source, target
            for row in reader:
                src = row.get("source", "").strip()
                tgt = row.get("target", "").strip()
                if not src or not tgt:
                    continue
                if src in node2idx and tgt in node2idx:
                    edges.append((node2idx[src], node2idx[tgt]))
                else:
                    log.warning("Unknown nodes in adjacency (LIST): %s → %s",
                               src, tgt)
        else:
            # MATRIX format: node_id as rows, node names as columns
            target_nodes = [col for col in fieldnames if col != "node_id"]
            for row in reader:
                src = row.get("node_id", "").strip()
                if not src or src not in node2idx:
                    log.warning("Unknown source node in adjacency (MATRIX): %s",
                               src)
                    continue
                src_idx = node2idx[src]

                for tgt in target_nodes:
                    val = row.get(tgt, "").strip()
                    # Edge exists if value is "1" (string or int)
                    if val in ("1", 1):
                        if tgt in node2idx:
                            edges.append((src_idx, node2idx[tgt]))
                        else:
                            log.warning(
                                "Unknown target node in adjacency (MATRIX): %s",
                                tgt)

    return edges


def read_edges(path: str) -> List[List[float]]:
    """
    Reads edges.csv and returns edge feature vectors.
    
    One row per directed edge. Must align with edge_pairs from adjacency.csv.
    
    Handles flexible column naming:
    - phase → fluid_type
    - has_control_valve
    - has_check_valve
    
    Returns:
        list of edge feature vectors, one per edge
    """
    feat_rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            log.warning("edges.csv is empty; using default features")
            return feat_rows

        for row in reader:
            # Fluid type - check both "fluid_type" and "phase" column names
            fluid_str = (row.get("fluid_type") or row.get("phase", "unknown")).strip().lower()
            ft = FLUID_TYPES.get(fluid_str, 0)
            
            feat_rows.append([
                _norm(ft, max(FLUID_TYPES.values())),
                float(_safe_int(row.get("has_control_valve", 0))),
                float(_safe_int(row.get("has_check_valve", 0))),
            ])
    return feat_rows


def read_hazop(path: str,
               node2idx: Dict[str, int]) -> List[Dict]:
    """
    Reads hazop.csv and returns list of HAZOP instance dicts.
    
    Rows whose node_id or guideword cannot be resolved are skipped with a warning.
    
    Returns:
        list of dicts with keys:
            - node_idx: int index into graph
            - guideword: int guideword index
            - deviation, causes, consequences, safeguards, recommendations: str
    """
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"hazop.csv is empty: {path}")

        for row in reader:
            node_id = row.get("node_id", "").strip()
            guideword = row.get("guideword", "").strip().upper().replace(" ", "_")

            # Normalize common variations
            GUIDEWORD_ALIASES = {
                "MORE_TEMPERATURE": "MORE_TEMP",
                "LESS_TEMPERATURE": "LESS_TEMP",
                "HIGH_TEMPERATURE": "MORE_TEMP",
                "LOW_TEMPERATURE": "LESS_TEMP",
            }

            guideword = GUIDEWORD_ALIASES.get(guideword, guideword)

            if not node_id or not guideword:
                log.warning("Skipping incomplete HAZOP row: %s", row)
                continue

            if node_id not in node2idx:
                log.warning("Unknown node in HAZOP: %s", node_id)
                continue

            if guideword not in GUIDEWORD2IDX:
                log.warning("Unknown guideword in HAZOP: %s", guideword)
                continue

            records.append({
                "node_idx": node2idx[node_id],
                "guideword": GUIDEWORD2IDX[guideword],
                "deviation": row.get("deviation", "").strip(),
                "causes": row.get("causes", "").strip(),
                "consequences": row.get("consequences", "").strip(),
                "safeguards": row.get("safeguards", "").strip(),
                "recommendations": row.get("recommendations", "").strip(),
            })

    if not records:
        raise ValueError(f"No valid HAZOP records found in {path}")

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(system_dir: str) -> Optional[Tuple[PyGData, List[Dict], List[str]]]:
    """
    Build a graph from a system folder containing nodes.csv, adjacency.csv,
    edges.csv, and hazop.csv.

    Args:
        system_dir: path to system directory

    Returns:
        (graph, hazop_records, node_ids) or None if any required file is missing/invalid
    """
    system_dir = Path(system_dir)

    nodes_path = system_dir / "nodes.csv"
    adj_path = system_dir / "adjacency.csv"
    edges_path = system_dir / "edges.csv"
    hazop_path = system_dir / "hazop.csv"

    # Check all required files exist
    for p in [nodes_path, adj_path, edges_path, hazop_path]:
        if not p.exists():
            log.warning("Missing file in %s: %s — skipping system",
                       system_dir.name, p.name)
            return None

    try:
        # Read nodes and build node→idx mapping
        node_ids, node_feat = read_nodes(str(nodes_path))
        node2idx = {nid: i for i, nid in enumerate(node_ids)}

        # Read adjacency (automatic format detection)
        edge_pairs = read_adjacency(str(adj_path), node2idx)

        # Read edge features
        edge_feats = read_edges(str(edges_path))

        # Read HAZOP data
        hazop_records = read_hazop(str(hazop_path), node2idx)

    except Exception as e:
        log.warning("Error reading system %s: %s", system_dir.name, e)
        return None

    # If no edges, create self-loops (allows isolated nodes)
    if not edge_pairs:
        n = node_feat.size(0)
        edge_pairs = [(i, i) for i in range(n)]
        edge_feats = [[0.0] * EDGE_FEAT_DIM] * n
        log.debug("System %s: no edges, using self-loops", system_dir.name)

    # Align edge_pairs and edge_feats to same length
    # (edges.csv row count should match edge_pairs count, but trim to be safe)
    n_edges = min(len(edge_pairs), len(edge_feats))
    if n_edges < len(edge_pairs):
        log.warning(
            "System %s: edge count mismatch (edges=%d, features=%d) — "
            "trimming to %d",
            system_dir.name, len(edge_pairs), len(edge_feats), n_edges)
    edge_pairs = edge_pairs[:n_edges]
    edge_feats = edge_feats[:n_edges]

    # Convert edge_pairs to tensor format
    src_nodes = [e[0] for e in edge_pairs]
    dst_nodes = [e[1] for e in edge_pairs]

    # Build PyG Data object
    graph = PyGData(
        x=node_feat,                                 # (N, NODE_FEAT_DIM)
        edge_index=torch.tensor([src_nodes, dst_nodes], dtype=torch.long),
        edge_attr=torch.tensor(edge_feats, dtype=torch.float),  # (E, EDGE_FEAT_DIM)
    )

    return graph, hazop_records, node_ids


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class HAZOPInstance:
    """Lightweight container for a single training instance."""
    __slots__ = ("graph", "node_idx", "guideword_idx",
                 "target_texts", "system_id")

    def __init__(self, graph, node_idx, guideword_idx, target_texts, system_id):
        self.graph = graph
        self.node_idx = node_idx
        self.guideword_idx = guideword_idx
        self.target_texts = target_texts   # dict: key → str
        self.system_id = system_id


class HAZOPDataset(Dataset):
    """
    Full dataset spanning all systems and HAZOP instances.
    
    Iterates every system folder, builds graphs, expands to per-row instances
    (one instance per HAZOP row per system).
    """

    def __init__(self, data_dir: str, tokenizer, max_input_len: int = 64):
        """
        Args:
            data_dir: root data directory containing system_* folders
            tokenizer: T5Tokenizer (used only for validation; actual tokenization
                      happens in model.forward)
            max_input_len: unused; kept for API compatibility
        """
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.output_keys = OUTPUT_KEYS
        self.node_feat_dim = NODE_FEAT_DIM
        self.num_guidewords = len(GUIDEWORDS)

        self.instances: List[HAZOPInstance] = []
        self.system_ids: List[str] = []
        self._system_instance_map: Dict[str, List[int]] = {}

        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        system_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        if not system_dirs:
            raise ValueError(f"No system directories found in {data_dir}")

        log.info("Scanning %d system directories …", len(system_dirs))
        skipped = 0

        for sdir in system_dirs:
            result = build_graph(str(sdir))
            if result is None:
                skipped += 1
                continue

            graph, hazop_records, _ = result
            sys_id = sdir.name

            if sys_id not in self._system_instance_map:
                self._system_instance_map[sys_id] = []
                self.system_ids.append(sys_id)

            # Create one instance per HAZOP row in this system
            for rec in hazop_records:
                idx = len(self.instances)
                self.instances.append(HAZOPInstance(
                    graph=graph,
                    node_idx=rec["node_idx"],
                    guideword_idx=rec["guideword"],
                    target_texts={k: rec[k] for k in OUTPUT_KEYS},
                    system_id=sys_id,
                ))
                self._system_instance_map[sys_id].append(idx)

        log.info("Loaded %d instances from %d systems (%d skipped)",
                 len(self.instances), len(self.system_ids), skipped)

        if not self.instances:
            raise ValueError("No valid HAZOP instances loaded from dataset")

    def subset(self, system_ids: List[str]) -> "HAZOPSubset":
        """Return a subset of the dataset by system IDs."""
        indices = []
        for sid in system_ids:
            indices.extend(self._system_instance_map.get(sid, []))
        return HAZOPSubset(self, indices)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict:
        inst = self.instances[idx]
        return {
            "graph": inst.graph,
            "node_idx": torch.tensor(inst.node_idx, dtype=torch.long),
            "guideword_idx": torch.tensor(inst.guideword_idx, dtype=torch.long),
            "target_texts": inst.target_texts,  # dict of strings
        }


class HAZOPSubset(Dataset):
    """View over HAZOPDataset restricted to a list of instance indices."""

    def __init__(self, parent: HAZOPDataset, indices: List[int]):
        self.parent = parent
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.parent[self.indices[i]]


# ──────────────────────────────────────────────────────────────────────────────
# Collate function for DataLoader
# ──────────────────────────────────────────────────────────────────────────────

def hazop_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collates a list of HAZOP instances into a single batch dict.

    Graphs are batched using torch_geometric's Batch class, which stacks
    multiple graphs into one large disconnected graph.

    Args:
        batch: list of instance dicts from HAZOPDataset

    Returns:
        dict with keys:
            - batched_graph: PyGBatch (stacked graphs)
            - node_idxs_local: (B,) local node indices within each graph
            - node_idxs_global: (B,) global node indices in batched graph
            - guideword_idxs: (B,) guideword indices
            - target_texts: dict of lists, one list per output field
    """
    from torch_geometric.data import Batch as PyGBatch

    graphs = [item["graph"] for item in batch]
    node_idxs = torch.stack([item["node_idx"] for item in batch])
    guideword_idxs = torch.stack([item["guideword_idx"] for item in batch])
    target_texts = {
        key: [item["target_texts"][key] for item in batch]
        for key in OUTPUT_KEYS
    }

    batched_graph = PyGBatch.from_data_list(graphs)

    # Adjust node indices to point into batched graph's global node space
    # PyGBatch.ptr is cumulative node count: [0, N1, N1+N2, N1+N2+N3, ...]
    ptr = batched_graph.ptr  # shape (batch_size+1,)
    offsets = ptr[:-1]       # shape (batch_size,) — start node index of each graph
    global_node_idxs = node_idxs + offsets

    return {
        "batched_graph": batched_graph,
        "node_idxs_local": node_idxs,          # within their own graph
        "node_idxs_global": global_node_idxs,  # within batched graph
        "guideword_idxs": guideword_idxs,
        "target_texts": target_texts,
    }
