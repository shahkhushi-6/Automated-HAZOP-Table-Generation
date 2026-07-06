# HAZOP GNN+T5 Model - AI-Powered Process Safety Analysis

Generate complete HAZOP (Hazard and Operability) analysis reports automatically using Graph Neural Networks + T5 Transformers.

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-red)
![License](https://img.shields.io/badge/license-proprietary-gray)

---

## 🎯 What This Does

Takes a **process P&ID (Piping & Instrumentation Diagram)** and generates a complete **HAZOP table** with:

- ✅ **Deviation**: Description of the hazard
- ✅ **Causes**: Why it could happen
- ✅ **Consequences**: What could go wrong
- ✅ **Safeguards**: Existing protections
- ✅ **Recommendations**: How to improve safety

**Input:** CSV files describing equipment, connections, and guidewords  
**Output:** Excel HAZOP table (ready for review by safety engineers)

---

## ⚡ Quick Start (5 minutes)

### Windows
```bash
# Clone/copy files to VS Code project folder
cd hazop-project
setup.bat         # One-time setup
python train.py   # Start training (requires data in ./data/ folder)
```

### Linux/macOS
```bash
cd hazop-project
chmod +x setup.sh
./setup.sh        # One-time setup
python train.py   # Start training
```

See **[SETUP_GUIDE.md](./SETUP_GUIDE.md)** for complete instructions.

---

## 📁 Project Structure

```
hazop-project/
├── data/
│   ├── system_0001/
│   │   ├── nodes.csv          # Equipment list
│   │   ├── adjacency.csv      # Connections
│   │   ├── edges.csv          # Pipe properties
│   │   └── hazop.csv          # Ground truth (training only)
│   └── system_0002/ ...
├── checkpoints/
│   ├── best_model.pt          # Trained model
│   ├── hparams.json           # Training config
│   ├── history.json           # Loss curves
│   └── test_results.json      # Test metrics
├── outputs/
│   └── hazop_output.xlsx      # Generated HAZOP table
├── dataset.py
├── model.py
├── train.py
├── infer.py
├── evaluate.py
├── requirements.txt
├── SETUP_GUIDE.md             # Full setup guide
├── EXAMPLE_*.csv              # Data format examples
├── setup.bat / setup.sh       # Auto setup scripts
└── README.md                  # This file
```

---

## 🔧 System Requirements

### Hardware
- **GPU**: RTX 4050, RTX 4000 Ada, or similar (8GB+ VRAM recommended)
- **CPU**: Intel Core i5 or better (for data loading)
- **RAM**: 16GB+
- **Disk**: 5GB+ (for model checkpoints and data)

### Software
- **Python**: 3.8 or 3.10, 3.11, 3.12
- **CUDA**: 13.0 compatible with PyTorch 2.x
- **pip**: Latest version (auto-upgraded by setup script)

### Verification
```bash
# After setup, check GPU availability
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

---

## 📊 Training

### Basic Usage
```bash
python train.py --data_dir ./data --epochs 50
```

### Recommended Settings (RTX 4050)
```bash
python train.py \
    --data_dir ./data \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --freeze_t5_epochs 5 \
    --patience 7
```

### Key Arguments
| Argument | Default | Notes |
|----------|---------|-------|
| `--data_dir` | `./data` | Folder with `system_*` directories |
| `--batch_size` | 16 | Reduce to 4-8 if out of memory |
| `--epochs` | 50 | Usually 30-50 for good convergence |
| `--lr` | 1e-4 | Learning rate; 1e-4 to 5e-5 typical |
| `--freeze_t5_epochs` | 5 | Train GNN only first N epochs |
| `--t5_model` | t5-small | Use t5-base for better quality (slower) |
| `--grad_accumulate_steps` | 1 | Use >1 to simulate larger batches |

### Output
- ✅ `checkpoints/best_model.pt` - Best checkpoint by validation loss
- ✅ `checkpoints/history.json` - Training curves
- ✅ `checkpoints/test_results.json` - Test set metrics

---

## 🔮 Inference (Generate HAZOP for New System)

### Usage
```bash
python infer.py \
    --checkpoint ./checkpoints/best_model.pt \
    --system_dir ./inference_data \
    --output_file ./outputs/hazop.xlsx
```

### What You Need
Place these files in `--system_dir`:
- `nodes.csv` - Equipment definitions (7 fields)
- `adjacency.csv` - Connection matrix (3+ fields)
- `edges.csv` - Pipe properties (3 fields)
- **NO hazop.csv** - Model will generate it

### Output
Generates Excel with 7 columns:
```
| Node | Guideword | Deviation | Causes | Consequences | Safeguards | Recommendations |
|------|-----------|-----------|--------|--------------|------------|-----------------|
| Pump | NO_FLOW   | [AI text] | [text] | [text]       | [text]     | [text]          |
```

### Customize
```bash
# Use specific guidewords
python infer.py \
    --checkpoint ./checkpoints/best_model.pt \
    --system_dir ./inference_data \
    --output_file ./hazop.xlsx \
    --guidewords NO_FLOW MORE_FLOW LESS_FLOW MORE_PRESSURE

# Better quality (slower)
python infer.py ... --num_beams 8

# Faster (lower quality)
python infer.py ... --num_beams 2
```

---

## 📋 Data Format

### nodes.csv
Required columns:
```csv
node_id,equipment_type,hazard_class,has_flow,has_pressure,has_temp,has_level,relief_device,has_spare
Pump-1,centrifugal_pump,flammable,1,1,0,0,0,1
HEX-1,shell_tube_exchanger,unknown,1,1,1,0,0,0
Vessel-1,vessel,toxic,1,1,1,1,1,0
```

**Equipment types supported:**
```
pump, centrifugal_pump, heat_exchanger, shell_tube_exchanger,
vessel, tank, storage, reactor, compressor, control_valve, valve,
separator, cooler, heater, filter, column, distillation_column, unknown
```

**Hazard classes:**
```
inert, flammable, corrosive, toxic, oxidising, cryogenic, unknown
```

Alternative column names (auto-detected):
- `has_flow_instrument` → `has_flow`
- `has_pressure_instrument` → `has_pressure`  
- `has_temp_instrument` → `has_temp`
- `has_level_instrument` → `has_level`

### adjacency.csv (Matrix Format)
```csv
node_id,Pump-1,HEX-1,Vessel-1
Pump-1,0,1,0
HEX-1,0,0,1
Vessel-1,0,0,0
```

Or LIST format:
```csv
source,target
Pump-1,HEX-1
HEX-1,Vessel-1
```

### edges.csv
One row per directed edge:
```csv
fluid_type,has_control_valve,has_check_valve
liquid,1,0
steam,0,1
gas,1,1
```

**Fluid types:**
```
liquid, gas, steam, two_phase, unknown
```

Alternative column name:
- `phase` → `fluid_type` (auto-detected)

### hazop.csv (Training Only - Not Needed for Inference)
```csv
node_id,guideword,deviation,causes,consequences,safeguards,recommendations
Pump-1,NO_FLOW,"Pump failure","Power loss; mechanical failure","No feed to HEX","Check valve; pressure relief","Install UPS"
```

**Valid guidewords:**
```
NO_FLOW, MORE_FLOW, LESS_FLOW, REVERSE_FLOW,
MORE_PRESSURE, LESS_PRESSURE,
MORE_TEMP, LESS_TEMP,
MORE_LEVEL, LESS_LEVEL,
MORE_CONCENTRATION, LESS_CONCENTRATION,
OTHER_THAN
```

---

## ⚠️ Fixed Issues

### ✅ Issue 1: bos_token_id Error
**Problem:** Model crashed on T5 encoder with missing `bos_token_id`  
**Fix:** Safe fallback in `model.py` - checks `decoder_start_token_id` → `bos_token_id` → defaults to 0

### ✅ Issue 2: edges.csv Phase Column
**Problem:** Code looked for `fluid_type` but data had `phase` column  
**Fix:** `read_edges()` now checks BOTH `fluid_type` AND `phase`, defaults gracefully

### ✅ Issue 3: Equipment Type Mismatch
**Problem:** Data had `centrifugal_pump`, code only knew `pump`  
**Fix:** Extended `EQUIPMENT_TYPES` dict with alternative names:
- `centrifugal_pump` → pump
- `shell_tube_exchanger` → heat_exchanger
- All with safe fallback to 0 (unknown)

### ✅ Issue 4: Flexible Column Naming
**Problem:** Instrument flags had inconsistent names  
**Fix:** All readers check multiple column name variations with `.get()`

---

## 🚀 Advanced Usage

### Resume Training from Checkpoint
```bash
# Modify train.py to load initial weights
# Then run:
python train.py --data_dir ./data --output_dir ./checkpoints_v2
```

### Use Larger T5 Model
```bash
python train.py --t5_model t5-base --batch_size 4
```

### Generate with Custom Guidewords
```bash
python infer.py \
    --checkpoint ./checkpoints/best_model.pt \
    --system_dir ./inference_data \
    --guidewords NO_FLOW MORE_FLOW LESS_FLOW REVERSE_FLOW
```

### Gradient Accumulation (Simulate Larger Batch)
```bash
# Batch size 8 × accumulate 4 = effective batch size 32
python train.py --batch_size 8 --grad_accumulate_steps 4
```

---

## 📈 Monitoring Training

### Watch Real-Time Progress
```
Epoch   1/50  train_loss=4.23  val_loss=3.89  deviation=3.21  causes=3.56  ...
Epoch   2/50  train_loss=3.99  val_loss=3.65  deviation=3.10  causes=3.42  ...
→ Saved best checkpoint (val_loss=3.65)
```

### Plot Training Curves
```python
import json
import matplotlib.pyplot as plt

with open("./checkpoints/history.json") as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
train_loss = [h["train_loss"] for h in history]
val_loss = [h["val_loss"] for h in history]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("training_curves.png")
plt.show()
```

### Check Test Results
```bash
cat ./checkpoints/test_results.json
# Output:
# {"test_loss": 2.34, "deviation": 1.89, "causes": 2.10, ...}
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Fix:** Reduce batch size or use gradient accumulation
```bash
python train.py --batch_size 4 --grad_accumulate_steps 2
```

### No Data Found
```
ValueError: No valid HAZOP instances loaded
```
**Fix:** Check folder structure (case-sensitive):
- `./data/system_0001/nodes.csv` ✓
- `./data/system_0001/adjacency.csv` ✓
- `./data/system_0001/edges.csv` ✓
- `./data/system_0001/hazop.csv` ✓

### Module Not Found
```
ModuleNotFoundError: No module named 'torch_geometric'
```
**Fix:** Reinstall PyG
```bash
pip install torch-geometric --force-reinstall
```

### Wrong Column Names
```
WARNING: Unknown guideword in HAZOP
```
**Fix:** Check spelling (case-sensitive) against `GUIDEWORDS` list in code

---

## 📚 Code Architecture

### dataset.py
- Flexible CSV readers with auto-detection
- Support for LIST and MATRIX adjacency formats
- Safe handling of missing/misnamed columns
- PyTorch Dataset and DataLoader integration

### model.py
- **GraphEncoder**: 2-layer GraphSAGE with edge features
- **HAZOPModel**: Full GNN+T5 pipeline
  - GNN extracts graph structure
  - T5 generates text for each HAZOP field
  - Field-specific prefix tokens for output control

### train.py
- Two-phase training: GNN-only then joint
- Gradient accumulation support
- Early stopping with patience
- Per-field loss tracking

### infer.py
- Batch generation over all nodes × guidewords
- Excel output with formatting
- Support for custom guideword selection

### evaluate.py
- Per-field loss computation
- Validation on test set

---

## 📝 License

Proprietary - For authorized use only

---

## 🤝 Support

**Quick help:**

1. Read **SETUP_GUIDE.md** for detailed setup
2. Check **EXAMPLE_*.csv** files for data format
3. Review error messages for specific fixes
4. Run with `--help` flag for all options:
   ```bash
   python train.py --help
   python infer.py --help
   ```

**Common issues & fixes in SETUP_GUIDE.md**

---

## ✨ Key Features

✅ **AI-Powered**: Uses state-of-the-art GNN+T5 architecture  
✅ **Fast Training**: 30-50 epochs on consumer GPU (~2 hours)  
✅ **High Quality**: Multi-field output (deviation, causes, consequences, etc.)  
✅ **Flexible Input**: Auto-detects column names and formats  
✅ **Production Ready**: Error handling, logging, checkpointing  
✅ **Easy to Use**: Simple CLI, comprehensive docs  
✅ **Extensible**: Modify model, add fields, customize T5 size  

---

## 🎓 How It Works

1. **Graph Encoding**: GNN learns structure of P&ID
2. **Hazard Context**: Combines node embedding with guideword
3. **Text Generation**: T5 decoder generates HAZOP text per field
4. **Multi-Task**: Learns all 5 fields (deviation, causes, etc.) jointly

Result: Comprehensive, AI-assisted HAZOP analysis in minutes!

---

**Ready to get started? See SETUP_GUIDE.md →**
