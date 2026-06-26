"""Colab session bootstrap + §8b GO/NO-GO run — paste these cells into Colab.

Discipline: ONE codebase, TWO configs (PAPER_OUTLINE.md §00). Colab is just the same
repo with a GPU and bigger epochs. The smoke test (scripts/smoke_test.py) already proved
this exact pipeline runs end-to-end locally; here we run it for real on the GPU.

The point of this run is the §8b CRITICAL-PATH gate: train >=3 models, compute SRC per
model, evaluate cross-source collapse, and look at the SRC<->collapse coupling. If the
coupling is there, proceed to all 7 models. If not, STOP and rethink SRC before burning
more GPU (see PAPER_OUTLINE.md §8b).

Run the cells in order. Each `# --- Colab cell ---` block is one cell.
"""

# ============================================================================
# --- Colab cell 1: mount Drive, get code, restore data, install deps ---
# ============================================================================
"""
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DIR = '/content/drive/MyDrive/cxr_data'      # where package_for_drive.py put the zips
REPO = 'https://github.com/<you>/<repo>.git'        # or upload the code folder manually

import os, glob, subprocess, shutil
if not os.path.exists('/content/Chest_Xray'):
    subprocess.run(['git','clone', REPO, '/content/Chest_Xray'], check=True)
os.chdir('/content/Chest_Xray')

# restore data to FAST local disk (never read 60k loose files off mounted Drive)
os.makedirs('data/raw', exist_ok=True)
for z in glob.glob(f'{DRIVE_DIR}/*.zip'):
    print('unzipping', os.path.basename(z)); subprocess.run(['unzip','-q','-o', z, '-d','data/raw'], check=True)
# reuse the prebuilt manifest so splits match local exactly (reproducible)
if os.path.exists(f'{DRIVE_DIR}/manifest.csv'):
    shutil.copy(f'{DRIVE_DIR}/manifest.csv', 'data/manifest.csv')
else:
    subprocess.run(['python','scripts/build_manifest.py'], check=True)

subprocess.run(['pip','-q','install','-r','requirements.txt'], check=True)
import tensorflow as tf
print('GPU:', tf.config.list_physical_devices('GPU'))   # must be non-empty
"""

# ============================================================================
# --- Colab cell 2: §8b GO/NO-GO — train 3 diverse models, audit, couple ---
# ============================================================================
"""
import subprocess, sys

MODELS = ['densenet201', 'resnet50', 'vit']   # 3 architecturally-different models for spread
EPOCHS = 20                                   # preliminary signal; full 100-epoch runs come after GO

# 1) TRAIN (real GPU). --resume skips any already trained.
subprocess.run([sys.executable,'scripts/train.py','--models',*MODELS,
                '--epochs',str(EPOCHS),'--batch-size','64','--resume'], check=True)

# 2) EVALUATE in-domain (writes results/<m>/metrics.json + Table A)
subprocess.run([sys.executable,'scripts/evaluate.py','--models',*MODELS], check=True)

# 3) CSA audit + SRC certificate per model (the predictor)
import numpy as np, tensorflow as tf
from src.data.loaders import load_manifest, filter_rows, make_dataset, CLASS_TO_IDX
from src.shortcut import csa
from src.shortcut.src_certificate import emit_certificate

df = load_manifest()
audit_df = filter_rows(df, split='test', roles=['in_domain'])
# cap audit set for speed; real masks loaded if available, else CSA geometric fallback
audit_df = audit_df.groupby('disease', group_keys=False).apply(lambda g: g.sample(min(len(g),300), random_state=42))
ds = make_dataset(audit_df, batch_size=64, training=False, shuffle=False)
images = np.concatenate([b[0].numpy() for b in ds], axis=0)
y_true = np.array([CLASS_TO_IDX[c] for c in audit_df['disease']])[:len(images)]

for name in MODELS:
    model = tf.keras.models.load_model(f'results/{name}/{name}_best.keras')
    audit = {ch: csa.causal_effect(model, images, y_true, ch, masks=None, n_boot=1000)
             for ch in csa.ALL_CHANNELS}
    cert = emit_certificate(name, audit, f'results/{name}/certificate.json')
    print(f'{name}: SRC={cert["src"]:.3f} valid={cert["valid"]} '
          f'dominant={max(cert["per_channel"], key=cert["per_channel"].get)}')

# 4) CROSS-SOURCE collapse (the outcome) + 5) C3 COUPLING (the go/no-go)
from src.shortcut import cross_domain
cross_domain.run_cross_source_matrix(models=MODELS, batch_size=64)
coupling = cross_domain.couple_src_to_collapse()
"""

# ============================================================================
# --- Colab cell 3: READ THE GATE, then save everything back to Drive ---
# ============================================================================
"""
import json
c = json.load(open('results/c3_coupling.json'))
print(json.dumps(c, indent=2))

# §8b decision rule (decide the threshold BEFORE looking — see PAPER_OUTLINE.md §8b):
if c.get('n_models', 0) >= 3 and 'delta_acc' in c:
    r2 = c['delta_acc']['r2']; r = c['delta_acc']['r']
    print(f"\\nSRC -> delta_acc:  r={r:+.3f}  R2={r2:.3f}")
    print("GO  (proceed to all 7 models)" if (r > 0 and r2 >= 0.3)
          else "WEAK/NO-GO -> rethink SRC definition before full runs (see §8b fallback)")
else:
    print("Need 3 models with valid SRC + cross-source results.")

# save checkpoints + results back to durable Drive (Colab disk is ephemeral!)
import subprocess, datetime
stamp = datetime.date.today().isoformat()
subprocess.run(['cp','-r','results', f'/content/drive/MyDrive/cxr_data/results_{stamp}'], check=True)
print('saved -> Drive/cxr_data/results_'+stamp)
"""

# ============================================================================
# --- Colab cell 4 (ONLY AFTER 'GO'): train the remaining 4 models ---
# ============================================================================
"""
import subprocess, sys
REST = ['efficientnetb0','mobilenetv3large','xception','lenet5']
subprocess.run([sys.executable,'scripts/train.py','--models',*REST,
                '--epochs','100','--batch-size','64','--resume'], check=True)
# then re-run cell 2 steps 2-5 with all 7 models for the final C3 result.
"""

NOTES = """
- Drive holds a FEW zips (durable); Colab unzips to /content (fast). Never read 60k loose
  files directly off mounted Drive — extremely slow.
- Reusing the prebuilt manifest.csv keeps splits identical to local (reproducibility).
- Colab's /content is WIPED at session end — cell 3 copies results back to Drive. Do it.
- masks=None in the audit uses CSA's geometric fallbacks. To use REAL lung masks (sharper
  background channel), wire the mask paths from configs/datasets.yaml (masks:true sources)
  into csa.causal_effect(..., masks=...). Fine to defer past the go/no-go.
- Cross-source coverage is uneven (Pneumonia has no cross_source until RSNA is downloaded);
  cross_domain.py restricts both sets to the classes that have cross-source rows and reports it.
"""
