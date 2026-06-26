"""Colab session guide — SMALL CELLS for easy debugging (PAPER_OUTLINE.md §00, §8b).

Discipline: ONE codebase, TWO configs. Colab = same repo + GPU + bigger epochs.
The smoke test (scripts/smoke_test.py) already proved this pipeline runs end-to-end
locally; here we run it for real on the GPU.

Each `# --- CELL n ---` block below is ONE Colab cell. Run in order. If a cell fails,
fix and re-run JUST that cell — you don't lose the unzip or the training.

Goal = the §8b CRITICAL-PATH gate: train >=3 models, compute SRC per model, measure
cross-source collapse, look at the SRC<->collapse coupling. GO -> all 7. NO-GO ->
rethink SRC before more GPU (PAPER_OUTLINE.md §8b).
"""

# =========================== CELL 1 — mount Drive ===========================
"""
from google.colab import drive
drive.mount('/content/drive')
"""

# =========================== CELL 2 — clone code ============================
"""
import os, subprocess
REPO = 'https://github.com/kamlishgoswami/Chest_Xray.git'
if not os.path.exists('/content/Chest_Xray'):
    subprocess.run(['git','clone', REPO, '/content/Chest_Xray'], check=True)
os.chdir('/content/Chest_Xray')
print('cwd:', os.getcwd())
print(os.listdir('.'))
"""

# =================== CELL 3 — unzip data to fast local disk =================
# Slow cell (~8 GB). Only needs to run ONCE per session.
"""
import os, glob, subprocess
DRIVE_DIR = '/content/drive/MyDrive/cxr_data'
os.makedirs('data/raw', exist_ok=True)
for z in glob.glob(f'{DRIVE_DIR}/*.zip'):
    print('unzipping', os.path.basename(z))
    subprocess.run(['unzip','-q','-o', z, '-d','data/raw'], check=True)
print('unzipped folders:', os.listdir('data/raw'))
"""

# ===================== CELL 4 — bring the manifest =========================
"""
import os, shutil, subprocess
DRIVE_DIR = '/content/drive/MyDrive/cxr_data'
if os.path.exists(f'{DRIVE_DIR}/manifest.csv'):
    shutil.copy(f'{DRIVE_DIR}/manifest.csv', 'data/manifest.csv')
    print('manifest copied from Drive')
else:
    subprocess.run(['python','scripts/build_manifest.py'], check=True)
    print('manifest rebuilt locally')
"""

# ===================== CELL 5 — install deps + GPU check ===================
"""
import subprocess
subprocess.run(['pip','-q','install','-r','requirements.txt'], check=True)
import tensorflow as tf
print('GPU:', tf.config.list_physical_devices('GPU'))   # must NOT be empty
"""

# ============ CELL 6 — DATA PATH CHECK (stop here if not 0) ================
# The one real risk: zip is named Shenzhen_Montgomery.zip but the manifest expects
# a 'Shenzhen+Montgomery/' folder. This verifies the paths resolve.
"""
import pandas as pd, os
df = pd.read_csv('data/manifest.csv')
missing = [p for p in df['image_path'].head(300) if not os.path.exists(p)]
print('missing:', len(missing), '/ 300')
if missing:
    print('EXAMPLE missing path:', missing[0])
    print('actual folders under data/raw:', os.listdir('data/raw'))
    print('>>> STOP: fix folder mapping before training.')
else:
    print('>>> OK: data paths resolve. Continue.')
"""

# ===================== CELL 7 — train 3 diverse models =====================
# Slow cell (~20-40 min on T4). --resume skips any already trained.
"""
import subprocess, sys
MODELS = ['densenet201', 'resnet50', 'vit']
EPOCHS = 20
subprocess.run([sys.executable,'scripts/train.py','--models',*MODELS,
                '--epochs',str(EPOCHS),'--batch-size','64','--resume'], check=True)
"""

# ===================== CELL 8 — evaluate in-domain =========================
"""
import subprocess, sys
MODELS = ['densenet201', 'resnet50', 'vit']
subprocess.run([sys.executable,'scripts/evaluate.py','--models',*MODELS], check=True)
"""

# ================== CELL 9 — CSA audit + SRC certificate ===================
"""
import numpy as np, tensorflow as tf
from src.data.loaders import load_manifest, filter_rows, make_dataset, CLASS_TO_IDX
from src.shortcut import csa
from src.shortcut.src_certificate import emit_certificate
MODELS = ['densenet201', 'resnet50', 'vit']

df = load_manifest()
audit_df = filter_rows(df, split='test', roles=['in_domain'])
audit_df = audit_df.groupby('disease', group_keys=False).apply(
    lambda g: g.sample(min(len(g),300), random_state=42))
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
"""

# ============ CELL 10 — cross-source collapse + C3 coupling ================
"""
from src.shortcut import cross_domain
MODELS = ['densenet201', 'resnet50', 'vit']
cross_domain.run_cross_source_matrix(models=MODELS, batch_size=64)
cross_domain.couple_src_to_collapse()
"""

# ================= CELL 11 — READ THE §8b VERDICT =========================
# Decide the threshold BEFORE looking (see PAPER_OUTLINE.md §8b — no p-hacking).
"""
import json
c = json.load(open('results/c3_coupling.json'))
print(json.dumps(c, indent=2))
if c.get('n_models',0) >= 3 and 'delta_acc' in c:
    r2 = c['delta_acc']['r2']; r = c['delta_acc']['r']
    print(f"\\nSRC -> delta_acc:  r={r:+.3f}  R2={r2:.3f}")
    print("GO (proceed to all 7)" if (r > 0 and r2 >= 0.3)
          else "WEAK/NO-GO -> rethink SRC before full runs (PAPER_OUTLINE.md §8b)")
else:
    print("Need 3 models with valid SRC + cross-source results.")
"""

# ============ CELL 12 — SAVE results back to Drive (do not skip) ===========
# Colab's /content is WIPED at session end. This persists checkpoints + results.
"""
import subprocess, datetime
stamp = datetime.date.today().isoformat()
subprocess.run(['cp','-r','results', f'/content/drive/MyDrive/cxr_data/results_{stamp}'], check=True)
print('saved -> Drive/cxr_data/results_'+stamp)
"""

# ========= CELL 13 — ONLY AFTER 'GO': train remaining 4 models ============
"""
import subprocess, sys
REST = ['efficientnetb0','mobilenetv3large','xception','lenet5']
subprocess.run([sys.executable,'scripts/train.py','--models',*REST,
                '--epochs','100','--batch-size','64','--resume'], check=True)
# then re-run CELLS 8-12 with MODELS = all 7 for the final C3 result.
"""

NOTES = """
- Drive holds a FEW zips (durable); Colab unzips to /content (fast). Never read 60k loose
  files off mounted Drive.
- Reusing the prebuilt manifest.csv keeps splits identical to local (reproducibility).
- /content is wiped at session end -> CELL 12 copies results to Drive. Do it.
- masks=None uses CSA geometric fallbacks. Real lung masks (sharper background channel)
  can be wired from configs/datasets.yaml later — fine to defer past the go/no-go.
- Cross-source coverage is uneven (Pneumonia has no cross_source until RSNA downloaded);
  cross_domain.py restricts both sets to classes with cross-source rows and reports it.
"""
