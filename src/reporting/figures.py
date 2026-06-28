"""Figures + LaTeX tables — P8 (PAPER_OUTLINE.md §7).

Reads the saved JSON/CSV artifacts (results/...) and emits the §7 figures and tables.
By design this consumes only on-disk results, so figures are regenerable WITHOUT retraining
(reproducibility, §10). Missing inputs are skipped with a printed note, not a crash — so the
small-data smoke test can run the whole reporting stage even when some stages were skipped.

Figures (§7): 2 per-channel effects+controls, 3 SRC bars, 4 SRC<->collapse/ECE (marquee),
5 XAI vs SRC, 6 accuracy/coverage, 7 SSP heatmap. (Fig 1 CSA schematic = hand-drawn, not here.)
Tables: A in-domain, C SRC, D XAI vs SRC (LaTeX).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DPI = 300


def _mpl():
    import matplotlib
    matplotlib.use("Agg")               # headless / Colab safe
    import matplotlib.pyplot as plt
    return plt


def _load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _models(results_dir):
    return sorted(d.name for d in Path(results_dir).iterdir()
                  if d.is_dir() and (d / f"{d.name}_best.keras").exists())


# ----------------------------------------------------------------- figures

def fig_channel_effects(results_dir, out):
    """Fig 2 — per-channel CSA effects + negative controls, per model (from certificate.json)."""
    plt = _mpl()
    rd = Path(results_dir)
    rows = []
    for m in _models(rd):
        cert = _load(rd / m / "certificate.json")
        if cert:
            rows.append((m, cert["per_channel"], cert.get("controls", {})))
    if not rows:
        print("[fig2] no certificates; skipped"); return False
    fig, ax = plt.subplots(figsize=(8, 5))
    import numpy as np
    channels = list(rows[0][1]) + list(rows[0][2])
    x = np.arange(len(channels)); w = 0.8 / len(rows)
    for i, (m, pc, ctrl) in enumerate(rows):
        vals = [pc.get(c, ctrl.get(c, 0.0)) for c in channels]
        ax.bar(x + i * w, vals, w, label=m)
    ax.set_xticks(x + 0.4); ax.set_xticklabels(channels, rotation=30, ha="right")
    ax.set_ylabel("causal effect (Δ true-class prob)"); ax.axhline(0, color="k", lw=0.5)
    ax.set_title("CSA per-channel effects (sham≈0 = valid)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig); return True


def fig_src_bars(results_dir, out):
    """Fig 3 — SRC per model with validity flag."""
    plt = _mpl()
    rd = Path(results_dir); data = []
    for m in _models(rd):
        cert = _load(rd / m / "certificate.json")
        if cert:
            data.append((m, cert["src"], cert.get("valid", False)))
    if not data:
        print("[fig3] no certificates; skipped"); return False
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [d[0] for d in data]
    ax.bar(names, [d[1] for d in data],
           color=["#2b8cbe" if d[2] else "#cccccc" for d in data])
    ax.set_ylabel("SRC"); ax.set_title("Shortcut Reliance Certificate (grey = invalid)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig); return True


def fig_coupling(results_dir, out):
    """Fig 4 (marquee) — SRC vs cross-source Δacc and Δece."""
    plt = _mpl()
    cs = _load(Path(results_dir) / "cross_source.json")
    rd = Path(results_dir)
    if not cs:
        print("[fig4] no cross_source.json; skipped"); return False
    pts = []
    for r in cs:
        cert = _load(rd / r["model"] / "certificate.json")
        if cert:
            pts.append((cert["src"], r["delta_acc"], r["delta_ece"], r["model"]))
    if not pts:
        print("[fig4] no SRC+collapse pairs; skipped"); return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, idx, ylab in ((axes[0], 1, "Δ accuracy (cross-source collapse)"),
                          (axes[1], 2, "Δ ECE (miscalibration)")):
        for s, da, de, m in pts:
            y = da if idx == 1 else de
            ax.scatter(s, y); ax.annotate(m, (s, y), fontsize=7)
        ax.set_xlabel("SRC"); ax.set_ylabel(ylab)
    axes[0].set_title("SRC → collapse"); axes[1].set_title("SRC → miscalibration")
    fig.suptitle("C3 coupling (marquee)"); fig.tight_layout()
    fig.savefig(out, dpi=DPI); plt.close(fig); return True


def fig_ssp_heatmap(results_dir, out):
    """Fig 7 — accuracy heatmap (models × perturbation, worst severity) from robustness.json."""
    plt = _mpl(); import numpy as np
    rd = Path(results_dir); rows, perts = [], None
    for m in _models(rd):
        rob = _load(rd / m / "robustness.json")
        if rob:
            pp = rob["per_perturbation"]; perts = perts or list(pp)
            rows.append((m, [min(pp[p]["accuracy"]) for p in perts]))
    if not rows:
        print("[fig7] no robustness.json; skipped"); return False
    fig, ax = plt.subplots(figsize=(8, 1 + 0.5 * len(rows)))
    mat = np.array([r[1] for r in rows])
    im = ax.imshow(mat, aspect="auto", cmap="inferno", vmin=0, vmax=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xticks(range(len(perts))); ax.set_xticklabels(perts, rotation=40, ha="right")
    fig.colorbar(im, ax=ax, label="worst-severity accuracy")
    ax.set_title("Robustness under SSP (worst severity)")
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig); return True


# ----------------------------------------------------------------- LaTeX tables

def table_src(results_dir, out):
    """Table C — SRC + per-channel + validity, as LaTeX."""
    rd = Path(results_dir); rows = []
    for m in _models(rd):
        c = _load(rd / m / "certificate.json")
        if c:
            pc = c["per_channel"]
            rows.append((m, c["src"], c.get("valid", False),
                         pc.get("border", 0), pc.get("background", 0), pc.get("source_signature", 0)))
    if not rows:
        print("[tableC] no certificates; skipped"); return False
    lines = [r"\begin{tabular}{lccccc}", r"\hline",
             r"Model & SRC & Valid & Border & Background & Source \\", r"\hline"]
    for m, src, v, b, bg, s in rows:
        lines.append(f"{m} & {src:.3f} & {'Y' if v else 'N'} & {b:.3f} & {bg:.3f} & {s:.3f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    Path(out).write_text("\n".join(lines)); return True


def table_baselines_lomo(results_dir, out):
    """Table E — SRC vs baseline predictors (R²) + LOMO out-of-sample R², as LaTeX."""
    rd = Path(results_dir)
    base = _load(rd / "src_vs_baselines.json"); lomo = _load(rd / "lomo.json")
    if not base:
        print("[tableE] no src_vs_baselines.json; skipped"); return False
    deps = ["delta_acc", "delta_ece", "delta_ece_post_ts"]
    preds = ["SRC", "in_domain_acc", "in_domain_ece", "out_of_lung_fraction"]
    lines = [r"\begin{tabular}{l" + "c" * len(deps) + "}", r"\hline",
             "Predictor & " + " & ".join(d.replace("_", "-") for d in deps) + r" \\", r"\hline"]
    for p in preds:
        cells = []
        for d in deps:
            v = base["comparison"].get(d, {}).get(p, {})
            cells.append(f"{v['r2']:.2f}" if "r2" in v else "--")
        lines.append(f"{p.replace('_','-')} & " + " & ".join(cells) + r" \\")
    if lomo:
        lines.append(r"\hline")
        cells = []
        for d in deps:
            v = lomo.get(d, {})
            cells.append(f"{v['lomo_r2']:.2f}" if isinstance(v, dict) and "lomo_r2" in v else "--")
        lines.append(r"LOMO (out-of-sample R$^2$) & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    Path(out).write_text("\n".join(lines)); return True


def table_audit_crosssource(results_dir, out):
    """Table A2 — per-model in-domain vs cross-source acc/ECE (+post-TS), as LaTeX."""
    cs = _load(Path(results_dir) / "cross_source.json")
    if not cs:
        print("[tableA2] no cross_source.json; skipped"); return False
    lines = [r"\begin{tabular}{lcccccc}", r"\hline",
             r"Model & In-Acc & Cross-Acc & $\Delta$Acc & In-ECE & Cross-ECE & ECE(post-TS) \\", r"\hline"]
    for r in cs:
        lines.append(f"{r['model']} & {r['in_domain_acc']:.3f} & {r['cross_source_acc']:.3f} & "
                     f"{r['delta_acc']:+.3f} & {r['in_domain_ece']:.3f} & {r['cross_source_ece']:.3f} & "
                     f"{r.get('cross_source_ece_post_ts', float('nan')):.3f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    Path(out).write_text("\n".join(lines)); return True


def fig_failure_panel(results_dir, out):
    """Fig 8 — curated qualitative panel (NOT all images): for the highest-SRC valid model show
    in-domain image + cross-source image + CSA(border) intervention + sham intervention, so the
    validity controls and a shortcut intervention are VISIBLE. Small grid, paper-ready."""
    plt = _mpl(); import numpy as np
    rd = Path(results_dir)
    # pick the highest-SRC VALID model
    best, best_src = None, -1
    for m in _models(rd):
        c = _load(rd / m / "certificate.json")
        if c and c.get("valid") and c["src"] > best_src:
            best, best_src = m, c["src"]
    if best is None:
        print("[fig8] no valid certificate; skipped"); return False
    try:
        import tensorflow as tf
        from src.data.loaders import load_manifest, filter_rows, make_dataset, CLASS_TO_IDX
        from src.shortcut import csa
        model = tf.keras.models.load_model(rd / best / f"{best}_best.keras")
        df = load_manifest()
        ind = filter_rows(df, split="test", roles=["in_domain"]).head(1)
        crs = filter_rows(df, roles=["cross_source"]).head(1)
        def first_img(sub):
            ds = make_dataset(sub, batch_size=1, training=False, shuffle=False)
            for b in ds: return b[0].numpy()[0]
        img_in = first_img(ind); img_cs = first_img(crs)
        panels = [("in-domain", img_in), ("cross-source", img_cs),
                  ("CSA: border", csa.intervene(img_cs, "border", None)),
                  ("sham (no-op)", csa.intervene(img_cs, "sham", None))]
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
        for ax, (title, im) in zip(axes, panels):
            ax.imshow(np.clip(im, 0, 1)); ax.set_title(title, fontsize=9); ax.axis("off")
        fig.suptitle(f"Failure-case panel — model {best} (SRC={best_src:.2f})", fontsize=10)
        fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig); return True
    except Exception as e:
        print(f"[fig8] skipped: {type(e).__name__}: {e}"); return False


# ----------------------------------------------------------------- orchestration

def generate_all(results_dir=None):
    """Emit every available figure/table; skip (don't crash) on missing inputs."""
    rd = Path(results_dir) if results_dir else (ROOT / "results")
    figs = rd / "figures"; tabs = rd / "tables"
    figs.mkdir(parents=True, exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)
    made = {
        "fig2_channel_effects": fig_channel_effects(rd, figs / "fig2_channel_effects.png"),
        "fig3_src_bars":        fig_src_bars(rd, figs / "fig3_src_bars.png"),
        "fig4_coupling":        fig_coupling(rd, figs / "fig4_coupling.png"),
        "fig7_ssp_heatmap":     fig_ssp_heatmap(rd, figs / "fig7_ssp_heatmap.png"),
        "fig8_failure_panel":   fig_failure_panel(rd, figs / "fig8_failure_panel.png"),
        "tableC_src":           table_src(rd, tabs / "table_c_src.tex"),
        "tableE_baselines_lomo": table_baselines_lomo(rd, tabs / "table_e_baselines_lomo.tex"),
        "tableA2_audit":        table_audit_crosssource(rd, tabs / "table_a2_audit.tex"),
    }
    (rd / "reporting_manifest.json").write_text(json.dumps(made, indent=2))
    print("[reporting] emitted:", {k: v for k, v in made.items() if v})
    return made
