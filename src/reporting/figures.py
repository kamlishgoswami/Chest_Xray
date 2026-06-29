"""Figures + LaTeX tables — P8 (PAPER_OUTLINE.md §7) — PUBLICATION STYLE.

Reads saved JSON/CSV artifacts (results/...) and emits journal-quality figures + LaTeX tables.
Consumes only on-disk results, so figures are regenerable WITHOUT retraining (reproducibility, §10).
Missing inputs are skipped (printed note), never crash, so partial runs still report what they have.

Style: a single _style() sets serif fonts, a colorblind-safe palette, light grids, panel labels
(a/b/c), fit lines with R², and exports BOTH .png (300 DPI) and .pdf (vector) for every figure.

Figures: 2 per-channel effects+controls · 3 SRC bars · 4 SRC↔collapse/ECE (marquee, with fit+R²) ·
5 XAI saliency panel (Grad-CAM/IG) · 5b in-lung-fraction vs SRC · 6 accuracy/coverage (abstention) ·
7 SSP robustness heatmap · 8 failure-case panel. (Fig 1 CSA schematic = hand-drawn.)
Tables: A2 cross-source audit · C SRC · E baselines/LOMO (LaTeX).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DPI = 300

# colorblind-safe palette (Wong 2011) — used consistently across figures
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
ACCENT = "#0072B2"      # primary (data/flow)
ACCENT2 = "#D55E00"     # intervention / highlight
VALID_C = "#009E73"     # valid certificate
INVALID_C = "#BBBBBB"   # invalid / control


def _zoo_load(ckpt):
    from src.models.zoo import load_model as _lm
    return _lm(str(ckpt))


def _style():
    """Set publication-quality matplotlib defaults once. Returns the pyplot module."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    })
    return plt


def _panel_label(ax, letter):
    """Add a bold (a)/(b)/(c) label in the top-left corner of an axis."""
    ax.text(-0.12, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")


def _save(fig, out):
    """Save BOTH png and pdf (vector) versions; return True."""
    out = Path(out)
    fig.savefig(out, dpi=DPI)
    fig.savefig(out.with_suffix(".pdf"))
    import matplotlib.pyplot as plt
    plt.close(fig)
    return True


def _load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _models(results_dir):
    return sorted(d.name for d in Path(results_dir).iterdir()
                  if d.is_dir() and (d / f"{d.name}_best.keras").exists())


def _short(name):
    """Compact, readable model labels for axes."""
    return {"densenet201": "DenseNet201", "efficientnetb0": "EffNet-B0", "resnet50": "ResNet50",
            "mobilenetv3large": "MobileNetV3", "xception": "Xception", "vit": "ViT",
            "lenet5": "LeNet5"}.get(name, name)


def _fit_line(ax, x, y, color=ACCENT2):
    """Draw an OLS fit line + annotate R² (if >=3 finite points). Returns r2 or nan."""
    import numpy as np
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3 or np.allclose(x[m].std(), 0):
        return float("nan")
    s, b = np.polyfit(x[m], y[m], 1)
    xs = np.linspace(x[m].min(), x[m].max(), 50)
    ax.plot(xs, s * xs + b, color=color, lw=2, alpha=0.85, zorder=1)
    r2 = float(np.corrcoef(x[m], y[m])[0, 1] ** 2)
    ax.text(0.04, 0.93, f"$R^2={r2:.2f}$", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.9))
    return r2


# ============================================================ core result figures

def fig_channel_effects(results_dir, out):
    """Fig 2 — per-channel CSA effects + controls, grouped bars per model."""
    plt = _style(); import numpy as np
    rd = Path(results_dir)
    rows = [(m, c["per_channel"], c.get("controls", {}))
            for m in _models(rd) if (c := _load(rd / m / "certificate.json"))]
    if not rows:
        print("[fig2] no certificates; skipped"); return False
    channels = list(rows[0][1]) + list(rows[0][2])
    pretty = [c.replace("_", " ").replace("source signature", "source-sig").title() for c in channels]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(channels)); w = 0.8 / len(rows)
    for i, (m, pc, ctrl) in enumerate(rows):
        vals = [pc.get(c, ctrl.get(c, 0.0)) for c in channels]
        ax.bar(x + i * w, vals, w, label=_short(m), color=PALETTE[i % len(PALETTE)],
               edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.axvspan(2.5, len(channels) - 0.5, color="0.92", zorder=0)  # shade the control region
    ax.text(len(channels) - 1.2, ax.get_ylim()[1] * 0.95, "controls", color="0.4",
            style="italic", ha="center", fontsize=9)
    ax.set_xticks(x + 0.4); ax.set_xticklabels(pretty, rotation=20, ha="right")
    ax.set_ylabel("Causal effect  (Δ true-class prob.)")
    ax.set_title("Per-channel shortcut reliance (sham ≈ 0 confirms validity)")
    ax.legend(ncol=2, loc="upper left")
    return _save(fig, out)


def fig_src_bars(results_dir, out):
    """Fig 3 — normalized SRC per model, coloured by validity."""
    plt = _style()
    rd = Path(results_dir)
    data = [(m, c["src"], c.get("valid", False))
            for m in _models(rd) if (c := _load(rd / m / "certificate.json"))]
    if not data:
        print("[fig3] no certificates; skipped"); return False
    data.sort(key=lambda d: (not d[2], -d[1]))  # valid first, then by SRC desc
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [_short(d[0]) for d in data]
    bars = ax.bar(names, [d[1] for d in data],
                  color=[VALID_C if d[2] else INVALID_C for d in data],
                  edgecolor="0.2", linewidth=0.6)
    for b, d in zip(bars, data):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02 * max(1, b.get_height()),
                f"{d[1]:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("SRC  (shortcut reliance / pathology reliance)")
    ax.set_title("Shortcut Reliance Certificate per model")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    # legend proxy
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=VALID_C, label="valid"), Patch(fc=INVALID_C, label="invalid")],
              loc="upper right")
    return _save(fig, out)


def fig_coupling(results_dir, out):
    """Fig 4 (MARQUEE) — SRC vs cross-source Δacc and Δece(post-TS), with fit lines + R²."""
    plt = _style(); import numpy as np
    rd = Path(results_dir)
    cs = _load(rd / "cross_source.json")
    if not cs:
        print("[fig4] no cross_source.json; skipped"); return False
    pts = [(c["src"], r["delta_acc"], r.get("delta_ece_post_ts", r.get("delta_ece")), r["model"])
           for r in cs if (c := _load(rd / r["model"] / "certificate.json"))]
    if not pts:
        print("[fig4] no SRC+collapse pairs; skipped"); return False
    src = np.array([p[0] for p in pts]); names = [p[3] for p in pts]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    specs = [(1, "Δ accuracy  (cross-source collapse)", "SRC → accuracy collapse", "a"),
             (2, "Δ ECE post-TS  (miscalibration)", "SRC → miscalibration  (headline)", "b")]
    for ax, (idx, ylab, title, lab) in zip(axes, specs):
        y = np.array([p[idx] for p in pts])
        for i, (s, yi, nm) in enumerate(zip(src, y, names)):
            ax.scatter(s, yi, s=70, color=PALETTE[i % len(PALETTE)], edgecolor="white",
                       linewidth=0.8, zorder=3)
            ax.annotate(_short(nm), (s, yi), fontsize=8, xytext=(5, 4),
                        textcoords="offset points", color="0.3")
        _fit_line(ax, src, y)
        ax.set_xlabel("SRC"); ax.set_ylabel(ylab); ax.set_title(title)
        _panel_label(ax, lab)
    fig.suptitle("Shortcut reliance predicts cross-source failure", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, out)


def fig_ssp_heatmap(results_dir, out):
    """Fig 7 — robustness heatmap (models × perturbation, worst-severity accuracy)."""
    plt = _style(); import numpy as np
    rd = Path(results_dir); rows, perts = [], None
    for m in _models(rd):
        rob = _load(rd / m / "robustness.json")
        if rob:
            pp = rob["per_perturbation"]; perts = perts or list(pp)
            rows.append((m, [min(pp[p]["accuracy"]) for p in perts]))
    if not rows:
        print("[fig7] no robustness.json; skipped"); return False
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.55 * len(rows)))
    mat = np.array([r[1] for r in rows])
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.6 else "black", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([_short(r[0]) for r in rows])
    ax.set_xticks(range(len(perts)))
    ax.set_xticklabels([p.replace("_", "\n") for p in perts], fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("worst-severity accuracy")
    ax.set_title("Robustness under synthetic covariate shift (SSP)")
    return _save(fig, out)


# ============================================================ XAI figures (NEW)

def fig_xai_summary(results_dir, out):
    """Fig 5b — XAI faithfulness + in-lung-fraction vs SRC (from xai.json + certificates)."""
    plt = _style(); import numpy as np
    rd = Path(results_dir)
    pts = []
    for m in _models(rd):
        xj = _load(rd / m / "xai.json"); cert = _load(rd / m / "certificate.json")
        if xj and cert:
            gc = xj.get("grad_cam", {})
            pts.append((cert["src"], gc.get("in_lung"), gc.get("deletion_auc"), m))
    if not pts:
        print("[fig5b] no xai.json; skipped"); return False
    src = np.array([p[0] for p in pts]); inl = np.array([p[1] for p in pts], float)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for i, (s, il, _, nm) in enumerate(pts):
        if il == il:
            ax.scatter(s, il, s=70, color=PALETTE[i % len(PALETTE)], edgecolor="white",
                       linewidth=0.8, zorder=3)
            ax.annotate(_short(nm), (s, il), fontsize=8, xytext=(5, 4),
                        textcoords="offset points", color="0.3")
    _fit_line(ax, src, inl)
    ax.set_xlabel("SRC"); ax.set_ylabel("Grad-CAM in-lung saliency fraction")
    ax.set_title("Higher shortcut reliance → less saliency inside the lungs")
    return _save(fig, out)


def fig_xai_panel(results_dir, out, n_examples=3):
    """Fig 5 — qualitative saliency panel: real CXR + Grad-CAM + Integrated-Gradients overlays
    for the highest-SRC valid model. Loads the model (works on the training Keras; skips gracefully
    on a version mismatch). Rows = example images, cols = [original, Grad-CAM, IG]."""
    plt = _style(); import numpy as np
    rd = Path(results_dir)
    best, bs = None, -1
    for m in _models(rd):
        c = _load(rd / m / "certificate.json")
        if c and c.get("valid") and c.get("src", -1) > bs:
            best, bs = m, c["src"]
    if best is None:
        print("[fig5] no valid certificate; skipped"); return False
    try:
        from src.data.loaders import load_manifest, filter_rows, make_dataset, CLASS_TO_IDX
        from src.xai import explain
        model = _zoo_load(rd / best / f"{best}_best.keras")
        df = filter_rows(load_manifest(), split="test", roles=["in_domain"])
        df = df.groupby("disease", group_keys=False).head(1).head(n_examples)
        ds = make_dataset(df, batch_size=1, training=False, shuffle=False)
        imgs = [b[0].numpy()[0] for b in ds][:n_examples]
        ys = [CLASS_TO_IDX[c] for c in df["disease"]][:len(imgs)]
        labels = list(df["disease"])[:len(imgs)]

        fig, axes = plt.subplots(len(imgs), 3, figsize=(8.5, 2.8 * len(imgs)))
        if len(imgs) == 1:
            axes = axes[None, :]
        col_titles = ["Chest X-ray", "Grad-CAM", "Integrated Gradients"]
        for r, (img, y, lab) in enumerate(zip(imgs, ys, labels)):
            base = np.clip(img, 0, 1)
            gc = explain.grad_cam(model, img, int(y))
            ig = explain.integrated_gradients(model, img, int(y))
            for c, (panel, cmap, overlay) in enumerate(
                    [(base, "gray", None), (base, "gray", gc), (base, "gray", ig)]):
                ax = axes[r, c]
                ax.imshow(panel, cmap=cmap)
                if overlay is not None:
                    ax.imshow(overlay, cmap="jet", alpha=0.45)
                ax.axis("off")
                if r == 0:
                    ax.set_title(col_titles[c], fontsize=11, fontweight="bold")
            axes[r, 0].text(-0.08, 0.5, lab, transform=axes[r, 0].transAxes,
                            rotation=90, va="center", ha="right", fontsize=10, fontweight="bold")
        fig.suptitle(f"Where {_short(best)} looks (SRC={bs:.2f})", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0.03, 0, 1, 0.96])
        return _save(fig, out)
    except Exception as e:
        print(f"[fig5] skipped (model load/render issue, fine on training env): {type(e).__name__}: {str(e)[:80]}")
        return False


def fig_abstention(results_dir, out):
    """Fig 6 — accuracy/coverage (selective prediction) curves per model, from abstention.json."""
    plt = _style()
    rd = Path(results_dir)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    any_data = False
    for i, m in enumerate(_models(rd)):
        ab = _load(rd / m / "abstention.json")
        if ab and "coverage" in ab:
            ax.plot(ab["coverage"], ab["accuracy"], marker="o", ms=3, lw=1.8,
                    color=PALETTE[i % len(PALETTE)], label=_short(m))
            any_data = True
    if not any_data:
        print("[fig6] no abstention.json; skipped"); return False
    ax.axhline(0.95, color="0.5", ls="--", lw=1)
    ax.text(0.02, 0.955, "95% target", fontsize=8, color="0.4")
    ax.set_xlabel("Coverage  (fraction of cases predicted)")
    ax.set_ylabel("Accuracy on covered cases")
    ax.set_title("Certificate-gated selective prediction")
    ax.legend(ncol=2)
    return _save(fig, out)


def fig_failure_panel(results_dir, out):
    """Fig 8 — curated failure-case panel: in-domain vs cross-source vs CSA(border) vs sham,
    for the highest-SRC valid model. Skips gracefully on a Keras load mismatch."""
    plt = _style(); import numpy as np
    rd = Path(results_dir)
    best, bs = None, -1
    for m in _models(rd):
        c = _load(rd / m / "certificate.json")
        if c and c.get("valid") and c.get("src", -1) > bs:
            best, bs = m, c["src"]
    if best is None:
        print("[fig8] no valid certificate; skipped"); return False
    try:
        from src.data.loaders import load_manifest, filter_rows, make_dataset
        from src.shortcut import csa
        model = _zoo_load(rd / best / f"{best}_best.keras")
        df = load_manifest()
        first = lambda sub: next(iter(make_dataset(sub, batch_size=1, training=False, shuffle=False)))[0].numpy()[0]
        img_in = first(filter_rows(df, split="test", roles=["in_domain"]).head(1))
        img_cs = first(filter_rows(df, roles=["cross_source"]).head(1))
        panels = [("In-domain", img_in, None), ("Cross-source", img_cs, None),
                  ("CSA: border removed", csa.intervene(img_cs, "border", None), ACCENT2),
                  ("Sham (no-op)", csa.intervene(img_cs, "sham", None), "0.5")]
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
        for ax, (title, im, ec) in zip(axes, panels):
            ax.imshow(np.clip(im, 0, 1), cmap="gray"); ax.axis("off")
            ax.set_title(title, fontsize=10, fontweight="bold")
            if ec:
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_edgecolor(ec); sp.set_linewidth(2.5)
                ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Failure-case panel — {_short(best)} (SRC={bs:.2f})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        return _save(fig, out)
    except Exception as e:
        print(f"[fig8] skipped (model load/render issue, fine on training env): {type(e).__name__}: {str(e)[:80]}")
        return False


# ============================================================ LaTeX tables

def table_src(results_dir, out):
    """Table C — SRC + per-channel + validity (LaTeX)."""
    rd = Path(results_dir)
    rows = [(m, c["src"], c.get("src_raw_mean", float("nan")), c.get("valid", False),
             c["per_channel"].get("border", 0), c["per_channel"].get("background", 0),
             c["per_channel"].get("source_signature", 0))
            for m in _models(rd) if (c := _load(rd / m / "certificate.json"))]
    if not rows:
        print("[tableC] no certificates; skipped"); return False
    L = [r"\begin{tabular}{lcccccc}", r"\hline",
         r"Model & SRC & raw-mean & Valid & Border & Background & Source \\", r"\hline"]
    for m, s, raw, v, b, bg, sr in rows:
        L.append(f"{_short(m)} & {s:.2f} & {raw:.3f} & {'Y' if v else 'N'} & "
                 f"{b:.3f} & {bg:.3f} & {sr:.3f} \\\\")
    L += [r"\hline", r"\end{tabular}"]
    Path(out).write_text("\n".join(L)); return True


def table_baselines_lomo(results_dir, out):
    """Table E — headline reviewer-defense: R² + partial-r|accuracy per predictor (LaTeX)."""
    rd = Path(results_dir)
    base = _load(rd / "src_vs_baselines.json"); lomo = _load(rd / "lomo.json")
    if not base:
        print("[tableE] no src_vs_baselines.json; skipped"); return False
    dep = "delta_ece_post_ts"
    preds = ["SRC_normalized", "SRC_raw_mean", "in_domain_acc", "in_domain_ece", "out_of_lung_fraction"]
    comp = base.get("comparison", {}).get(dep, {})
    L = [r"\begin{tabular}{lcc}", r"\hline",
         r"Predictor of cross-source $\Delta$ECE (post-TS) & $R^2$ & partial-$r\,|\,$acc. \\", r"\hline"]
    for p in preds:
        v = comp.get(p, {})
        r2 = f"{v['r2']:.2f}" if "r2" in v else "--"
        pr = v.get("partial_r_given_acc")
        prc = f"{pr:+.2f}" if isinstance(pr, (int, float)) and pr == pr else "--"
        L.append(f"{p.replace('_','-')} & {r2} & {prc} \\\\")
    if lomo and isinstance(lomo.get(dep), dict) and "lomo_r2" in lomo[dep]:
        L += [r"\hline", f"SRC LOMO (out-of-sample $R^2$) & {lomo[dep]['lomo_r2']:.2f} & -- \\\\"]
    L += [r"\hline", r"\end{tabular}"]
    Path(out).write_text("\n".join(L)); return True


def table_audit_crosssource(results_dir, out):
    """Table A2 — per-model in-domain vs cross-source acc/ECE (+post-TS) (LaTeX)."""
    cs = _load(Path(results_dir) / "cross_source.json")
    if not cs:
        print("[tableA2] no cross_source.json; skipped"); return False
    L = [r"\begin{tabular}{lcccccc}", r"\hline",
         r"Model & In-Acc & Cross-Acc & $\Delta$Acc & In-ECE & Cross-ECE & ECE$_{\text{post-TS}}$ \\",
         r"\hline"]
    for r in cs:
        L.append(f"{_short(r['model'])} & {r['in_domain_acc']:.3f} & {r['cross_source_acc']:.3f} & "
                 f"{r['delta_acc']:+.3f} & {r['in_domain_ece']:.3f} & {r['cross_source_ece']:.3f} & "
                 f"{r.get('cross_source_ece_post_ts', float('nan')):.3f} \\\\")
    L += [r"\hline", r"\end{tabular}"]
    Path(out).write_text("\n".join(L)); return True


# ============================================================ orchestration

def generate_all(results_dir=None):
    """Emit every available figure/table (png+pdf); skip (don't crash) on missing inputs."""
    rd = Path(results_dir) if results_dir else (ROOT / "results")
    figs = rd / "figures"; tabs = rd / "tables"
    figs.mkdir(parents=True, exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)
    made = {
        "fig2_channel_effects": fig_channel_effects(rd, figs / "fig2_channel_effects.png"),
        "fig3_src_bars":        fig_src_bars(rd, figs / "fig3_src_bars.png"),
        "fig4_coupling":        fig_coupling(rd, figs / "fig4_coupling.png"),
        "fig5_xai_panel":       fig_xai_panel(rd, figs / "fig5_xai_panel.png"),
        "fig5b_xai_summary":    fig_xai_summary(rd, figs / "fig5b_xai_summary.png"),
        "fig6_abstention":      fig_abstention(rd, figs / "fig6_abstention.png"),
        "fig7_ssp_heatmap":     fig_ssp_heatmap(rd, figs / "fig7_ssp_heatmap.png"),
        "fig8_failure_panel":   fig_failure_panel(rd, figs / "fig8_failure_panel.png"),
        "tableC_src":           table_src(rd, tabs / "table_c_src.tex"),
        "tableE_baselines_lomo": table_baselines_lomo(rd, tabs / "table_e_baselines_lomo.tex"),
        "tableA2_audit":        table_audit_crosssource(rd, tabs / "table_a2_audit.tex"),
    }
    (rd / "reporting_manifest.json").write_text(json.dumps(made, indent=2))
    print("[reporting] emitted:", {k: v for k, v in made.items() if v})
    return made
