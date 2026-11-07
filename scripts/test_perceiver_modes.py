"""Unit check for compressors/perceiver.py query modes (direction perceiver-learned). Stdlib + torch only.

Loads the compressor package without the rest of vlmeval (copies local_models/{helpers.py,compressors/} into a
temporary package), then checks: (1) ``pooled`` and ``learned`` forwards are bit-identical to a reference copy
of perceiver.py (``--ref <file or git revision>``, e.g. ``--ref 94091f4``); (2) ``learned_residual``
at init equals fixed pooling exactly, has the pooled token count / kept indices, and its latents and gates get
non-zero gradients; (3) a state dict saved from ``learned_residual`` reloads with strict=True.

    python scripts/test_perceiver_modes.py [--ref 94091f4]
"""
import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODELS = os.path.join(REPO, "VLMEvalKit", "vlmeval", "vlm", "neo", "local_models")


def load_package(ref_path=None):
    tmp = tempfile.mkdtemp(prefix="perceiver_test_")
    pkg = os.path.join(tmp, "lm")
    os.makedirs(os.path.join(pkg, "compressors"))
    open(os.path.join(pkg, "__init__.py"), "w").close()
    shutil.copy(os.path.join(LOCAL_MODELS, "helpers.py"), pkg)
    for f in ("__init__.py", "base.py", "registry.py", "fixed_pooling.py", "perceiver.py"):
        src = os.path.join(LOCAL_MODELS, "compressors", f)
        if f == "__init__.py":
            open(os.path.join(pkg, "compressors", f), "w").close()  # skip auto-discovery of every method
        else:
            shutil.copy(src, os.path.join(pkg, "compressors", f))
    if ref_path:
        if os.path.isfile(ref_path):
            src = open(ref_path).read()
        else:  # a git revision: take the committed perceiver.py of that revision
            rel = os.path.relpath(os.path.join(LOCAL_MODELS, "compressors", "perceiver.py"), REPO)
            src = subprocess.run(["git", "-C", REPO, "show", f"{ref_path}:{rel}"], check=True,
                                 capture_output=True, text=True).stdout
        with open(os.path.join(pkg, "compressors", "perceiver_ref.py"), "w") as f:
            f.write(src.replace('@register_compressor("perceiver")', '@register_compressor("perceiver_ref")'))
    sys.path.insert(0, tmp)
    fixed = importlib.import_module("lm.compressors.fixed_pooling")
    perc = importlib.import_module("lm.compressors.perceiver")
    ref = importlib.import_module("lm.compressors.perceiver_ref") if ref_path else None
    return fixed, perc, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="reference perceiver.py for bit-identity: a file path or a git revision")
    args = ap.parse_args()
    fixed, perc, ref = load_package(args.ref)
    torch.manual_seed(0)
    D, hd = 64, 16
    shapes = [(2, 6, 10), (1, 7, 9), (1, 5, 5)]
    xs = [torch.randn(*s, D) for s in shapes]

    for ratio in (2, 4, 8):
        cr = 1.0 / ratio
        fp = fixed.FixedPoolingCompressor(cr)
        # (1) bit-identity of the old modes with the reference file
        if ref is not None:
            for mode in ("pooled", "learned"):
                torch.manual_seed(1)
                new = perc.PerceiverCompressor(cr, hidden_size=D, head_dim=hd, queries=mode)
                torch.manual_seed(1)
                old = ref.PerceiverCompressor(cr, hidden_size=D, head_dim=hd, queries=mode)
                new.load_state_dict(old.state_dict(), strict=True)
                for x in xs:
                    a, b = new(x), old(x)
                    assert a.shape == b.shape and torch.equal(a, b), f"{mode} {ratio}x differs from reference"
                    h, w = x.shape[1], x.shape[2]
                    assert new.num_output_tokens(h, w) == old.num_output_tokens(h, w)
                    assert torch.equal(new.kept_indices(h, w), old.kept_indices(h, w))
            print(f"{ratio}x: pooled/learned bit-identical to reference")
        # (2) learned_residual at init == fixed pooling
        torch.manual_seed(2)
        m = perc.PerceiverCompressor(cr, hidden_size=D, head_dim=hd, queries="learned_residual")
        assert float(m.latents.abs().max()) == 0.0 and float(m.attn_gate) == 0.0 and float(m.mlp_gate) == 0.0
        for x in xs:
            h, w = x.shape[1], x.shape[2]
            y, y_fp = m(x), fp(x)
            assert y.shape == y_fp.shape, (y.shape, y_fp.shape)
            assert torch.equal(y, y_fp), f"learned_residual {ratio}x init != fixed pooling, max diff {(y - y_fp).abs().max()}"
            assert m.num_output_tokens(h, w) == fp.num_output_tokens(h, w)
            assert torch.equal(m.kept_indices(h, w), fp.kept_indices(h, w))
        # gradients reach the latents and the gates; perturbed latents change the output per position
        x = xs[0]
        m(x).pow(2).sum().backward()
        assert m.latents.grad is not None and float(m.latents.grad.abs().max()) > 0, "no gradient to latents"
        assert m.attn_gate.grad is not None and float(m.attn_gate.grad.abs()) > 0, "no gradient to attn_gate"
        assert m.mlp_gate.grad is not None and float(m.mlp_gate.grad.abs()) > 0, "no gradient to mlp_gate"
        with torch.no_grad():
            m.latents.normal_(std=1.0)
            d = (m(x) - fp(x))[0]  # [S, D]
            cos = torch.nn.functional.cosine_similarity(d[:, None], d[None], dim=-1)
            off = cos[~torch.eye(d.shape[0], dtype=torch.bool)].mean()
        assert float(off) < 0.9, f"latent contributions nearly identical across positions (cos {off:.3f})"
        # (3) round trip
        m2 = perc.PerceiverCompressor(cr, hidden_size=D, head_dim=hd, queries="learned_residual")
        m2.load_state_dict(m.state_dict(), strict=True)
        assert torch.equal(m2(x), m(x))
        n_lat = m.latents.numel()
        print(f"{ratio}x: learned_residual init == fixed pooling, grads ok, per-position latent cos {off:.3f}, "
              f"latents {n_lat} params")
    print("all checks passed")


if __name__ == "__main__":
    main()
