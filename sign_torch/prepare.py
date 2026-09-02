"""Turn a directory of complexes into SIGN graphs, reusing the authors' own featurisation.

The chain, and which half is theirs:

    <id>_pocket.pdb  --openbabel-->  <id>_pocket.mol2      ours, see below
    mol2 + ligand    --gen_feature-->  coords, features    theirs, unchanged
    those            --cons_lig_pock_graph_with_spatial_context-->  a merged complex   theirs
    that             --build_complex_graph-->  a PyG Data                              ours

**Nothing ships `_pocket.mol2`.** Zero of 285 CASF-2016 core-set complexes have one and zero
of PDBbind v2019 either; both provide `_pocket.pdb`. Whatever distribution the authors worked
from, it is not the one on pdbbind.org, so the conversion has to happen here. openbabel does
it, and openbabel is already required for reading the ligand.

That conversion is a place to be suspicious rather than comfortable: pdb -> mol2 assigns bond
orders and atom types by perception, and this benchmark has already been bitten twice by a
chemistry toolkit quietly changing what it perceives. Worth revisiting if the trained model
underperforms for no other visible reason.

**Parallel workers need shared memory.** They return built `Data` objects, and torch passes
tensors between processes through /dev/shm, which podman caps at 64 MB by default — the run
then dies with "No space left on device" partway through rather than at the start. Pass
`--shm-size=4g`. Serial mode does not need it.

usage:
    podman run --shm-size=4g ... python -m sign_torch.prepare \
        --complexes /data --out /outputs/graphs.pt --workers 10
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import tempfile

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openbabel import pybel  # noqa: E402

from featurizer import Featurizer  # noqa: E402
from preprocess_pdbbind import (cons_lig_pock_graph_with_spatial_context,  # noqa: E402
                                gen_feature)

from .data import build_complex_graph  # noqa: E402

# the 4 x 9 ligand-atom / pocket-atom pairs the auxiliary interaction matrix is defined over
PAIR_IDS = [(a, b) for a in [6, 7, 8, 16]
            for b in [6, 7, 8, 16, 15, 9, 17, 35, 53]]


def ensure_pocket_mol2(complex_dir: str, cid: str, scratch: str) -> bool:
    """Write `<id>_pocket.mol2` beside a staged copy if only the pdb exists."""
    target = os.path.join(scratch, cid, f"{cid}_pocket.mol2")
    if os.path.exists(target):
        return True
    source = os.path.join(complex_dir, f"{cid}_pocket.pdb")
    if not os.path.exists(source):
        return False
    pocket = next(pybel.readfile("pdb", source))
    pocket.write("mol2", target, overwrite=True)
    return True


def stage(complex_dir: str, cid: str, scratch: str) -> str:
    """Symlink the ligand beside the converted pocket; gen_feature wants one directory."""
    target = os.path.join(scratch, cid)
    os.makedirs(target, exist_ok=True)
    for name in (f"{cid}_ligand.mol2", f"{cid}_pocket.mol2"):
        source = os.path.join(complex_dir, name)
        link = os.path.join(target, name)
        if os.path.exists(source) and not os.path.exists(link):
            os.symlink(source, link)
    return target


def _build_one(job):
    """One complex, in its own scratch directory. Returns (id, data, error)."""
    cid, complex_dir, label, cut_dist, num_angle = job
    scratch = tempfile.mkdtemp(prefix=f"sign-{cid}-")
    try:
        stage(complex_dir, cid, scratch)
        if not ensure_pocket_mol2(complex_dir, cid, scratch):
            raise FileNotFoundError("no pocket in either format")

        feats = gen_feature(scratch, cid, Featurizer(save_molecule_codes=False))
        ligand = (feats["lig_fea"], feats["lig_co"], feats["lig_atoms"], feats["lig_eg"])
        pocket = (feats["pock_fea"], feats["pock_co"], feats["pock_atoms"], feats["pock_eg"])
        lig_size, coords, features, atoms = cons_lig_pock_graph_with_spatial_context(
            ligand, pocket, add_fea=3, theta=cut_dist,
            keep_pock=False, pocket_spatial=True)

        data = build_complex_graph(coords, features, atoms, lig_size, PAIR_IDS,
                                   cut_dist, num_angle, label=label)
        if data is None:
            raise ValueError("degenerate geometry; the original abandons these too")
        data.complex_id = cid
        return cid, data, None
    except Exception as e:
        return cid, None, f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SIGN graphs from complexes")
    parser.add_argument("--complexes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cut_dist", type=float, default=5.0)
    parser.add_argument("--num_angle", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel processes; 0 runs serially. The authors parallelise "
                             "this too — each complex is independent, and openbabel plus six "
                             "angle graphs is slow enough that serial costs hours.")
    parser.add_argument("--labels", default=None,
                        help="csv with complex_id,y_true; restricts to those ids and attaches "
                             "the label. Without it the graphs carry no y and can only embed.")
    args = parser.parse_args()

    labels = {}
    if args.labels:
        import csv
        with open(args.labels) as f:
            labels = {r["complex_id"]: float(r["y_true"]) for r in csv.DictReader(f)}

    available = {d for d in os.listdir(args.complexes)
                 if os.path.isdir(os.path.join(args.complexes, d))}
    ids = sorted(available & set(labels)) if labels else sorted(available)
    if args.limit:
        ids = ids[: args.limit]

    jobs = [(cid, os.path.join(args.complexes, cid), labels.get(cid),
             args.cut_dist, args.num_angle) for cid in ids]

    graphs, failed = [], []
    if args.workers and args.workers > 1:
        # Each complex is independent and the work is openbabel plus six angle graphs, so
        # this scales nearly linearly. A worker gets its own scratch directory: they stage
        # files under the complex id, and sharing one would let two workers collide.
        import multiprocessing as mp

        with mp.Pool(args.workers) as pool:
            for cid, data, error in tqdm(
                pool.imap_unordered(_build_one, jobs, chunksize=4),
                total=len(jobs), desc="Building graphs",
            ):
                if error:
                    failed.append(f"{cid}: {error}")
                else:
                    graphs.append(data)
    else:
        for job in tqdm(jobs, desc="Building graphs"):
            cid, data, error = _build_one(job)
            if error:
                failed.append(f"{cid}: {error}")
            else:
                graphs.append(data)

    if failed:
        print(f"\n{len(failed)} complexes failed:")
        for f in failed[:10]:
            print("  ", f)
    if not graphs:
        raise SystemExit("no graphs built")

    torch.save(graphs, args.out)
    atoms = sum(g.num_nodes for g in graphs)
    bonds = sum(g.num_bonds for g in graphs)
    print(f"\n{len(graphs)} graphs -> {args.out}")
    print(f"  {atoms} atoms, {bonds} bonds, {atoms / len(graphs):.0f} atoms per complex")


if __name__ == "__main__":
    main()
