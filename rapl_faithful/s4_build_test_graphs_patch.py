"""D11 — imported by s4_build_test_graphs.py and verify_slim.py (RAPL files untouched)."""
from src.dataset.PyG_dataset_disk import KGDataset

# Runtime fix of a bug in the released KGDataset._paths_to_triplets (RAPL file untouched): it splits a path string on
# "->" and .strip()s each segment, so a Freebase name with leading/trailing spaces (e.g. '  Adventures by Disney - ...')
# is never found and the build stops with KeyError (51 of 3,531 CWQ test samples). The released
# PostProcessedDataset._paths_to_triplets already tolerates the same case (prints, falls back to a default id). Here the
# original function is called unchanged, only with whitespace-stripped entity names, so those paths get their real ids.
_orig_paths_to_triplets = KGDataset._paths_to_triplets
def _paths_to_triplets_stripped(self, id2entities, id2relations, paths):
    return _orig_paths_to_triplets(self, {k: v.strip() for k, v in id2entities.items()}, id2relations, paths)
KGDataset._paths_to_triplets = _paths_to_triplets_stripped
