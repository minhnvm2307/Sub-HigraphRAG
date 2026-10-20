"""Step 5 — RAPL/model_inference.py run verbatim (its main(), same CLI), with D12 installed so the slim line graphs get
their exact x back at load (slim_x.py). Nothing else changes.
Run from RAPL/:   cd RAPL && python ../rapl_faithful/run_model_inference.py --num_layers 2 --dataset_name cwq ...
"""
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"     # as model_inference.py sets before torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), '..', 'rapl_faithful'))
assert os.path.basename(os.getcwd()) == 'RAPL'
import slim_x
slim_x.install('data/cwq/raw/gte-large-en-v1.5/test.pth', tag_build=False)
import model_inference
model_inference.main()
