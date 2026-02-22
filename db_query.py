"""Quick N=5 cross-model alpha (excluding qwen3:32b) for paper."""
import sys
sys.path.insert(0, r'c:\Users\schol\Documents\Python Projects\tiktok_research')

from cross_model_validity import CrossModelValidator

v = CrossModelValidator(experiment_id=12)

# Monkey-patch to exclude qwen3:32b
_orig = v.get_final_labels_by_model
def filtered_labels(temperature=None):
    data, ctypes = _orig(temperature)
    for key in data:
        data[key].pop('qwen3:32b', None)
    return data, ctypes
v.get_final_labels_by_model = filtered_labels

_orig_f = v.get_final_floats_by_model
def filtered_floats(temperature=None):
    data = _orig_f(temperature)
    for key in data:
        data[key].pop('qwen3:32b', None)
    return data
v.get_final_floats_by_model = filtered_floats

# Run pooled only (temperature=None)
results = v.compute_cross_model_alpha(temperature=None)
print("\n=== N=5 Cross-Model Alpha (excluding qwen3:32b, pooled temps) ===")
for r in results:
    print(f"  {r['construct_name']}: alpha={r['cross_model_alpha']} [{r['alpha_ci_lower']}, {r['alpha_ci_upper']}] (N={r['num_models']} models, {r['num_multi_model_chunks']} chunks)")

v.close()
