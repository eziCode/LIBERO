import h5py
import json

custom = '/Users/ezraakresh/Documents/LIBERO/datasets/custom/shake_cup_demo.hdf5'
ref = '/Users/ezraakresh/Documents/LIBERO/datasets/libero_90/KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove_demo.hdf5'

def get_structure(path):
    info = {}
    with h5py.File(path, 'r') as f:
        info['root_attrs'] = list(f.attrs.keys())
        info['data_attrs'] = list(f['data'].attrs.keys())
        demo = sorted(f['data'].keys())[0]
        info['demo_attrs'] = list(f[f'data/{demo}'].attrs.keys())
        info['demo_keys'] = list(f[f'data/{demo}'].keys())
        info['obs_keys'] = list(f[f'data/{demo}/obs'].keys())
        info['actions_shape'] = f[f'data/{demo}/actions'].shape[1]
        info['states_shape'] = f[f'data/{demo}/states'].shape[1]
        info['robot_states_shape'] = f[f'data/{demo}/robot_states'].shape[1]
        # Check data attrs values (non-model_file)
        info['data_attr_values'] = {}
        for k in f['data'].attrs.keys():
            v = f['data'].attrs[k]
            if isinstance(v, (str, int, float)):
                info['data_attr_values'][k] = v
            elif hasattr(v, 'tolist'):
                info['data_attr_values'][k] = v.tolist()
    return info

c = get_structure(custom)
r = get_structure(ref)

print("=== ROOT ATTRS ===")
print(f"  Custom:    {c['root_attrs']}")
print(f"  Reference: {r['root_attrs']}")

print("\n=== DATA GROUP ATTRS ===")
print(f"  Custom:    {c['data_attrs']}")
print(f"  Reference: {r['data_attrs']}")
missing_in_custom = set(r['data_attrs']) - set(c['data_attrs'])
extra_in_custom   = set(c['data_attrs']) - set(r['data_attrs'])
print(f"  Missing in custom:  {missing_in_custom}")
print(f"  Extra in custom:    {extra_in_custom}")

print("\n=== DEMO ATTRS ===")
print(f"  Custom:    {c['demo_attrs']}")
print(f"  Reference: {r['demo_attrs']}")
missing = set(r['demo_attrs']) - set(c['demo_attrs'])
extra   = set(c['demo_attrs']) - set(r['demo_attrs'])
print(f"  Missing in custom: {missing}")
print(f"  Extra in custom:   {extra}")

print("\n=== DEMO KEYS (datasets) ===")
print(f"  Custom:    {sorted(c['demo_keys'])}")
print(f"  Reference: {sorted(r['demo_keys'])}")
missing = set(r['demo_keys']) - set(c['demo_keys'])
extra   = set(c['demo_keys']) - set(r['demo_keys'])
print(f"  Missing in custom: {missing}")
print(f"  Extra in custom:   {extra}")

print("\n=== OBS KEYS ===")
print(f"  Custom:    {sorted(c['obs_keys'])}")
print(f"  Reference: {sorted(r['obs_keys'])}")
missing = set(r['obs_keys']) - set(c['obs_keys'])
extra   = set(c['obs_keys']) - set(r['obs_keys'])
print(f"  Missing in custom: {missing}")
print(f"  Extra in custom:   {extra}")

print("\n=== DATASET SHAPES ===")
print(f"  actions dim:      custom={c['actions_shape']}   ref={r['actions_shape']}")
print(f"  states dim:       custom={c['states_shape']}    ref={r['states_shape']}  ← scene-specific, OK")
print(f"  robot_states dim: custom={c['robot_states_shape']}    ref={r['robot_states_shape']}")

print("\n=== DATA ATTR VALUES (non-binary) ===")
for k in sorted(set(c['data_attr_values']) | set(r['data_attr_values'])):
    cv = c['data_attr_values'].get(k, '<MISSING>')
    rv = r['data_attr_values'].get(k, '<MISSING>')
    match = '✓' if cv == rv else '✗'
    if k not in ('bddl_file_content', 'model_file', 'env_args'):
        print(f"  {match} {k}:")
        print(f"      custom: {cv}")
        print(f"      ref:    {rv}")
