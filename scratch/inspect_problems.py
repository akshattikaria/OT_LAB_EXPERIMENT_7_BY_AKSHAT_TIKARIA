import sys, os
sys.path.insert(0, os.path.abspath('.'))
import moo_utils

for i in range(1, 15):
    cls = getattr(moo_utils, f'ProblemP{i}', None)
    if cls:
        try:
            inst = cls()
            b = getattr(inst, 'bounds', None)
            b_str = f"shape={getattr(b, 'shape', None)}, min={getattr(b, 'min', lambda: None)()} max={getattr(b, 'max', lambda: None)()}" if b is not None else "None"
            print(f"P{i}: name={getattr(inst, 'name', '')} | n_var={getattr(inst, 'n_var', '')} | n_obj={getattr(inst, 'n_obj', '')} | n_constr={getattr(inst, 'n_constr', 0)} | ref_point={getattr(inst, 'ref_point', None)} | bounds={b_str}")
            if inst.__doc__:
                print("   doc:", inst.__doc__.strip().split('\n')[0])
        except Exception as e:
            print(f"P{i}: init error: {e}")
