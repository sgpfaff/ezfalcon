"""Runner script for EX2 that saves figures to _figures/EX2/."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_fig_dir = '_figures/EX2'
_fig_count = [0]

_orig_show = plt.show
def _save_show():
    fig = plt.gcf()
    _fig_count[0] += 1
    path = f'{_fig_dir}/fig_{_fig_count[0]:02d}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close(fig)
plt.show = _save_show

# Stub out IPython display (not needed for figure generation)
import builtins
class _NullHTML:
    def __init__(self, *a, **kw): pass
    def __repr__(self): return ''
try:
    import IPython.display as _ipyd
    _ipyd.display = lambda *a, **kw: None
except ImportError:
    pass

# Run the example
exec(open('EX2_Disk.py').read(), {'__name__': '__main__'})
print("EX2 complete.")
