from pathlib import Path
import re
import yaml


def find_checkpoint(cfg):
    cp_iter = cfg['train']['checkpoint_every']
    steps = cfg['train']['steps']
    n_cp, fname_cp = 0, None
    for n_iter in range(cp_iter, steps + cp_iter, cp_iter):
        fname = cfg['train']['checkpoint_name'].format(n_iter=n_iter//cp_iter)
        if Path(fname).exists():
            n_cp, fname_cp = n_iter, fname
    return n_cp, fname_cp


def replace_e_float(d):
    p = re.compile(r'^-?\d+(\.\d+)?e-?\d+$')
    for name, val in d.items():
        if type(val) == dict:
            replace_e_float(val)
        elif type(val) == str and p.match(val):
            d[name] = float(val)


def load_cfg(name, prefix='.'):
    with open(f'{prefix}/config/{name}.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        replace_e_float(cfg)
        return cfg
