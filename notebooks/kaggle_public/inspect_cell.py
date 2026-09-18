import glob, os, torch

path = glob.glob('/kaggle/input/**/model_soup_S6_b1.pth', recursive=True)[0]
print(f'{os.path.basename(path)}: {os.path.getsize(path)/1e6:.1f} MB on disk')

ck = torch.load(path, map_location='cpu', weights_only=False)
meta = ck['meta']
print('fusion rule      :', meta['fusion'])
print('rebalancing steps:', meta['balance_steps'])
print('stream weights   :', {k: round(float(v), 4) for k, v in meta['weights'].items()})


def nbytes(x):
    """Stored bytes, walking the nested dicts an int8 member keeps."""
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(nbytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(nbytes(v) for v in x)
    return 0


print()
print(len(ck['models']), 'members:')
total = 0.0
for key, cfg in meta['members'].items():
    mb = nbytes(ck['models'][key]) / 1e6
    total += mb
    video = not str(cfg['modality']).lower().startswith(('skel', 'imu'))
    size = f"{cfg.get('size')}px" if video else '-'
    print(f"  {key:<34} {cfg['modality']:<12} {size:>6} "
          f"arch={str(cfg.get('arch', '-')):<14} int8={cfg.get('int8', False)} {mb:7.1f} MB")

det = nbytes(ck.get('detector', {})) / 1e6
print(f"  {'person detector (fp16)':<34} {'IR':<12} {'-':>6} "
      f"arch=ssdlite320    int8=False {det:7.1f} MB")
print()
print(f'members {total:.1f} MB + detector {det:.1f} MB = {total + det:.1f} MB of tensors')
