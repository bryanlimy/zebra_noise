from pathlib import Path

import zebranoise

filename = Path('visual_stimulus') / 'output.avi'
filename.parent.mkdir(parents=True, exist_ok=True)

zebranoise.zebra_noise(filename, xsize=1920, ysize=1080, tdur=60*5, fps=30, seed=0)