import os
import numpy as np
import json
import random
from pathlib import Path
import re
import sys

IEMOCAP_DIR = Path(sys.argv[1])

print ('Generating metalabels...')
metalabel = {}
for i in range(5):
    sess = i + 1
    label_dir = IEMOCAP_DIR / f"Session{sess}" / "dialog" / "EmoEvaluation"
    for labelfile in label_dir.rglob('*.txt'):
        with open(labelfile, 'r') as f:
            for line in f.readlines():
                m = re.match(r".*(Ses.*)\t(.*)\t.*", line)
                if m:
                    name, label = m.groups()
                    metalabel[name+'.wav'] = label
with open(f'metalabel.json', 'w') as f:
    json.dump(metalabel, f, indent=4)
