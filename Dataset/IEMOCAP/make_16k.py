import librosa
import os
import soundfile as sf
from pathlib import Path
import sys

Path('Audio_16k').mkdir(exist_ok=True)
IEMOCAP_DIR = Path(sys.argv[1])
print ("Downsampling IEMOCAP to 16k")
for i in range(5):
    sess = i + 1
    current_dir = IEMOCAP_DIR / f"Session{sess}" / "sentences" / "wav"
    for full_audio_name in current_dir.rglob('*.wav'):
        audio, sr = librosa.load(str(full_audio_name), sr=None)
        audio_name = full_audio_name.name
        assert sr == 16000
        sf.write(os.path.join('Audio_16k', audio_name), audio, 16000)

