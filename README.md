# Speaker Verification 

## Usage
1. Drop the video into `Original Video` card
2. Click `Upload Reference Audio Files` button. Here you can upload indiviudal `.wav` files or a single `.npy` file.
3. Click `Leggo` button

## Generate the .npy file from existing embeddings
```python
import numpy as np
embedding = [[192-elem array values], [192-elem array values], [192-elem array values]]
with open('/app/data/test.npy', 'wb') as f:
   np.save(f, np.array(embedding)
```

## Generating embeddings from audio file
```python
import torchaudio
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="/models/speechbrain")
signal, fs =torchaudio.load('/app/data/references/ref1.wav')
embeddings = classifier.encode_batch(signal)
embeddings.numpy()[0][0]
```