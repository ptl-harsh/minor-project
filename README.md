# PixelPhrase Project

A simple image captioning system using a pre-trained CNN (InceptionV3) and LSTM decoder.


## Details:

**Encoder (CNN)**: Pre-trained InceptionV3, outputs 2048 features.

**Feature Processor**: Dropout 0.5, Dense 256 units (ReLU).

**Decoder (RNN)**: Embedding layer of size 256 (with zero masking), Dropout 0.5, LSTM with 256 units.

**Fusion**: Element-wise addition of image and sequence representations.

**Output Layers**: Dense 256 (ReLU), then Dense over vocabulary with softmax.

***


### Dataset

Flicker8K

The dataset consists of images and corresponding captions:
- **img_cap_dataset.zip**: Zip archive containing all images for captioning.
- **captions.txt**: CSV file with two columns (`image`,`caption`), providing multiple human-annotated captions per image.

Place these files in the `datasets/` directory before running the notebook.

### Setup
1. Clone or download the repo.
2. Install dependencies:
   ```bash
   pip install tensorflow pandas numpy matplotlib nltk tqdm
   ```

3. Download NLTK data:
   ```bash
   import nltk
   nltk.download('punkt')
   ```

### File Structure: 

```
script.ipynb              # Colab notebook with full pipeline
datasets/                 # Contains images and captions
  ├── img_cap_dataset.zip  # All training images
  └── captions.txt         # Corresponding captions for each image
features.pkl              # Cached image features
image_caption_model.h5    # Saved trained model
README.md                 # This file
```


### image_caption_gen.ipynb

- Mount Google Drive and unzip data

- Load and clean captions

- Extract InceptionV3 features

- Prepare sequences and train CNN-LSTM model

- Generate sample captions and compute BLEU scores

***

## Results

    BLEU-1: ~0.38

    BLEU-4: ~0.13


## Next Steps

  - Add attention mechanism

  - Use beam search for caption generation

  - Experiment with different CNN backbones








