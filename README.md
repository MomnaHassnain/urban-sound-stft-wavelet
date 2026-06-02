Here is a README you can copy into a `README.md` file for your code repository / S2 Code zip. It describes exactly what your script does and how to run it with UrbanSound8K.

***

# Fusing Wavelet and STFT Features with Bi-LSTM for Urban Sound Classification

This repository contains the author-generated code used in the paper:

> *Fusing wavelet and STFT features with multiple windows for urban sound classification*

The code implements wavelet-based feature extraction combined with Mel-filterbank and MFCC processing, sequence standardization, and Bi-LSTM training and evaluation on the UrbanSound8K dataset.

## 1. Overview

The pipeline has three main stages:

1. **Feature extraction (wavelet + Mel + MFCC):**
   - Load audio clips from UrbanSound8K.
   - Compute a multilevel discrete wavelet transform (Daubechies-4).
   - Apply log compression and a Mel filterbank to obtain Mel energies.
   - Compute MFCCs and their first and second temporal derivatives.
   - Standardize each clip to a fixed sequence length of 128 frames (padding or truncation).

2. **Dataset construction:**
   - Use the UrbanSound8K metadata (`UrbanSound8K.csv`) to iterate over all clips.
   - For each clip, extract a feature tensor of shape \(128 \times D\) and store it with the corresponding class label.
   - Convert the list of features and labels into NumPy arrays and split into train and test sets (80 / 20, `random_state=0`).

3. **Model training and evaluation (Bi-LSTM):**
   - Train bidirectional LSTM models with different numbers of units and activation functions.
   - Use early stopping and model checkpoints.
   - Evaluate on the test set and compute accuracy, weighted precision, weighted recall, weighted F1 score, and average specificity.
   - Save per-configuration results and the best model’s performance.

This code underpins the Gaussian-window + wavelet Bi-LSTM configuration described in the manuscript.

## 2. Requirements

This code was developed and tested in Google Colab with Python 3.10 and the following main packages:

- `numpy`
- `pandas`
- `librosa`
- `PyWavelets` (`pywt`)
- `scipy`
- `tqdm`
- `tensorflow` / `keras`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install the dependencies (for example):

```bash
pip install numpy pandas librosa PyWavelets scipy tqdm tensorflow scikit-learn matplotlib seaborn
```

## 3. Data

### UrbanSound8K

The code assumes that:

- The UrbanSound8K metadata file is available at:

```text
/content/UrbanSound8K.csv
```

- The audio files are stored in:

```text
/content/UrbanSound8K/
```

with the standard folder structure:

```text
UrbanSound8K/
  fold1/
    100032-3-0-0.wav
    ...
  fold2/
  ...
  fold10/
```

You can adjust these paths by changing:

```python
metadata_path = '/content/UrbanSound8K.csv'
audio_dataset_path = '/content/UrbanSound8K'
```

in the script or notebook.

UrbanSound8K is publicly available from the original authors:
https://urbansounddataset.weebly.com/urbansound8k.html

## 4. Feature extraction

### Wavelet transform

```python
def wavelet_transform(audio, wavelet='db4', level=None):
    """
    Compute the Discrete Wavelet Transform (DWT) of the input audio signal
    using PyWavelets and concatenate the absolute values of all coefficients.
    """
    coeffs = pywt.wavedec(audio, wavelet, level=level)
    wavelet_features = np.hstack([np.abs(c) for c in coeffs])
    return wavelet_features
```

### Wavelet + Mel + MFCC features

```python
def features_extractor_wavelet(file, wavelet='db4', n_mfcc=40, level=None):
    # 1. Load audio (librosa.load)
    # 2. Compute wavelet_transform(audio)
    # 3. Log compress the wavelet coefficients
    # 4. Apply a 40-band Mel filterbank
    # 5. Log compress Mel energies
    # 6. Apply DCT to obtain MFCCs (scipy.fftpack.dct)
    # 7. Compute delta and delta-delta coefficients (librosa.feature.delta)
    # 8. Stack MFCC, delta, and delta-delta to form a 3*n_mfcc feature vector per frame
    # 9. Return a (T x D) array (time frames x feature dimension)
```

### Sequence standardization

```python
def pad_or_truncate(features, target_length):
    """
    Pad or truncate a feature sequence to a fixed number of frames.
    If frames < target_length, pad with zeros.
    If frames > target_length, truncate to target_length.
    """
    ...
```

### Dataset extraction

The script loops over all rows in `UrbanSound8K.csv`, builds the full path to each file, extracts features, standardizes them to 128 frames, and stores them along with the class label:

```python
extracted_features = []

for index_num, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
    file_name = os.path.join(
        os.path.abspath(audio_dataset_path),
        f'fold{row["fold"]}',
        row["slice_file_name"]
    )
    final_class_labels = row["class"]
    features = extract_features_with_wavelet(file_name)

    if features is not None:
        features = pad_or_truncate(features, target_length=128)
        extracted_features.append([features, final_class_labels])

extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
```

Later cells convert `extracted_features_df['feature']` into a NumPy array `X` of shape `(N, 128, D)` and `y` into one-hot encoded labels, and perform an 80/20 train–test split using `train_test_split` from `scikit-learn`.

## 5. Model training (Bi-LSTM)

The training block performs a small grid search over:

- LSTM units: `{64, 128}`
- Activation functions: `{relu, tanh, leaky_relu}`

For each combination, it constructs a Bi-LSTM model:

```python
model = Sequential()

# First Bidirectional LSTM
model.add(Bidirectional(
    LSTM(units, input_shape=(X_train.shape [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/788325008/51e87721-1b39-4239-862d-f9142e4bfadb/image.jpg?AWSAccessKeyId=ASIA2F3EMEYEQZ36EQRU&Signature=RZMWU7dKbzw6EJTDksD1kqg5YEI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEFgaCXVzLWVhc3QtMSJHMEUCIQCsk7puHhky8EaQBKu0W266hV6j7jXBaFKDHoI8FDNcBgIgAr9yq3QREzuobwHcph6RaoDMYu2%2BAQxFjPTwUCOKdScq8wQIIRABGgw2OTk3NTMzMDk3MDUiDBxm63MhFp7XxRI2DCrQBG5LHHbo959lGaXvM5Jqmj3KMDXCwvKuWbilKk0NGqdV7V%2FV4STvjyUfJTJMqBgTQz477G5Q2AoOpU4c7GMy%2FmRU2ZP1CSjuLJwH2KCR3aAWY7PzxLVEb8Xguh0ET1IuKUkrn7kFgfv0Xz%2Fl5nrGumAOWRwFmRDTRWHI5oa8PbGHoxkBmrWFckloGmS5uMzW1%2BiigbTr61q%2BM798JiUPXre3iNP8TxlK%2B1dhz0gV8FTsoBOm6V6WI6QRK4cLP%2FnHPwxP%2Bw1z%2FqsM0BACgOfuOaDfu7XWarg9JIasU%2FsAHnCo4oLcCsjovzgGvw91%2Fpkn0scjz3iqVdva3HHktvRSJEdeq2mrdOPrgcOZeBVfxtDuwFK40fygIZx6jiJPKF%2FcLe6MCtYgb1gn7tURAoEjWwGaFK9IoDkpuKAWtOOh%2BVKFqjXI%2FCpgafQg%2Fca20cIkwkd7vchKcEtZyM9Alc2RXMwDBfqGOkhR8adSlVvEoHSElBuXVwycCVWycNUU3Q0097Vad3uziQ8RFJ8aycjfsph50ZlY7cXGWviCM7c7KJOGHK2esvBkpFlKAbnrhd4CJ0W092V5A1cYiW9AMCW2vVgHPdgBZIk5%2BnsvKqzRTzzI0d%2BZzbIwqH%2FsFCeuob9QO2oqWRMse5upbgSD6Bgg2xHGis%2B1fuZqFOVOXs%2BqTH3Qlt0qlpGWrJl1FeViaZpCldw1VNJUh4UvAJWnxobVnQJVBfyaJbLZ9t0uKX5XS5gmqVXKLiWyvqB4%2B6fAAfEpc9QKep7b2ar%2FrYkzNYc%2F6kowjpf60AY6mAHoKD4QukSyL0x9Q%2FnflRrLoPQt0RTR2sX9CRWwmmpXgtJoHfHh2pbYm89Uk%2BYxqL0XGRFgDlsOO%2BhO%2BryshRX0Vw3WyhdFB%2FCEBL1ZE5XDcwKQ0HqdACdpqGD62p%2B8np0ZHkRtAjv8pg9irb8rMSn09OQ9qN8Rl%2Bm8dHf1hfXkAJxIM2pTf4HidRXUVvnIBsofcwBmtrvnnA%3D%3D&Expires=1780389263), X_train.shape [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/788325008/538e7850-e2b0-41e0-943b-baa4065fd6ba/image-2.jpg?AWSAccessKeyId=ASIA2F3EMEYEQZ36EQRU&Signature=G5c48xL78IufxlWXwW95P9oYiVk%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEFgaCXVzLWVhc3QtMSJHMEUCIQCsk7puHhky8EaQBKu0W266hV6j7jXBaFKDHoI8FDNcBgIgAr9yq3QREzuobwHcph6RaoDMYu2%2BAQxFjPTwUCOKdScq8wQIIRABGgw2OTk3NTMzMDk3MDUiDBxm63MhFp7XxRI2DCrQBG5LHHbo959lGaXvM5Jqmj3KMDXCwvKuWbilKk0NGqdV7V%2FV4STvjyUfJTJMqBgTQz477G5Q2AoOpU4c7GMy%2FmRU2ZP1CSjuLJwH2KCR3aAWY7PzxLVEb8Xguh0ET1IuKUkrn7kFgfv0Xz%2Fl5nrGumAOWRwFmRDTRWHI5oa8PbGHoxkBmrWFckloGmS5uMzW1%2BiigbTr61q%2BM798JiUPXre3iNP8TxlK%2B1dhz0gV8FTsoBOm6V6WI6QRK4cLP%2FnHPwxP%2Bw1z%2FqsM0BACgOfuOaDfu7XWarg9JIasU%2FsAHnCo4oLcCsjovzgGvw91%2Fpkn0scjz3iqVdva3HHktvRSJEdeq2mrdOPrgcOZeBVfxtDuwFK40fygIZx6jiJPKF%2FcLe6MCtYgb1gn7tURAoEjWwGaFK9IoDkpuKAWtOOh%2BVKFqjXI%2FCpgafQg%2Fca20cIkwkd7vchKcEtZyM9Alc2RXMwDBfqGOkhR8adSlVvEoHSElBuXVwycCVWycNUU3Q0097Vad3uziQ8RFJ8aycjfsph50ZlY7cXGWviCM7c7KJOGHK2esvBkpFlKAbnrhd4CJ0W092V5A1cYiW9AMCW2vVgHPdgBZIk5%2BnsvKqzRTzzI0d%2BZzbIwqH%2FsFCeuob9QO2oqWRMse5upbgSD6Bgg2xHGis%2B1fuZqFOVOXs%2BqTH3Qlt0qlpGWrJl1FeViaZpCldw1VNJUh4UvAJWnxobVnQJVBfyaJbLZ9t0uKX5XS5gmqVXKLiWyvqB4%2B6fAAfEpc9QKep7b2ar%2FrYkzNYc%2F6kowjpf60AY6mAHoKD4QukSyL0x9Q%2FnflRrLoPQt0RTR2sX9CRWwmmpXgtJoHfHh2pbYm89Uk%2BYxqL0XGRFgDlsOO%2BhO%2BryshRX0Vw3WyhdFB%2FCEBL1ZE5XDcwKQ0HqdACdpqGD62p%2B8np0ZHkRtAjv8pg9irb8rMSn09OQ9qN8Rl%2Bm8dHf1hfXkAJxIM2pTf4HidRXUVvnIBsofcwBmtrvnnA%3D%3D&Expires=1780389263)), return_sequences=True)
))
# Activation (ReLU, tanh, or LeakyReLU)
...
model.add(Dropout(0.5))

# Second Bidirectional LSTM
model.add(Bidirectional(LSTM(units, return_sequences=False)))
...
model.add(Dropout(0.5))

# Dense layer + activation
model.add(Dense(64))
...
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

Training uses early stopping and model checkpoints:

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpointer = ModelCheckpoint(
    filepath=f'models/audio_classification_bilstm_{units}_{activation}.keras',
    verbose=1,
    save_best_only=True
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, checkpointer],
    verbose=1
)
```

After training, the code computes:

- Confusion matrix (`confusion_matrix`)
- Weighted precision, recall, F1 (`precision_score`, `recall_score`, `f1_score`, `average='weighted'`)
- Average specificity (mean true negative rate across classes)

All results are stored in a `pandas` DataFrame and used to generate the tables reported in the paper.

## 6. Reproducing the published results

To reproduce the wavelet + Bi-LSTM configuration:

1. **Download UrbanSound8K** and place `UrbanSound8K.csv` and the audio folds as described above.
2. **Install dependencies** as in Section 2.
3. **Run the notebook or script**:
   - Run the cells that:
     - Load metadata.
     - Extract wavelet+MFCC features and build `extracted_features_df`.
     - Build `X` and `y` arrays and perform the 80/20 train–test split.
     - Train the Bi-LSTM models with the hyperparameter grid.
4. **Collect metrics** from the printed `df_results` table and confusion matrix:
   - These metrics correspond to the values reported for the Gaussian + wavelet Bi-LSTM configuration in the manuscript (Table 4 and the confusion matrix figure).

## 7. License and reuse

This code is provided to support reproducibility of the results reported in the accompanying manuscript. Users may adapt and reuse the code for research purposes; please cite the paper if you use this implementation in academic work.

***

If you want, I can customize this README further (e.g., add exact file names like `final_thesis_project.ipynb`, or separate sections for each window configuration) once you decide how you’ll organize the repo/files.
