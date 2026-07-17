# ml_manual_cpp

Neural network dibangun dari nol (tanpa PyTorch/TensorFlow/Eigen) di C++, dengan binding Python via pybind11. Mendukung dua jenis model:

1. **MLP** (Multi-Layer Perceptron) — Dense layer, BatchNorm, beberapa fungsi aktivasi & loss.
2. **Transformer** (gaya GPT-2) — token embedding + positional encoding, multi-head self-attention dengan causal masking, feed-forward GELU, layer norm, residual, weight tying, plus tokenizer BPE byte-level sendiri.

Target lingkungan: **Linux, Termux (Android), dan Raspberry Pi** — semua keputusan desain (tipe `float`, cache-blocking matmul, opsi OpenBLAS, weight tying, rekomputasi attention saat backward) diarahkan untuk hemat memori dan jalan di CPU ARM dengan sumber daya terbatas.

> **Status:** kode ini ditulis lengkap tapi **belum pernah dikompilasi/dijalankan**. Kemungkinan ada bug yang baru ketahuan saat build pertama kali — lihat bagian [Troubleshooting](#troubleshooting).

---

## Struktur proyek

```
ml_manual_cpp/
├── include/            # header (.h) — satu per komponen
├── src/                # implementasi (.cpp)
├── bindings/
│   └── pybind_module.cpp   # jembatan C++ <-> Python
├── CMakeLists.txt
├── sample_data.py       # data & korpus contoh
├── training.py          # skrip training (MLP + SequenceModel + tokenizer)
├── demo.py               # skrip demo: load checkpoint, prediksi/generate teks
└── README.md
```

### Komponen inti (`include/` + `src/`)

| Komponen | Fungsi |
|---|---|
| `matrix_ops` | Kelas `Matrix` 2D, operasi dasar (matmul blocked/OpenBLAS, transpose, dll) |
| `tensor3d` | Kelas `Tensor3D` (batch × seq_len × features) untuk data sekuensial |
| `activations` | ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, Linear |
| `losses` | MSE, BinaryCrossEntropy, SoftmaxCrossEntropy |
| `layer_base` | Interface abstrak untuk layer MLP (`DenseLayer`, `BatchNormLayer`) |
| `dense_layer` | Fully-connected layer |
| `batch_norm_layer` | Batch normalization |
| `neural_network` | Menyusun `DenseLayer`/`BatchNormLayer` jadi MLP, forward/backward/update |
| `trainer` | Training loop untuk `NeuralNetwork` (mini-batch, shuffle, train/val split) |
| `embedding_layer` | Token embedding + positional encoding sinusoidal |
| `attention_layer` | Multi-head self-attention (causal mask opsional) |
| `layer_norm_tensor` | Layer normalization untuk `Tensor3D` |
| `feed_forward_block` | Feed-forward position-wise (Linear → GELU → Linear) |
| `transformer_block` | Satu blok Transformer pre-norm (LN→Attention→residual→LN→FFN→residual) |
| `sequence_model` | Menyusun `EmbeddingLayer` + N `TransformerBlock` jadi model bahasa (weight tying) |
| `sequence_trainer` | Training loop untuk `SequenceModel` (potong token stream jadi batch) |
| `bpe_tokenizer` | Tokenizer BPE byte-level (train, encode, decode, checkpoint) |

Semua komponen yang punya parameter belajar mendukung **checkpoint save/load** (format biner sendiri, lihat `save()`/`load()` di masing-masing kelas).

---

## Requirements

- Compiler C++17 (g++/clang)
- CMake ≥ 3.15
- Python 3 + `numpy`
- `pybind11` (`pip install pybind11 --break-system-packages`)
- Opsional: OpenBLAS (`pkg install openblas` di Termux, atau `apt install libopenblas-dev` di Raspberry Pi OS) untuk matmul lebih cepat

---

## Build

```bash
pip install pybind11 --break-system-packages   # kalau belum ada

cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OPENBLAS=ON
cmake --build build -j$(nproc)
```

Kalau OpenBLAS tidak dipasang atau `-DUSE_OPENBLAS=ON` dihilangkan, build otomatis fallback ke matmul cache-blocked manual (tidak perlu ubah kode).

Hasil build: `build/ml_manual_cpp*.so` (modul Python).

---

## Menjalankan

### 1. Training

```bash
python training.py
```

Ini akan:
- Melatih MLP klasifikasi biner sintetis (20 epoch), simpan `mlp_checkpoint.bin`
- Melatih tokenizer BPE dari korpus contoh, simpan `tokenizer_checkpoint.bin`
- Melatih `SequenceModel` (mini transformer, 5 epoch) di atas korpus yang sudah di-tokenize, simpan `sequence_checkpoint.bin`

Progress loss per epoch dicetak ke console.

### 2. Demo

```bash
python demo.py
```

Ini **memuat checkpoint** hasil `training.py` (tidak training ulang) dan:
- Menjalankan beberapa prediksi MLP
- Melakukan text generation autoregresif dari `SequenceModel` (sampling dengan temperature), menampilkan prompt vs hasil generate

> `demo.py` butuh ketiga file `.bin` di atas sudah ada. Kalau belum, dia akan cetak pesan minta jalankan `training.py` dulu.

---

## Catatan desain & keterbatasan yang perlu diketahui

- **Model demo sengaja kecil** (`embed_dim=32`, 2 layer, korpus 1 kalimat diulang) — untuk verifikasi pipeline, bukan kualitas hasil. Hasil generate paling-paling meniru pola kalimat training, bukan "reasoning".
- **Tidak ada KV-cache** saat generate — tiap token baru, seluruh context di-forward ulang dari awal. Lambat untuk generate panjang; optimasi lanjutan kalau dibutuhkan.
- **Checkpoint tidak ada versioning/validasi arsitektur** — load checkpoint dari konfigurasi model yang beda bisa gagal diam-diam atau crash. Jangan campur checkpoint antar-eksperimen dengan arsitektur berbeda.
- **AttentionLayer merekomputasi attention weights saat backward** (bukan cache) — hemat memori, sedikit lebih lambat.
- **Weight tying** aktif di `SequenceModel` — proyeksi output ke vocab berbagi bobot dengan embedding table (mengurangi jumlah parameter).
- **BPE tokenizer** dilatih ulang tiap kali `training.py` dijalankan (tidak ada tokenizer "default" bawaan) — pastikan `tokenizer_checkpoint.bin` dari sesi training yang sama dipakai saat `demo.py`.

---

## Troubleshooting

Karena kode ini belum pernah dikompilasi:

- **Error kompilasi**: cek pesan error compiler baris-per-baris — kemungkinan typo signature atau include yang kurang antar header/cpp yang ditulis terpisah selama pengembangan.
- **`pybind11` tidak ketemu saat CMake**: pastikan `pip show pybind11` menunjukkan lokasi terpasang, atau set manual `-Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)`.
- **Loss `NaN` atau tidak turun**: kemungkinan bug di jalur backward yang paling kompleks (attention softmax-jacobian, atau gradien weight-tying di `SequenceModel`). Cek dengan menurunkan `learning_rate` dulu untuk pastikan bukan sekadar meledak (exploding gradient) sebelum curiga ada bug logika.
- **Angka aneh setelah load checkpoint**: pastikan konfigurasi model (`embed_dim`, `num_heads`, dll) saat load konsisten dengan saat training — checkpoint tidak memvalidasi ini secara otomatis.

Kalau menemukan bug saat build/run, catat pesan error atau perilaku anehnya (loss, output) untuk dibahas lebih lanjut.
