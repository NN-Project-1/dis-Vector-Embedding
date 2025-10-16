# DIS-VECTOR: AN EFFECTIVE APPROACH FOR CONTROLLABLE ZERO-SHOT VOICE CONVERSION AND CLONING IN LOW-RESOURCE LANGUAGES 🎤✨

Welcome to the DIS-Vector project! This repository presents an advanced low-resource, zero-shot voice conversion and cloning model that leverages disentangled embeddings, clustering techniques, and language-based similarity matching to achieve highly natural and controllable voice synthesis.

The **DIS-Vector** model introduces a novel approach to voice conversion by disentangling speech components content, pitch, rhythm, and timbre into separate embedding spaces, enabling fine-grained control over voice synthesis. Unlike traditional voice conversion models, DIS-Vector is capable of zero-shot voice cloning, meaning it can synthesize voices from unseen speakers and languages without requiring large-scale speaker-specific training data.

We have Approach 1, which is the base version of DIS-Vector. The details are provided [here](https://github.com/NN-Project-1/dis-Vector-Embedding/blob/main/readme1.md).


---

## 📚 Table of Contents
1. [Overview](#1-overview)  
2. [Dis-Vector Model Details](#2-dis-vector-model-details)  
3. [VITS-TTS Integration](#3-vits-tts-integration)  
4. [GPT-TTS Integration](#4-gpt-tts-integration)  
5. [Length Analysis](#5-length-analysis)  
6. [Speech Component Representation](#6-speech-component-representation)  
7. [Types of Loss Functions](#7-types-of-loss-functions)  
8. [Evaluation](#8-evaluation)  
   1. [Test Setup](#81-test-setup)  
   2. [Distance Measurement](#82-distance-measurement)  
   3. [Ground Truth vs. TTS Output Similarity](#83-ground-truth-vs-tts-output-similarity)  
9. [Clustering & Language Matching](#9-clustering--language-matching)  
   1. [K-Means Clustering for Speaker Embeddings](#91-k-means-clustering-for-speaker-embeddings)  
   2. [Language-Based Similarity Matching](#92-language-based-similarity-matching)  
   3. [Closest Language Matching During Inference](#93-closest-language-matching-during-inference)  

---

## 1. Overview
The Dis-Vector model represents a significant advancement in voice conversion and synthesis by employing disentangled embeddings and clustering methodologies to precisely capture and transfer speaker characteristics. It introduces a novel **language-based similarity approach** and **K-Means clustering** for efficient speaker retrieval and closest language matching during inference.

### Features
- **Disentangled Embeddings**: Separate encoders for content, pitch, rhythm, and timbre.  
- **Zero-Shot Capabilities**: Effective voice cloning and conversion across different languages.  
- **High-Quality Synthesis**: Enhanced accuracy and flexibility in voice cloning.  
- **K-Means Clustering**: Optimized speaker embedding retrieval for inference.  
- **Language-Based Similarity Matching**: Determines the closest match from the embedding database to improve synthesis quality.  

### Sample Output Demo
Explore our [live demo here](https://nn-project-1.github.io/dis-vector_web/) showcasing the capabilities of the Dis-Vector model! Users can listen to synthesized audio samples highlighting accurate replication and transformation of speaker characteristics.

---

## 2. Dis-Vector Model Details
The DIS-Vector model is a multi-encoder disentanglement-based speech representation framework developed for expressive, cross-lingual, and zero-shot voice conversion. The design goal is to separate key aspects of human speech content, pitch, rhythm, and timbre into distinct, independently controllable latent embeddings. This architecture enables precise manipulation of each speech attribute while maintaining perceptual coherence in synthesis, allowing natural-sounding speaker conversion and expressive style transfer without retraining.

<p align="center">
  <img src="architecture/DIS-Vector-V2.png" alt="Dis-Vector Architecture"  width="400">
</p>

DIS-Vector follows a **parallel encoder structure** in which each encoder extracts a distinct feature domain from synchronized mel-spectrogram frames represented as `[B, T, F]`, where `B` is batch size, `T` denotes sequence length, and `F` represents mel-frequency bins. The four encoders—content, pitch, rhythm, and timbre—operate concurrently to generate their respective latent representations. These outputs are concatenated into a single **512-dimensional embedding vector**, forming a unified disentangled speech representation suitable for downstream decoding and synthesis.


### Encoder Specifications  

#### Content Encoder  

The **Content Encoder** captures linguistic and phonetic structures that define the spoken message while remaining independent of speaker characteristics. It utilizes a **hybrid CNN–LSTM architecture**. The convolutional layers model local spectral correlations and phonetic transitions from short-term mel segments, whereas the LSTM layers preserve long-term linguistic continuity across frames. This combination ensures that both spectral detail and sequential context are effectively represented.

The convolutional stack contains several 1D convolutional layers with kernel sizes between 3 and 5, followed by ReLU activation and layer normalization. The LSTM network, with a hidden dimension of 256, processes these convolutional outputs sequentially to generate a **256-dimensional content latent vector** (`z_c`). This latent vector encodes the phoneme-level linguistic structure necessary for accurate speech reconstruction and style-independent synthesis.

---

#### Pitch Encoder  

The **Pitch Encoder** focuses on modeling the **fundamental frequency (F₀)** contour and tonal variations that determine intonation and expressiveness. It employs a **CNN–LSTM design** similar to the content encoder. The convolutional layers extract frequency periodicity and harmonic structure from mel-spectrogram inputs, while the LSTM captures frame-to-frame pitch progression and smooth tonal movement.

The F₀ contour is first extracted from the waveform using a pitch estimation algorithm such as **PyWorld** or **YAAPT**, followed by log-normalization and alignment with mel frames. The CNN captures harmonic energy variations, and the LSTM models dynamic changes over time. The resulting **128-dimensional pitch latent vector** (`z_p`) represents tonal shape, direction, and smoothness while suppressing speaker-specific spectral effects. This latent serves as a precise prosodic descriptor, enabling tonal transfer across different speakers without losing natural pitch consistency.

---

#### Rhythm Encoder  

The **Rhythm Encoder** models the temporal structure of speech at the **frame level**, focusing on the timing, duration, and energy variations that define rhythmic patterns. The input mel-spectrogram sequence is processed using a **CNN–LSTM architecture** designed to learn both local and sequential temporal cues. The convolutional layers extract short-range frame-level amplitude modulations and energy transitions corresponding to syllable boundaries and intra-word timing. Each convolutional operation is followed by ReLU activation and layer normalization to stabilize feature scaling across frames.

The convolutional output sequence is passed to the LSTM network, which operates over the same frame-aligned time axis. The LSTM captures extended dependencies between consecutive frames, modeling duration patterns, inter-phoneme gaps, and rhythmic continuity across the entire utterance. During training, frame-level duration labels are obtained from **forced alignment outputs** (e.g., Montreal Forced Aligner), where each mel frame is explicitly aligned to its corresponding phoneme boundary. These aligned frame-level mappings enable the LSTM to learn the precise temporal distribution of speech frames.

The final hidden state sequence from the LSTM is mean-pooled across frames to obtain a **64-dimensional rhythm latent vector (`z_r`)**. This vector encodes detailed timing features, including speech rate, stress placement, and pause distribution. During synthesis, `z_r` determines frame-level timing control, allowing modification of utterance pacing and duration patterns while preserving the linguistic and pitch characteristics encoded in other latent representations.

---

#### Timbre Encoder  

The **Timbre Encoder** processes mel-spectrogram sequences at the frame level to extract speaker-specific spectral features that define vocal identity, resonance, and spectral coloration. The encoder is implemented using a **Transformer-based architecture** optimized for long-range dependency modeling across both frequency and temporal dimensions. Each Transformer block consists of **multi-head self-attention (MHSA)**, **position-wise feed-forward layers**, **layer normalization**, and **residual connections**. The MHSA mechanism computes attention weights across all time–frequency positions, allowing each frame embedding to integrate spectral context from the entire utterance.

During processing, mel-spectrogram frames are linearly projected into a fixed embedding space and combined with positional encodings that preserve frame order. The attention module analyzes correlations between frequency bands, capturing resonance and spectral envelope patterns that distinguish one speaker from another. Feed-forward sublayers apply non-linear transformations to refine feature separability, while residual normalization stabilizes gradient flow during training. The output of the final Transformer layer is mean-pooled across frames to generate a fixed-length **64-dimensional timbre latent vector (`z_t`)**. This latent vector represents the static and dynamic timbral attributes that uniquely characterize the speaker’s voice, including vocal tract shape, formant structure, and spectral slope behavior.

After all encoders complete their feature extraction, the resulting latent vectors are concatenated to form a unified **512-dimensional composite representation** defined as:

z_DIS = [z_c; z_p; z_r; z_t]

---

### Decoder and Reconstruction  

The **Decoder** performs frame-level mel-spectrogram reconstruction from the unified 512-dimensional **DIS-vector** `[z_c; z_p; z_r; z_t]`. The input vector sequence is first linearly projected and temporally expanded to match the original frame resolution. This projection initializes the decoder input sequence, where each frame embedding represents the fused acoustic state derived from content, pitch, rhythm, and timbre components.

The decoder architecture is implemented using **stacked Transformer-based upsampling blocks** followed by **convolutional refinement layers**. Each Transformer block consists of multi-head self-attention (MHSA), feed-forward sublayers, and residual normalization. The MHSA mechanism computes attention weights over all frame positions, allowing each reconstructed frame to access long-range contextual information across the entire utterance. This operation models inter-frame dependencies in both temporal and spectral dimensions, ensuring that transitions between phonemes and prosodic segments remain continuous.

Following the attention layers, temporal upsampling is performed through learned linear interpolation modules that double the frame resolution at each stage. This process restores the original temporal resolution of the mel-spectrogram without loss of synchronization. After upsampling, **1D convolutional refinement layers** with kernel size 5 are applied to each frame sequence to enhance local spectral resolution and correct frame-level distortions. These convolutional layers reconstruct harmonic detail and formant structure from the encoded latent features.

The decoder output is a sequence of mel-spectrogram frames `Ŝ ∈ ℝ^{T×F}`, where each frame corresponds directly to its original temporal index. The reconstructed spectrogram is then converted into the final waveform using a pretrained **HiFi-GAN neural vocoder** operating in 22.05 kHz sampling mode. The vocoder synthesizes waveform samples directly from the decoder’s mel output, preserving amplitude envelope and spectral envelope consistency.  

This reconstruction flow maintains strict frame-level alignment between input latents and output acoustics, ensuring that content, pitch, rhythm, and timbre information are coherently mapped to the final audio representation without cross-domain interference.


---

### Design Rationale  

The architecture of DIS-Vector follows the **principle of structured disentanglement**. The CNN–LSTM encoders (for content, pitch, and rhythm) are optimized for local spectral and sequential modeling, providing stability and temporal precision. The Transformer-based timbre encoder introduces global spectral attention, enabling accurate representation of speaker-specific qualities that convolutional networks typically overlook.  

This hybrid integration achieves a balance between local detail and global dependency modeling, producing high-fidelity, controllable, and speaker-adaptive synthesis. The resulting DIS-vector representation supports flexible manipulation across speech dimensions, allowing expressive and natural voice conversion across languages and speaker styles.

---

## 3. VITS-TTS Integration

<p align="center">
  <img src="architecture/d-v.png" alt="DIS-Vector Architecture" width="400">
</p>

Integrating VITS with DIS-Vector enhances its capabilities by leveraging disentangled embeddings of speech components (content, pitch, rhythm, and timbre). DIS-Vector provides fine-grained control over these components, enabling high-quality voice conversion and zero-shot voice cloning. This integration empowers VITS to generate speech in new voices, adapting to different speakers and languages without the need for speaker-specific training data, offering more flexibility and realism in synthetic speech generation.

---

## 4. GPT-TTS Integration

<p align="center">
  <img src="architecture/gpt.png" alt="DIS-Vector Architecture" width="400">
</p>


Our GPT-based architecture for zero-shot voice cloning processes input text through a byte-pair encoding (BPE) tokenizer to generate subword tokens, which are then embedded and passed through a stack of autoregressive GPT-style Transformer blocks. These blocks are trained to predict discrete latent codes from a Vector-Quantized Variational Autoencoder (VQ-VAE), which encodes ground-truth acoustic features into discrete indices that serve as the training targets. Crucially, the pre-computed 512-dimensional DIS-Vector speaker embeddings, which disentangle content, pitch, rhythm, and timbre, are projected to the model's dimension and injected into the network through a dual-conditioning mechanism: they are concatenated with the encoder's input tokens and also used for Feature-wise Linear Modulation (FiLM) conditioning within the Transformer blocks' activations. This ensures the model's predictions are conditioned on both the linguistic content and the precise vocal characteristics of the target speaker. The output predicted VQ-VAE codes are subsequently decoded into frame-level acoustic representations, which, together with the DIS-Vector embeddings, condition a HiFi-GAN vocoder to synthesize the final waveform directly from the latent features, enabling high-fidelity, zero-shot voice cloning.

---

## 5. Length Analysis
Ablation experiments were conducted with embedding lengths: 256, 512, 768, 1024.  

- 256D: Noticeable information loss, especially in timbre variations.  
- 768D & 1024D: Redundant, slower convergence without significant quality improvement.  
- 512D: Best balance; sufficient capacity to disentangle content, pitch, rhythm, timbre while maintaining training stability and efficiency.  

Final architecture adopts **512D embeddings** as an experimentally validated optimal trade-off.

---

## 6. Speech Component Representation

A speech signal \( s(t) \) is decomposed into four components:

<p align="center">
  <img src="architecture/combine.png" alt="DIS-Vector Architecture" width="200">
</p>

- **C(t) (Content):** Linguistic information  
- **P(t) (Pitch):** Fundamental frequency \( F_0 \)  
- **R(t) (Rhythm):** Duration and timing patterns  
- **T(t) (Timbre):** Speaker identity characteristics  

---

## 7. Types of Loss Functions

- **Mean Squared Error (MSE) Loss**: Minimizes difference between predicted and actual continuous components.  

<p align="center">
  <img src="architecture/mse.png" alt="DIS-Vector Architecture" width="200">
</p>

- **Kullback-Leibler (KL) Divergence Loss**: Measures difference between distributions, ensures embedding alignment.  

<p align="center">
  <img src="architecture/KL.png" alt="DIS-Vector Architecture" width="200">
</p>

- **Disentanglement Loss**: Ensures embeddings (content, pitch, rhythm, timbre) remain distinct.  

<p align="center">
  <img src="architecture/loss.png" alt="DIS-Vector Architecture" width="200">
</p>

Where:  
- **L_content**: Linguistic consistency  
- **L_pitch**: Preserves F0  
- **L_rhythm**: Maintains timing  
- **L_timbre**: Preserves speaker identity  

---

## 8. Evaluation

### 8.1 Test Setup

<p align="center">
  <img src="architecture/plot_dif.png" alt="DIS-Vector Architecture" width="300">
</p>

- **Pitch Testing**: Pitch Error Rate (PER)  
- **Rhythm Testing**: Rhythm Error Rate (RER)  
- **Timbre Testing**: Timbre Error Rate (TER)  
- **Content Testing**: Content Preservation Rate (CPR)  

### 8.2 Distance Measurement
- **Cosine Similarity**: Evaluates feature transfer and voice synthesis  

### 8.3 Ground Truth vs. TTS Output Similarity
- Measures similarity in pitch, rhythm, timbre, and content  

---

## 9. Clustering & Language Matching

### 9.1 K-Means Clustering for Speaker Embeddings
Dis-Vector utilizes a **language-annotated speaker embedding database**, where each speaker is mapped to a distinct feature representation based on their **timbre and prosody characteristics**. To enable efficient **cross-speaker and cross-language voice conversion**, we apply **K-Means clustering** on these high-dimensional embeddings. This clustering process helps to:

- **Group speakers** based on intrinsic vocal attributes such as pitch, intonation, and articulation patterns.
- **Enable zero-shot voice conversion** by leveraging cluster-based matching, even for unseen speakers.
- **Assign cluster centroids as representative embeddings**, allowing the system to select the closest match for synthesis.
- **Improve generalization and adaptation** by ensuring robust speaker variation capture while maintaining speaker identity.

By organizing the embedding space into well-defined clusters, Dis-Vector ensures a more structured and interpretable representation of speaker embeddings, enhancing the **quality and accuracy of voice conversion**.

### 9.2 Language-Based Similarity Matching


During inference, the model selects the **most suitable speaker embedding** by computing **cosine similarity** between the **target speaker’s embedding** and the **pre-clustered speaker embeddings** in the database. This method prioritizes selecting a **linguistically similar speaker**, leading to:

- **Better prosody preservation**, as speakers from the same linguistic background share similar pitch and rhythm structures.
- **Accurate voice adaptation**, ensuring that even when a target speaker’s language is unseen during training, the system can infer the best match.
- **Efficient feature transfer**, allowing for natural-sounding synthesis without distorting speaker identity.

The language-based similarity approach refines the voice conversion process by focusing on both **speaker similarity and linguistic consistency**, ensuring the most **natural and high-quality voice generation**.


### 9.3 Closest Language Matching During Inference
o further enhance cross-lingual voice adaptation, Dis-Vector integrates a **nearest language matching** strategy. Given a target speaker's embedding, the system performs the following steps:

1. **Determine the closest linguistic cluster** by measuring the embedding distance to pre-computed cluster centroids.
2. **Apply a threshold-based similarity measure** to ensure the closest linguistic match is selected.
3. **If a direct match is unavailable**, the system chooses a linguistically nearest neighbor based on phonetic and prosody similarities.

This technique ensures:
- **Minimal loss in speech naturalness** by selecting speakers with the most similar phonetic structures.
- **Improved speaker adaptation**, even in cases where the target speaker’s language is underrepresented in the dataset.
- **Scalability for zero-shot voice conversion**, allowing seamless expansion with new speakers and languages.

By leveraging this clustering-based framework, Dis-Vector significantly improves the accuracy and efficiency of voice conversion in **multilingual and low-resource language settings**, making it a robust solution for **global voice synthesis applications**.

---

## 10. DIS-VECTOR: Controllable Zero-Shot Voice Conversion & Cloning Features


✅ **Zero-Shot Voice Conversion**  
→ Convert voices between unseen speakers without retraining.  

✅ **Low-Resource Language Adaptation**  
→ High-quality synthesis even in underrepresented languages.  

✅ **Cross-Gender Voice Cloning**  
→ Convert voices across gender (male ↔ female) while keeping tone natural.  

✅ **Cross-Lingual Voice Cloning**  
→ Clone and adapt voices across different languages.  

✅ **Indian Language Adaptation**  
→ Supports major Indian languages like Hindi, Tamil, Telugu, Malayalam, and Bengali.  

✅ **Disentangled Embedding Control**  
→ Independent manipulation of **Content**, **Pitch**, **Rhythm**, and **Timbre**.  

✅ **Fine-Grained Voice Cloning**  
→ Clone and adjust voice traits precisely using latent vectors.  

✅ **Language-Based Similarity Matching**  

✅ **Closest Language Matching During Inference**  

✅ **Feature Transfer Mechanism**  
→ Transfer **content**, **pitch**, **rhythm**, and **timbre** between any two speakers.  

✅ **Scalable Zero-Shot Cloning System**  

For more details, refer to the documentation. 🚀


