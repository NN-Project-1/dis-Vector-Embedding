# DIS-VECTOR: AN EFFECTIVE APPROACH FOR CONTROLLABLE ZERO-SHOT VOICE CONVERSION AND CLONING IN LOW-RESOURCE LANGUAGES 🎤✨

Welcome to the DIS-Vector project! This repository presents an advanced low-resource, zero-shot voice conversion and cloning model that leverages disentangled embeddings, clustering techniques, and language-based similarity matching to achieve highly natural and controllable voice synthesis.

The **DIS-Vector** model introduces a novel approach to voice conversion by disentangling speech components content, pitch, rhythm, and timbre into separate embedding spaces, enabling fine-grained control over voice synthesis. Unlike traditional voice conversion models, DIS-Vector is capable of zero-shot voice cloning, meaning it can synthesize voices from unseen speakers and languages without requiring large-scale speaker-specific training data.

We have Approach 1, which is the base version of DIS-Vector. The details are provided [here](https://github.com/NN-Project-1/dis-Vector-Embedding/blob/main/readme1.md).

<p align="center">
  <img src="architecture/DIS-Vector-V2.png" alt="Dis-Vector Architecture"  width="400">
</p>

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
10. [Conclusion](#10-conclusion)  

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
The Dis-Vector model consists of several key components that work together to achieve effective voice conversion and synthesis:

- **Architecture**: Multi-encoder design with dedicated encoders for each feature type:  
  - **Content Encoder**: Captures linguistic content and phonetic characteristics.  
  - **Pitch Encoder**: Extracts pitch-related features for accurate pitch reproduction.  
  - **Rhythm Encoder**: Analyzes rhythmic patterns to preserve original speech flow.  
  - **Timbre Encoder**: Captures unique vocal qualities for natural-sounding outputs.  

- **Disentangled Embeddings**: 512-dimensional embedding vector structured as:  
  - 256 elements for **content features**  
  - 128 elements for **pitch features**  
  - 64 elements for **rhythm features**  
  - 64 elements for **timbre features**  

- **Zero-Shot Capability**: Enables voice cloning and conversion across languages without extensive training.  

- **Feature Transfer**: Allows transfer of individual features from source to target voice while retaining original essence.  

---

## 3. VITS-TTS Integration

<p align="center">
  <img src="architecture/dis-vits.png" alt="DIS-Vector Architecture">
</p>

Integration with VITS leverages disentangled embeddings (content, pitch, rhythm, timbre) for fine-grained control. This enables high-quality voice conversion and zero-shot cloning, allowing generation of new voices without speaker-specific training, improving flexibility and realism.

---

## 4. GPT-TTS Integration

<p align="center">
  <img src="architecture/gpt.png" alt="DIS-Vector Architecture" width="400">
</p>

GPT-based architecture processes input text via BPE tokenizer, embeds subword tokens, and passes them through GPT-style Transformer blocks to predict VQ-VAE discrete codes. DIS-Vector embeddings are injected via dual-conditioning (concatenation + FiLM), ensuring predictions are conditioned on content and vocal characteristics. Predicted codes are decoded and passed to HiFi-GAN for waveform synthesis, enabling high-fidelity zero-shot voice cloning.

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
- Groups speakers by timbre and prosody  
- Enables zero-shot conversion for unseen speakers  
- Uses cluster centroids for nearest-match synthesis  
- Enhances speaker variation capture while maintaining identity  

### 9.2 Language-Based Similarity Matching
- Cosine similarity between target and database embeddings  
- Prioritizes linguistically similar speakers  
- Ensures prosody preservation and accurate adaptation  
- Enables natural feature transfer  

### 9.3 Closest Language Matching During Inference
1. Determine closest linguistic cluster  
2. Apply threshold-based similarity  
3. Choose nearest neighbor if direct match unavailable  

Ensures minimal loss of naturalness, better adaptation, and scalability for zero-shot conversion.

---

## 10. Conclusion
The Dis-Vector model’s zero-shot capabilities, enhanced by clustering and similarity-based retrieval, enable effective voice cloning and conversion across languages, setting a benchmark for high-quality, customizable voice synthesis.  

For more details, refer to the documentation. 🚀
