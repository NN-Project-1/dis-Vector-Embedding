# DIS-VECTOR: AN EFFECTIVE APPROACH FOR CONTROLLABLE ZERO-SHOT VOICE CONVERSION AND CLONING IN LOW-RESOURCE LANGUAGES 🎤✨

Welcome to the DIS-Vector project! This repository presents an advanced low-resource, zero-shot voice conversion and cloning model that leverages disentangled embeddings, clustering techniques, and language-based similarity matching to achieve highly natural and controllable voice synthesis.

The **DIS-Vector** model introduces a novel approach to voice conversion by disentangling speech components—content, pitch, rhythm, and timbre—into separate embedding spaces, enabling fine-grained control over voice synthesis. Unlike traditional voice conversion models, DIS-Vector is capable of zero-shot voice cloning, meaning it can synthesize voices from unseen speakers and languages without requiring large-scale speaker-specific training data.

We have **Approach 1**, which is the base version of DIS-Vector. The details are provided **[here](https://github.com/NN-Project-1/dis-Vector-Embedding/blob/main/readme1.md)**.

**Demo:** [https://nn-project-1.github.io/dis-vector_web/](https://nn-project-1.github.io/dis-vector_web/)


---

## 📚 Table of Contents
1. [Overview](#1-overview)  
2. [Dis-Vector Model Details](#2-dis-vector-model-details)
3. [Length Analysis](#5-length-analysis)  
4. [E2E-TTS Integration](#3-vits-tts-integration)  
5. [Types of Loss Functions](#7-types-of-loss-functions)  
6. [Evaluation](#8-evaluation)  
   1. [Test Setup](#81-test-setup)  
   2. [Distance Measurement](#82-distance-measurement)  
   3. [Ground Truth vs. TTS Output Similarity](#83-ground-truth-vs-tts-output-similarity)  
7. [Clustering & Language Matching](#9-clustering--language-matching)  
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


## 2. Dis-Vector Model Details
The DIS-Vector model is a multi-encoder disentanglementbased speech representation framework developed for expressive, cross-lingual, and zero-shot voice conversion. The design goal is to separate key aspects of human speech content, pitch, rhythm, and timbre into distinct, independently controllable latent embeddings. This architecture enables precise manipulation of each speech attribute while maintaining perceptual coherence in synthesis, allowing natural-sounding speaker conversion and expressive style transfer without retraining.

<p align="center">
  <img src="architecture/DIS-Vector-V2.png" alt="Dis-Vector Architecture"  width="400">
</p>

DIS-Vector follows a **parallel encoder structure** in which each encoder extracts a distinct feature domain from synchronized mel-spectrogram frames represented as `[B, T, F]`, where `B` is batch size, `T` denotes sequence length, and `F` represents mel-frequency bins. The four encoders content, pitch, rhythm, and timbre operate concurrently to generate their respective latent representations. These outputs are concatenated into a single **512-dimensional embedding vector**, forming a unified disentangled speech representation suitable for downstream decoding and synthesis.


### Encoder Specifications  

#### Content Encoder  

The **Content Encoder** captures linguistic and phonetic structures that define the spoken message while remaining independent of speaker characteristics. It utilizes a **hybrid CNN–LSTM architecture**. The convolutional layers model local spectral correlations and phonetic transitions from short-term mel segments, whereas the LSTM layers preserve long-term linguistic continuity across frames. This combination ensures that both spectral detail and sequential context are effectively represented.

The convolutional stack contains several 1D convolutional layers with kernel sizes between 3 and 5, followed by ReLU activation and layer normalization. The LSTM network, with a hidden dimension of 256, processes these convolutional outputs sequentially to generate a **256-dimensional content latent vector** (`z_c`). This latent vector encodes the phoneme-level linguistic structure necessary for accurate speech reconstruction and style independent synthesis.


#### Pitch Encoder  

The **Pitch Encoder** focuses on modeling the **fundamental frequency (F₀)** contour and tonal variations that determine intonation and expressiveness. It employs a **CNN–LSTM design** similar to the content encoder. The convolutional layers extract frequency periodicity and harmonic structure from mel-spectrogram inputs, while the LSTM captures frame-to-frame pitch progression and smooth tonal movement.

The F₀ contour is first extracted from the waveform using a pitch estimation algorithm such as **PyWorld**, followed by log-normalization and alignment with mel frames. The CNN captures harmonic energy variations, and the LSTM models dynamic changes over time. The resulting **128-dimensional pitch latent vector** (`z_p`) represents tonal shape, direction, and smoothness while suppressing speaker-specific spectral effects. This latent serves as a precise prosodic descriptor, enabling tonal transfer across different speakers without losing natural pitch consistency.


#### Rhythm Encoder  

The **Rhythm Encoder** models the temporal structure of speech at the **frame level**, focusing on the timing, duration, and energy variations that define rhythmic patterns. The input mel-spectrogram sequence is processed using a **CNN–LSTM architecture** designed to learn both local and sequential temporal cues. The convolutional layers extract short-range frame-level amplitude modulations and energy transitions corresponding to syllable boundaries and intra-word timing. Each convolutional operation is followed by ReLU activation and layer normalization to stabilize feature scaling across frames.

The convolutional output sequence is passed to the LSTM network, which operates over the same frame-aligned time axis. The LSTM captures extended dependencies between consecutive frames, modeling duration patterns, inter-phoneme gaps, and rhythmic continuity across the entire utterance. During training, frame-level duration labels are obtained from **forced alignment outputs** (e.g., Montreal Forced Aligner), where each mel frame is explicitly aligned to its corresponding phoneme boundary. These aligned frame-level mappings enable the LSTM to learn the precise temporal distribution of speech frames.

The final hidden state sequence from the LSTM is mean-pooled across frames to obtain a **64-dimensional rhythm latent vector (`z_r`)**. This vector encodes detailed timing features, including speech rate, stress placement, and pause distribution. During synthesis, `z_r` determines frame-level timing control, allowing modification of utterance pacing and duration patterns while preserving the linguistic and pitch characteristics encoded in other latent representations.


#### Timbre Encoder  

The **Timbre Encoder** processes mel-spectrogram sequences at the frame level to extract speaker specific spectral features that define vocal identity, resonance, and spectral coloration. The encoder is implemented using a **Transformer-based architecture** optimized for long range dependency modeling across both frequency and temporal dimensions. Each Transformer block consists of **multi-head self-attention (MHSA)**, **position-wise feed-forward layers**, **layer normalization**, and **residual connections**. The MHSA mechanism computes attention weights across all time–frequency positions, allowing each frame embedding to integrate spectral context from the entire utterance.

During processing, mel-spectrogram frames are linearly projected into a fixed embedding space and combined with positional encodings that preserve frame order. The attention module analyzes correlations between frequency bands, capturing resonance and spectral envelope patterns that distinguish one speaker from another. Feed-forward sublayers apply non-linear transformations to refine feature separability, while residual normalization stabilizes gradient flow during training. The output of the final Transformer layer is mean-pooled across frames to generate a fixed-length **64-dimensional timbre latent vector (`z_t`)**. This latent vector represents the static and dynamic timbral attributes that uniquely characterize the speaker’s voice, including vocal tract shape, formant structure, and spectral slope behavior.

After all encoders complete their feature extraction, the resulting latent vectors are concatenated to form a unified **512-dimensional composite representation** defined as:

                                             z_DIS = [z_c; z_p; z_r; z_t]

### Decoder and Reconstruction  

The **Decoder** performs frame-level mel-spectrogram reconstruction from the unified 512-dimensional **DIS-vector** `[z_c; z_p; z_r; z_t]`. The input vector sequence is first linearly projected and temporally expanded to match the original frame resolution. This projection initializes the decoder input sequence, where each frame embedding represents the fused acoustic state derived from content, pitch, rhythm, and timbre components.

The decoder architecture is implemented using **stacked Transformer-based upsampling blocks** followed by **convolutional refinement layers**. Each Transformer block consists of multi-head self-attention (MHSA), feed-forward sublayers, and residual normalization. The MHSA mechanism computes attention weights over all frame positions, allowing each reconstructed frame to access long-range contextual information across the entire utterance. This operation models inter-frame dependencies in both temporal and spectral dimensions, ensuring that transitions between phonemes and prosodic segments remain continuous.

Following the attention layers, temporal upsampling is performed through learned linear interpolation modules that double the frame resolution at each stage. This process restores the original temporal resolution of the mel-spectrogram without loss of synchronization. After upsampling, **1D convolutional refinement layers** with kernel size 5 are applied to each frame sequence to enhance local spectral resolution and correct frame-level distortions. These convolutional layers reconstruct harmonic detail and formant structure from the encoded latent features.

The decoder output is a sequence of mel-spectrogram frames `Ŝ ∈ ℝ^{T×F}`, where each frame corresponds directly to its original temporal index. The reconstructed spectrogram is then converted into the final waveform using a pretrained **HiFi-GAN neural vocoder** operating in 22.05 kHz sampling mode. The vocoder synthesizes waveform samples directly from the decoder’s mel output, preserving amplitude envelope and spectral envelope consistency.  

This reconstruction flow maintains strict frame-level alignment between input latents and output acoustics, ensuring that content, pitch, rhythm, and timbre information are coherently mapped to the final audio representation without cross-domain interference.


## 3. Length Analysis 

Ablation experiments were conducted to determine the optimal latent dimensionality for the unified DIS-vector representation. The embedding dimension directly affects the model’s ability to encode disentangled acoustic information across linguistic, prosodic, rhythmic, and timbral domains. Models were trained with varying latent sizes 256, 512, 768, and 1024 dimensions under identical training conditions and data configurations.  

At 256 dimensions, the reduced latent capacity led to significant degradation in reconstruction fidelity, particularly in representing timbral richness and cross-speaker spectral variations. The decoder exhibited over-smoothing effects in the high-frequency regions, indicating insufficient embedding granularity to retain speaker-specific nuances.  

Increasing the latent dimension to 768 and 1024 improved information retention marginally but introduced redundancy across latent channels. These higher-dimensional variants resulted in slower convergence rates and unstable disentanglement behavior, as excessive capacity allowed overlapping feature representations between pitch, rhythm, and timbre subspaces.  

Empirical evaluation showed that 512-dimensional embeddings provided an optimal trade-off between representational richness and computational efficiency. This configuration maintained high perceptual quality while ensuring stable convergence across training epochs. The 512D latent space demonstrated sufficient discriminative power to separate linguistic, prosodic, and speaker-dependent information without introducing redundancy. Consequently, the final DIS-vector architecture employs a **512-dimensional unified representation**, experimentally validated as the most balanced and efficient configuration for precise and interpretable acoustic disentanglement.


## 4. E2E-TTS Integration

The Dis-Vector model represents a significant advancement in voice conversion and synthesis by employing disentangled embeddings and clustering methodologies to precisely capture and transfer speaker characteristics. It introduces a novel language-based similarity approach and K-Means clustering for efficient speaker retrieval and closest language matching during inference.
Integrating the DIS-Vector framework within modern TTS systems enhances synthesis controllability and disentanglement across linguistic, prosodic, and timbral domains. The unified 512-dimensional latent vector acts as a conditioning signal that independently modulates acoustic, rhythmic, and phonetic representations within the synthesis pipeline. This section describes the integration of DIS-Vector with **TTS** (VITS-based) and **GPT-TTS**, focusing on architectural details and embedding-level interaction mechanisms.  

### 4.1 VITS Integration  

<p align="center">
  <img src="architecture/d-v.png" alt="VITS + DIS-Vector Integration" width="450">
</p>

The VITS architecture integrates disentangled speech embeddings from DIS-Vector to support multi-speaker, zero-shot, and cross-lingual synthesis through a unified generative pipeline comprising a text encoder, posterior encoder, and flow-based decoder with a HiFi-GAN vocoder. The text encoder combines convolutional and Transformer layers to extract local phonetic features and long-range linguistic dependencies, producing frame-level linguistic priors. The posterior encoder, implemented with LSTMs, processes ground-truth mel-spectrograms into latent posterior variables that capture prosodic and temporal patterns. Pitch and rhythm latents from the DIS-Vector are applied to these posterior variables via Adaptive Instance Normalization (AdaIN) and Feature-wise Linear Modulation (FiLM), enabling explicit control over F₀ dynamics, rhythm, and energy contours. The flow-based decoder maps the posterior distribution into an acoustic prior while the timbre latent modulates affine coupling and flow transformations, controlling formant structure and harmonic balance. The HiFi-GAN vocoder reconstructs waveforms from the decoded mel-spectrograms using the same timbre latent for spectral consistency. DIS-Vector integration occurs throughout: the content latent aligns text encoder outputs with frame-level priors, the rhythm and pitch latents modulate prosodic structure, and the timbre latent conditions the decoder and vocoder for speaker identity control. During inference, substituting the timbre embedding from unseen speakers enables zero-shot cloning with preservation of linguistic, rhythmic, and prosodic detail, allowing fine-grained, disentangled manipulation of speech factors within an end-to-end generative framework.

### 4.2 GPT-TTS Integration

<p align="center">
  <img src="architecture/gpt.png" alt="DIS-Vector Architecture" width="400">
</p>

The GPT-based TTS architecture functions as an autoregressive text-to-acoustic generator using a Transformer decoder trained on discrete latent representations from a Vector-Quantized Variational Autoencoder (VQ-VAE). Input text is tokenized with Byte-Pair Encoding (BPE) to form subword units, which are embedded and processed through stacked Transformer decoder layers to predict quantized acoustic tokens. The DIS-Vector provides a 512-dimensional conditioning latent used for linguistic alignment and speaker-specific modulation. This vector is projected to match the Transformer embedding dimension and integrated through two mechanisms: concatenation with token embeddings at each timestep to incorporate content, prosody, and timbre cues, and Feature-wise Linear Modulation (FiLM) to control feed-forward and attention activations through scale and shift coefficients, ensuring accurate pitch, rhythm, and spectral shaping. The decoder outputs discrete VQ indices corresponding to quantized mel-spectrogram segments, which are reconstructed by the VQ-VAE decoder into continuous acoustic features. A HiFi-GAN vocoder synthesizes the waveform from these features, conditioned by the same DIS-Vector embeddings to maintain coherence between linguistic and acoustic parameters. The integration of DIS-Vector enables zero-shot voice cloning and cross-lingual synthesis by transferring rhythm and timbre embeddings from unseen speakers while preserving linguistic and acoustic structure for controlled and consistent generation.


## 5. Types of Loss Functions

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

## Experiments and Evaluation

The experimental evaluation of the DIS-Vector framework is designed to rigorously validate explicit disentanglement, zero-shot generalization, and perceptual quality under controlled inference conditions. The core objective is to empirically demonstrate that content, pitch, rhythm, and timbre are independently encoded in separate latent spaces and can be manipulated without inducing unintended changes in non-target speech attributes. All experiments are conducted under strict zero-shot conditions, where both speakers and language pairs used during evaluation are entirely unseen during training.

DIS-Vector represents speech using four explicitly disentangled latent embeddings: content (`z_c`), pitch (`z_p`), rhythm (`z_r`), and timbre (`z_t`). Each factor is learned using an independent encoder, while decoding is performed using a shared decoder–vocoder pipeline. This architectural choice ensures that any observed factor independence in synthesized speech arises from true latent disentanglement rather than decoder specialization.

To validate disentanglement, factor-wise latent manipulation experiments are conducted. Given a source utterance and a target utterance, only one latent embedding is replaced at a time while the remaining embeddings are kept fixed. The modified latent tuple is concatenated to form the unified DIS-vector and passed through the shared decoder and neural vocoder. This controlled substitution establishes a causal link between the manipulated latent factor and the resulting acoustic variation in the synthesized speech.

All experiments are performed under zero-shot inference conditions. No speakers or language pairs overlap with training data, and no speaker adaptation, fine-tuning, or language-specific calibration is applied at inference time. For each latent factor, one hundred synthesized utterances are generated using factor-wise substitution, and both target-factor variation and non-target-factor stability are quantitatively evaluated.

Content consistency is evaluated using cosine similarity between content embeddings extracted from synthesized speech and target content references. Lower similarity values indicate stronger content adaptation, while higher values indicate content preservation. Pitch variation is measured using fundamental frequency (F0) RMSE, rhythmic variation is evaluated using Dynamic Time Warping (DTW) on frame-level duration contours, and timbre variation is quantified using Equal Error Rate (EER) from a pretrained speaker verification model. Perceptual quality and similarity are evaluated using Mean Opinion Score (MOS).

### Quantitative Disentanglement Results

| Modified Factor | Content Cosine ↓ | Pitch RMSE ↑ | Rhythm DTW ↑ | Timbre EER ↑ | MOS ↑ |
|-----------------|-----------------|--------------|--------------|--------------|-------|
| Content Only    | 0.42            | 1.2 Hz       | 0.85         | 0.45         | 3.8   |
| Pitch Only      | 0.91            | 45.8 Hz      | 0.88         | 0.42         | 3.7   |
| Rhythm Only     | 0.89            | 1.5 Hz       | 0.92         | 0.43         | 3.6   |
| Timbre Only     | 0.90            | 1.3 Hz       | 0.86         | 0.48         | 4.1   |
| All Factors     | 0.45            | 43.2 Hz      | 0.90         | 0.46         | 3.9   |

Lower values indicate stronger change in the target factor, while higher values indicate successful manipulation. The results show that manipulating each latent factor produces a strong response in its corresponding metric while leaving non-target metrics largely unchanged. Across all conditions, cross-factor interference remains below 15%, confirming effective disentanglement under zero-shot inference.

Replacing only the content embedding produces a substantial shift in acoustic–phonetic structure, reflected by low content cosine similarity, while pitch, rhythm, and timbre remain stable. Pitch manipulation results in large F0 deviations without affecting content or speaker identity. Rhythm manipulation alters temporal pacing while preserving phonetic structure and timbre. Timbre manipulation yields the strongest speaker identity shift with minimal impact on content and prosody. These results collectively confirm independent controllability of all four factors.

### ASR-Based Content Preservation Evaluation

To further verify linguistic content preservation, ASR-based evaluation is conducted using Word Error Rate (WER) and Character Error Rate (CER). External ASR systems are employed to avoid bias from internal representations. Whisper (large-v3) is used for English, while an Indic ASR pipeline based on IndicWav2Vec and IndicTrans is used for Hindi, Tamil, Telugu, and Malayalam.

The evaluation dataset consists of five hundred utterances across five languages, with ten speakers per language and no speaker overlap with training data. Original speech and DIS-Vector–synthesized speech are transcribed using identical ASR configurations, and WER and CER are computed by aligning ASR hypotheses.

### ASR Content Preservation Results

| Language   | WER (%) Original | WER (%) DIS-Vector | CER (%) Original | CER (%) DIS-Vector |
|-----------|------------------|--------------------|------------------|--------------------|
| English   | 3.2              | 4.1                | 1.1              | 1.8                |
| Hindi     | 5.8              | 6.5                | 2.3              | 2.9                |
| Tamil     | 7.2              | 8.1                | 3.1              | 3.8                |
| Telugu    | 6.9              | 7.7                | 2.8              | 3.4                |
| Malayalam | 6.5              | 7.2                | 2.5              | 3.1                |

The average WER increase of 0.9% and CER increase of 0.7% demonstrate that linguistic content is preserved even under zero-shot and cross-speaker synthesis. These results confirm that the content embedding encodes acoustic–phonetic structure while remaining invariant to pitch, rhythm, and timbre manipulation.

### Subjective Evaluation: Mean Opinion Scores

Subjective evaluation is conducted using Mean Opinion Score (MOS) to assess perceptual naturalness, speaker similarity, and transfer quality. Twenty-five listeners evaluate randomized samples using a five-point scale under controlled listening conditions.

#### Monolingual Voice Conversion

| Source Language | Target Language | MOS |
|-----------------|-----------------|-----|
| English (M)     | English (F)     | 3.8 |
| Hindi (F)       | Hindi (M)       | 3.7 |

#### Cross-Lingual Voice Conversion

| Source Language | Target Language | MOS |
|-----------------|-----------------|-----|
| English (M)     | Hindi (F)       | 3.9 |
| Hindi (F)       | Telugu (M)      | 3.7 |

#### Zero-Shot Cross-Lingual Voice Cloning

| Source Language | Target Language | MOS |
|-----------------|-----------------|-----|
| English (M)     | Hindi (F)       | 3.8 |
| Hindi (M)       | English (F)     | 3.6 |

#### Zero-Shot Monolingual Voice Cloning

| Source Language | Target Language | MOS |
|-----------------|-----------------|-----|
| English (M)     | English (F)     | 3.9 |
| Hindi (M)       | Hindi (F)       | 3.7 |

### Test Setup

<p align="center">
  <img src="architecture/plot_dif.png" alt="DIS-Vector Evaluation Pipeline" width="300">
</p>

The test setup evaluates pitch using Pitch Error Rate (PER), rhythm using Rhythm Error Rate (RER), timbre using Timbre Error Rate (TER), and content using Content Preservation Rate (CPR), ensuring consistent and factor-specific assessment across all experiments.



## 7. Clustering & Language Matching

### 7.1 K-Means Clustering for Speaker Embeddings
Dis-Vector utilizes a **language-annotated speaker embedding database**, where each speaker is mapped to a distinct feature representation based on their **timbre and prosody characteristics**. To enable efficient **cross-speaker and cross-language voice conversion**, we apply **K-Means clustering** on these high-dimensional embeddings. This clustering process helps to:

- **Group speakers** based on intrinsic vocal attributes such as pitch, intonation, and articulation patterns.
- **Enable zero-shot voice conversion** by leveraging cluster-based matching, even for unseen speakers.
- **Assign cluster centroids as representative embeddings**, allowing the system to select the closest match for synthesis.
- **Improve generalization and adaptation** by ensuring robust speaker variation capture while maintaining speaker identity.

By organizing the embedding space into well-defined clusters, Dis-Vector ensures a more structured and interpretable representation of speaker embeddings, enhancing the **quality and accuracy of voice conversion**.

### 7.2 Language-Based Similarity Matching

During inference, the model selects the **most suitable speaker embedding** by computing **cosine similarity** between the **target speaker’s embedding** and the **pre-clustered speaker embeddings** in the database. This method prioritizes selecting a **linguistically similar speaker**, leading to:

- **Better prosody preservation**, as speakers from the same linguistic background share similar pitch and rhythm structures.
- **Accurate voice adaptation**, ensuring that even when a target speaker’s language is unseen during training, the system can infer the best match.
- **Efficient feature transfer**, allowing for natural-sounding synthesis without distorting speaker identity.

The language-based similarity approach refines the voice conversion process by focusing on both **speaker similarity and linguistic consistency**, ensuring the most **natural and high-quality voice generation**.

### 7.3 Closest Language Matching During Inference
To further enhance cross-lingual voice adaptation, Dis-Vector integrates a **nearest language matching** strategy. Given a target speaker's embedding, the system performs the following steps:

1. **Determine the closest linguistic cluster** by measuring the embedding distance to pre-computed cluster centroids.
2. **Apply a threshold-based similarity measure** to ensure the closest linguistic match is selected.
3. **If a direct match is unavailable**, the system chooses a linguistically nearest neighbor based on phonetic and prosody similarities.

This technique ensures:
- **Minimal loss in speech naturalness** by selecting speakers with the most similar phonetic structures.
- **Improved speaker adaptation**, even in cases where the target speaker’s language is underrepresented in the dataset.
- **Scalability for zero-shot voice conversion**, allowing seamless expansion with new speakers and languages.

By leveraging this clustering-based framework, Dis-Vector significantly improves the accuracy and efficiency of voice conversion in **multilingual and low-resource language settings**, making it a robust solution for **global voice synthesis applications**.



## Experimental Analysis

The MOS evaluation results indicate that the **DIS-Vector** model achieves consistently high perceptual quality across both monolingual and cross-lingual settings. Monolingual voice conversion and cloning experiments exhibit slightly higher MOS values, reflecting improved speaker similarity and naturalness when the source and target languages are identical.

Cross-lingual and zero-shot voice cloning experiments demonstrate the robustness of the DIS-Vector representation in disentangling speaker identity from linguistic content. The relatively small degradation in MOS scores across unseen language and speaker combinations suggests strong generalization capability, making the proposed approach suitable for multilingual and zero-shot voice synthesis scenarios.


## References

- **Shabdh: A Multi-Lingual Zero-Shot Voice Cloning Approach with Speaker Disentanglement**  
  🔗 https://ieeexplore.ieee.org/document/10890203



## 8. DIS-VECTOR: Controllable Zero-Shot Voice Conversion & Cloning Features

✅ **Zero-Shot Voice Conversion**  
✅ **Low-Resource Language Adaptation**  
✅ **Cross-Gender Voice Cloning**  
✅ **Cross-Lingual Voice Cloning**  
✅ **Indian Language Adaptation**  
            → Supports major Indian languages like Hindi, Tamil, Telugu, Malayalam, and Bengali.  
✅ **Disentangled Embedding Control**  
✅ **Language-Based Similarity Matching**  
✅ **Feature Transfer Mechanism**  
→ Transfer **content**, **pitch**, **rhythm**, and **timbre** between any two speakers.  

For more details, refer to the documentation. 🚀


