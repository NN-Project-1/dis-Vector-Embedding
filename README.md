##  Dis-Vector: Disentangled Voice Embeddings for Conversion and Synthesis 🎤✨


Welcome to the **Dis-Vector** project! This repository contains the implementation and evaluation of our advanced voice conversion and synthesis model that utilizes disentangled embeddings to accurately capture and transfer speaker characteristics across languages.

## 📚 Table of Contents
1. [Overview](#overview)
2. [Dis-Vector Model Details](#dis-vector-model-details)
3. [Datasets](#datasets)
4. [Evaluation](#evaluation)
5. [Results](#results)
6. [MOS Score Analysis](#mos-score-analysis)
7. [Conclusion](#conclusion)
8. [License](#license)

## 📝 Overview
The Dis-Vector model represents a significant advancement in voice conversion and synthesis by employing disentangled embeddings to precisely capture and transfer speaker characteristics. Its architecture features separate encoders for content, pitch, rhythm, and timbre, enhancing both the accuracy and flexibility of voice cloning.

## 🛠️ Dis-Vector Model Details
The Dis-Vector model consists of several key components that work together to achieve effective voice conversion and synthesis:

- **Architecture**: The model employs a multi-encoder architecture, with dedicated encoders for each feature type:
  - **Content Encoder**: Captures linguistic content and phonetic characteristics.
  - **Pitch Encoder**: Extracts pitch-related features to ensure accurate pitch reproduction.
  - **Rhythm Encoder**: Analyzes rhythmic patterns and timing to preserve the original speech flow.
  - **Timbre Encoder**: Captures unique vocal qualities of the speaker, allowing for more natural-sounding outputs.

- **Disentangled Embeddings**: The model produces a 512-dimensional embedding vector, organized as follows:
  - 256 elements for **content features**
  - 128 elements for **pitch features**
  - 64 elements for **rhythm features**
  - 64 elements for **timbre features**

- **Zero-Shot Capability**: The Dis-Vector model demonstrates remarkable zero-shot performance, enabling voice cloning and conversion across different languages without needing extensive training data for each target voice.

- **Feature Transfer**: The model facilitates the transfer of individual features from the source voice to the target voice, allowing for customizable voice synthesis while retaining the original speech's essence.

- **Evaluation Metrics**: Performance is assessed using various metrics, including Pitch Error Rate (PER), Rhythm Error Rate (RER), Timbre Error Rate (TER), and Content Preservation Rate (CPR), ensuring a comprehensive evaluation of the synthesized speech quality.

## 🗃️ Datasets
For our evaluation, we utilized the following datasets:

| Dataset Name  | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| **LIMMITS**   | Contains recordings from speakers of various Indian languages (English, Hindi, Kannada, Telugu, Bengali) with ~1 hour of speech per speaker. |
| **VCTK**      | Includes recordings from multiple speakers with different accents, providing a rich diversity for evaluation. |

## 📊 Evaluation
Quantitative analysis measures the performance of the Dis-Vector model using distance metrics and statistical measures.

### 1. Test Setup
- **Pitch Testing**: Evaluates pitch variations using Pitch Error Rate (PER).
- **Rhythm Testing**: Assesses rhythmic patterns with Rhythm Error Rate (RER).
- **Timbre Testing**: Analyzes vocal qualities using Timbre Error Rate (TER).
- **Content Testing**: Ensures content accuracy using Content Preservation Rate (CPR).
  
### 2. Distance Measurement
- **Cosine Similarity**: Evaluates feature transfer and voice synthesis. 

### 3. Ground Truth vs. TTS Output Similarity
- Similarity scores for pitch, rhythm, timbre, and content help measure synthesis accuracy.

## 📈 Results
The results of our evaluation showcase the efficacy of the Dis-Vector model compared to traditional models.

### MOS Score for Monolingual Voice Conversion

| Source Language (Gender) | Target Language (Gender) | MOS Score |
|--------------------------|--------------------------|-----------|
| English Male             | English Female           | 3.8       |
| Hindi Female             | Hindi Male               | 3.7       |

### MOS Score for Zero-Shot Cross-Lingual Voice Cloning

| Source Language (Gender) | Target Language (Gender) | MOS Score |
|--------------------------|--------------------------|-----------|
| English Male             | Hindi Female             | 3.9       |
| Hindi Female             | Telugu Male              | 3.7       |

### Comparison of DIS-Vector with D-Vector

| Source Lang. | Target Lang. | MOS LIMMITS Baseline | MOS (DIS Vector) |
|--------------|--------------|----------------------|-------------------|
| English      | English Female| 3.5                  | 3.9               |
| Hindi        | Hindi Female  | 3.4                  | 3.7               |

### Zero-Shot Cross-Lingual Cloning

| Source Lang. | Target Lang. | MOS LIMMITS Baseline | MOS (DIS Vector) |
|--------------|--------------|----------------------|-------------------|
| English      | Hindi Female  | 3.3                  | 3.8               |
| Hindi        | English Female | 3.2                  | 3.6               |

### Comparison with SpeechSplit2

| Language      | SpeechSplit2 MOS Score | DIS-Vector MOS Score |
|---------------|------------------------|-----------------------|
| English Male   | 3.4                    | 3.8                   |
| English Female | 3.5                    | 3.9                   |

## 🏁 Conclusion
The Dis-Vector model's zero-shot capabilities enable effective voice cloning and conversion across different languages, setting a new benchmark for high-quality, customizable voice synthesis. The results of our experiments, including detailed embeddings and synthesis outputs, are available in the accompanying Git repository.

For more details, please refer to the documentation in this repository! Happy experimenting! 🚀
