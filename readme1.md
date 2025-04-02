# Approach-1: Initializing DIS-Vector with Specified Pitch Range and Timbre-Specific Characteristics

When we first start with the DIS-vector, we begin by defining a specified pitch range and extracting timbre-specific characteristics. The process involves understanding how pitch and timbre interact in speech and capturing these nuances effectively. Extracting pitch and timbre features is crucial for accurately representing speaker identity. We begin by analyzing the fundamental frequency (F0), which varies across different speakers. Typically, adult male voices have an F0 range between 85 Hz and 180 Hz, while adult female voices generally fall between 165 Hz and 255 Hz. Children's voices tend to have a significantly higher pitch, usually ranging from 250 Hz to 400 Hz, depending on age and vocal development. However, defining a fixed pitch range for processing can be challenging, as natural speech exhibits dynamic variations due to emotional expressions, intonation, and speaking style. These variations necessitate adaptive pitch processing techniques to ensure natural-sounding synthesis and accurate speaker representation.
The fundamental frequency (F0) is calculated using the autocorrelation method:

\[
F0 = \frac{1}{T}
\]

where T is the period of the waveform. Alternatively, using cepstral analysis:

\[
F0 = \text{argmax}(\text{cepstrum}(s(t)))
\]

where s(t) is the cepstrum of the speech signal.

Beyond pitch, timbre features provide critical insights into the unique vocal characteristics of a speaker. These include formant frequencies (F1, F2, F3), spectral envelope, harmonic-to-noise ratio (HNR), and Mel-frequency cepstral coefficients (MFCCs). Formant frequencies are extracted using Linear Predictive Coding (LPC):

\[
s(n) = \sum_{k=1}^{p} a_k s(n-k) + e(n)
\]

where a_k are the LPC coefficients. Spectral envelope modeling via MFCCs is derived from the log-magnitude spectrum of speech using the Discrete Cosine Transform (DCT):

\[
\text{MFCC}(m) = \sum_{k=1}^{M} \log E(k) \cdot \cos\left(\frac{m(k-0.5)\pi}{M}\right)
\]

where E(k) represents the Mel-scaled log energies. The harmonic-to-noise ratio (HNR) measures the proportion of harmonic sound to noise and is calculated as:

\[
\text{HNR} = 10 \log_{10}\left(\frac{\text{harmonic power}}{\text{noise power}}\right)
\]

Despite our methodologies, several challenges remain in achieving a perfect match between extracted and target features. A key issue is the loss of fine-grained timbre details when using traditional feature extraction techniques. If the pitch range is too strictly defined, natural pitch variations may not be captured, leading to robotic-sounding voice synthesis. Additionally, background noise, microphone quality, and recording conditions significantly impact timbre extraction, requiring robust denoising and preprocessing techniques. Another major challenge is balancing speaker-independent and speaker-specific features. If pitch normalization is applied too aggressively, the natural speaker identity may be distorted, making the converted voice sound unnatural. On the other hand, failing to normalize pitch effectively could lead to mismatched features that hinder voice conversion performance.

To address these issues, we have explored advanced techniques such as pitch contour modeling, adaptive feature normalization, and self-supervised learning models like Wav2Vec2 and HuBERT. These methods help in preserving speaker individuality while maintaining flexibility for voice conversion applications. However, despite these methodologies, we found that the extracted pitch and timbral features did not always achieve a perfect match with the original speaker characteristics. The discrepancies in pitch alignment and timbre consistency led us to reconsider our approach. As a result, we plan to introduce improvements in Version-2, where we will refine feature extraction techniques, enhance timbre modeling, and explore more advanced neural architectures to achieve higher accuracy in voice conversion.
