import os
import pickle
import numpy as np
import soundfile as sf
from numpy.random import RandomState
from utils import *

def generate_spectrogram_and_f0(config):
    sample_rate = 16000
    data_path = config.data_dir
    feature_path = config.feat_dir
    wav_output_path = os.path.join(feature_path, config.wav_dir)
    spmel_output_path = os.path.join(feature_path, config.spmel_dir)
    f0_output_path = os.path.join(feature_path, config.f0_dir)
    speaker_metadata = pickle.load(open('spk_meta.pkl', "rb"))

    base_dir, speaker_list, _ = next(os.walk(data_path))
    random_seed = 1

    for speaker_id in sorted(speaker_list):
        if speaker_id not in speaker_metadata:
            print(f'Warning: {speaker_id} not found in speaker metadata; skipping feature generation.')
            continue
        print(f'Processing features for speaker {speaker_id}')
        
        for output_folder in [wav_output_path, spmel_output_path, f0_output_path]:
            os.makedirs(os.path.join(output_folder, speaker_id), exist_ok=True)

        _, _, audio_files = next(os.walk(os.path.join(base_dir, speaker_id)))

        if speaker_metadata[speaker_id][1] == 'M':
            pitch_min, pitch_max = 50, 250
        elif speaker_metadata[speaker_id][1] == 'F':
            pitch_min, pitch_max = 100, 600
        else:
            continue

        rng = RandomState(random_seed) 
        audio_waveforms, f0_values, spectral_envelopes, aperiodicities = [], [], [], []
        for audio_file in sorted(audio_files):
            signal, _ = sf.read(os.path.join(base_dir, speaker_id, audio_file))
            if signal.shape[0] % 256 == 0:
                signal = np.concatenate((signal, np.array([1e-06])), axis=0)
            filtered_signal = filter_wav(signal, rng)

            # Extract WORLD analyzer parameters
            f0, spectral_envelope, aperiodicity = get_world_params(filtered_signal, sample_rate)
            audio_waveforms.append(filtered_signal)
            f0_values.append(f0)
            spectral_envelopes.append(spectral_envelope)
            aperiodicities.append(aperiodicity)

        # Smooth pitch for monotonic speech synthesis
        f0_values = average_f0s(f0_values, mode='global')

        for idx, (audio_file, waveform, f0, sp_env, ap_env) in enumerate(zip(audio_files, audio_waveforms, f0_values, spectral_envelopes, aperiodicities)):

            monotonic_waveform = get_monotonic_wav(waveform, f0, sp_env, ap_env, sample_rate)
            spectrogram_mel = get_spmel(waveform)
            f0_extracted, f0_normalized = extract_f0(waveform, sample_rate, pitch_min, pitch_max)
            assert len(spectrogram_mel) == len(f0_extracted)

            # Segment features into fixed-length chunks for training
            segment_start = 0
            segment_size = 49151
            while segment_start * segment_size < len(monotonic_waveform):
                waveform_segment = monotonic_waveform[segment_start * segment_size:(segment_start + 1) * segment_size]
                if len(waveform_segment) < segment_size:
                    waveform_segment = np.pad(waveform_segment, (0, segment_size - len(waveform_segment)))
                np.save(os.path.join(wav_output_path, speaker_id, os.path.splitext(audio_file)[0] + f'_{segment_start}'),
                        waveform_segment.astype(np.float32), allow_pickle=False)
                segment_start += 1
            
            features = [spectrogram_mel, f0_normalized]
            output_paths = [spmel_output_path, f0_output_path]
            for feature, output_path in zip(features, output_paths):
                segment_start = 0
                segment_size = 192
                while segment_start * segment_size < len(feature):
                    feature_segment = feature[segment_start * segment_size:(segment_start + 1) * segment_size]
                    if len(feature_segment) < segment_size:
                        if feature_segment.ndim == 2:
                            feature_segment = np.pad(feature_segment, ((0, segment_size - len(feature_segment)), (0, 0)))
                        else:
                            feature_segment = np.pad(feature_segment, ((0, segment_size - len(feature_segment)), ))
                    np.save(os.path.join(output_path, speaker_id, os.path.splitext(audio_file)[0] + f'_{segment_start}'),
                            feature_segment.astype(np.float32), allow_pickle=False)
                    segment_start += 1


def generate_metadata(config):
    feature_path = config.feat_dir
    wav_output_path = os.path.join(feature_path, config.wav_dir) 
    base_dir, speaker_list, _ = next(os.walk(wav_output_path))
    speaker_metadata = pickle.load(open('spk_meta.pkl', "rb"))
    dataset_entries = []

    for speaker_id in sorted(speaker_list):
        speaker_index, _ = speaker_metadata[speaker_id]
        speaker_embedding = np.zeros((config.dim_spk_emb,), dtype=np.float32)
        speaker_embedding[int(speaker_index)] = 1.0

        utterance_list = []
        _, _, file_list = next(os.walk(os.path.join(base_dir, speaker_id)))
        file_list = sorted(file_list)
        for audio_file in file_list:
            utterance_list.append(os.path.join(speaker_id, audio_file))
        for utterance in utterance_list:
            dataset_entries.append((speaker_id, speaker_embedding, utterance))

    with open(os.path.join(feature_path, 'dataset.pkl'), 'wb') as metadata_file:
        pickle.dump(dataset_entries, metadata_file)

def preprocess_audio_data(config):
    print('Starting audio data preprocessing...')
    generate_spectrogram_and_f0(config)
    generate_metadata(config)
    print('Preprocessing completed.')
