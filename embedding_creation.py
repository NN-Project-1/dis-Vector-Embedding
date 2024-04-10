import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig, CharactersConfig
from TTS.utils.downloaders import download_vctk
from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.managers import save_file
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.data import get_length_balancer_weights
from TTS.tts.utils.languages import LanguageManager, get_language_balancer_weights
from TTS.tts.utils.speakers import SpeakerManager, get_speaker_balancer_weights, get_speaker_manager
torch.set_num_threads(8)


OUT_PATH = "/home/vijay/Desktop/All_Files/ALL_TTS/DIS_Vector_Embedding_Integration/output" 
DATA_FOLDER = "/home/vijay/Desktop/All Files/ALL TTS /DIS_Vector_Embedding_Integration/Data"
MANIFEST_FOLDER = "/home/vijay/Desktop/All_Files/ALL_TTS/DIS_Vector_Embedding_Integration/output/manifest_folder"

with open("/home/vijay/Desktop/All_Files/ALL_TTS/DIS_Vector_Embedding_Integration/output/manifest_folder/charecters.txt", "r") as f:
    charecter_set = f.read().strip("\n")

SKIP_TRAIN_EPOCH = False
BATCH_SIZE = 16
SAMPLE_RATE = 16000
MAX_AUDIO_LEN_IN_SECONDS = 10
NUM_RESAMPLE_THREADS = 10

## Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = "/home/vijay/Desktop/All_Files/ALL_TTS/DIS_Vector_Embedding_Integration/model_se.pth.tar"

SPEAKER_ENCODER_CONFIG_PATH = "/home/vijay/Desktop/All_Files/ALL_TTS/DIS_Vector_Embedding_Integration/config_se.json"

# List of TSV files
tsv_files = [
    "Malayalam_Male.tsv", "Malayalam_Female.tsv",
    "Tamil_Male.tsv", "Tamil_Female.tsv",
    "English_Male.tsv","English_Female.tsv",
    "Hindi_Female.tsv","Hindi_Male.tsv",
    "Kannada_Female.tsv","Kannada_Male.tsv",
    "Marathi_Male.tsv","Marathi_Female.tsv",
    "Telugu_Female.tsv","Telugu_Male.tsv"
]

DATASETS_CONFIG_LIST=[]
D_VECTOR_FILES = []

for filename in tsv_files:
    language, gender = filename.split("_")[0], filename.split("_")[1].split(".")[0]
    meta_file_train = os.path.join(MANIFEST_FOLDER, filename)
    print("manifest_train :",meta_file_train)
    dataset_config = BaseDatasetConfig(
        formatter="v_for",
        meta_file_train=meta_file_train,
        path=OUT_PATH,
        language=language,
    )

    embeddings_folder = os.path.join(OUT_PATH, f"{language}_{gender}")
    os.makedirs(embeddings_folder, exist_ok=True)
    embeddings_file = os.path.join(embeddings_folder,"speakers.pth")
    
    if not os.path.isfile(embeddings_file):
       print("embeddings_folder :",embeddings_folder)
       compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_speakers_file=None,
            config_dataset_path=None,
            formatter_name=dataset_config.formatter,
            dataset_name=dataset_config.dataset_name,
            dataset_path=dataset_config.path,
            meta_file_train=dataset_config.meta_file_train,
            meta_file_val=dataset_config.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    DATASETS_CONFIG_LIST.append(dataset_config)
    D_VECTOR_FILES.append(embeddings_file)

for dataset_config in DATASETS_CONFIG_LIST:
    print(f"Language: {dataset_config.language}, TSV File: {dataset_config.meta_file_train}")
print('DATASETS_CONFIG_LIST:::',DATASETS_CONFIG_LIST)

# audio_config = VitsAudioConfig(
#     sample_rate=SAMPLE_RATE,
#     hop_length=256,
#     win_length=1024,
#     fft_size=1024,
#     mel_fmin=0.0,
#     mel_fmax=None,
#     num_mels=80,
# )
# model_args = VitsArgs(
#     d_vector_file=D_VECTOR_FILES,
#     use_language_embedding=True,
#     embedded_language_dim=4,
#     use_d_vector_file=True,
#     d_vector_dim=512,
#     num_layers_text_encoder=10,
#     speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
#     speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
#     resblock_type_decoder="2",  
# )
# config = VitsConfig(
#     output_path=OUT_PATH,
#     model_args=model_args,
#     run_name="yourtts_2_spk_from_scratch",
#     project_name="YourTTS",
#     run_description="""
#             - Original YourTTS trained using VCTK dataset
#         """,
#     dashboard_logger="tensorboard",
#     logger_uri=None,
#     audio=audio_config,
#     batch_size=16,
#     batch_group_size=48,
#     eval_batch_size=16,
#     num_loader_workers=8,
#     eval_split_max_size=256,
#     print_step=50,
#     plot_step=100,
#     epochs=1000,
#     log_model_step=1000,
#     save_step=5000,
#     save_n_checkpoints=2,
#     save_checkpoints=True,
#     target_loss="loss_1",
#     print_eval=False,
#     use_phonemes=False,
#     phonemizer="espeak",
#     phoneme_language="hi",
#     compute_input_seq_cache=True,
#     add_blank=True,
#     text_cleaner="multilingual_cleaners",
#     characters=CharactersConfig(
#         characters_class="TTS.tts.models.vits.VitsCharacters",
#         pad="_",
#         eos="&",
#         bos="*",
#         blank=None,
#         characters=charecter_set,
#         punctuations="!\u00a1'(),-.:;\u00bf? ",
#         phonemes="",
#         is_unique=True,
#         is_sorted=True,
#     ),
#     phoneme_cache_path= "/content/drive/MyDrive/TA_ML_Data/phoneme_cache_path",
#     precompute_num_workers=12,
#     start_by_longest=True,
#     datasets=DATASETS_CONFIG_LIST,
#     cudnn_benchmark=False,
#     mixed_precision=False,
#     test_sentences = [
#     ["ചെയ്യുന്ന അഭിപ്രായങ്ങ മലയാള മനോരമയുടേതല്ല അഭിപ്രായങ്ങളുടെ","Malayalam_Female",None,"Malayalam"],
#     ["മലയാള മനോരമയുടേതല്ല അഭിപ്രായങ്ങളുടെ ","Malayalam_Male",None,"Malayalam"],
#     ["எந்தக் கட்சி எந்த நபரிடம் எவ்வளவு பணம் பெற்றது என்பதைத் தெளிவாகப் பட்டியல் போட்டுக் காட்டி உள்ளது இந்தப் புதிய தரவுகள்","Tamil_Female",None,"Tamil"],
#     ["தயவு செய்து எனது உணவுக்கான பில் கொண்டு வாருங்கள்","Tamil_Male",None,"Tamil"]
#       ],
#     use_weighted_sampler=True,
#     speaker_encoder_loss_alpha=9.0,
#     )

# train_samples, eval_samples = load_tts_samples(
#     config.datasets,
#     eval_split=True,
#     eval_split_max_size=config.eval_split_max_size,
#     eval_split_size=config.eval_split_size,
# )
# language_manager = LanguageManager(config=config)
# config.model_args.num_languages = language_manager.num_languages

# model = Vits.init_from_config(config)

# trainer = Trainer(
#     TrainerArgs(restore_path=None, skip_train_epoch=False),
#     config,
#     output_path=OUT_PATH,
#     model=model,
#     train_samples=train_samples,
#     eval_samples=eval_samples,
# )
# trainer.fit()