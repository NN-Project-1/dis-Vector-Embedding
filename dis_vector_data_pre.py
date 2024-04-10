import os
import librosa
import argparse
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import PosixPath
parser = argparse.ArgumentParser()
parser.add_argument("--syspin_resampled_data_path", required=True,
                    help="Path to resampled challenge dataset")
parser.add_argument("--syspin_raw_data_path", required=True, 
                    help="Path to raw challenge dataset")
parser.add_argument("--manifest_folder", default="syspin_14spk_basemodel_1hr_per_spk",
                    help="save location for manifest files")
parser.add_argument("--languages", default=["Malayalam","Tamil"], 
                    help="challenge data languages")
parser.add_argument("--generate_charecter_set", default=True, 
                    help="Find list of charecters in dataset")
parser.add_argument("--speakers", default=["Male", "Female"], 
                    help="challenge data speakers")
parser.add_argument("--duration_folder", default="speaker_durations", 
                    help="store duration of speakers")
parser.add_argument("--run_mode", default="syspin_prep", 
                    help="select which function to process")
parser.add_argument("--nj", default=64, 
                    help="num proc")
parser.add_argument("--sr", default=16000, 
                    help="sample rate")
parser.add_argument("--file_limit", default=2000, 
                    help="num of files to process per speaker")
parser.add_argument("--dur_range", default=(2, 15), 
                    help="duration range of audio")
parser.add_argument("--dur_selection_per_speaker", default=1, 
                    help="num of hours per speaker")

def read(path):
    try:
        y, sr = librosa.load(str(path), sr=16000) 
        return len(y) / sr
    except:
        return None
        
def get_files(path, extension='.wav'):
    path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))

def get_durations(path, save_path):
    files = get_files(path)[:3000]
    
    results = []  # Create an empty list to store the results
    # Loop through each audio file and process them one by one
    for file in tqdm(files, total=len(files)):
      result = read(file)
      results.append(result)
    with open(save_path, "w") as f:
        for idx in range(len(files)):
            if results[idx] == None: continue
            if results[idx] < 2 or results[idx] > 15: continue
            f.write(f"{str(files[idx])}\t{results[idx]}\n")
    
def syspin_prep():
          charecters = set()
          languages=["Malayalam","Tamil","English","Kannada","Hindi","Marathi","Telugu"]
          speakers=["Male",'Female']
          for lang in languages:  
            for spk in speakers:
              manifest_files = []
              spk = lang + "_" + spk
              speaker_folder =os.path.join(args.syspin_resampled_data_path, spk)
              speaker_folder =os.path.join(speaker_folder, "wav")
              duration_file = os.path.join(args.duration_folder, spk + ".tsv")
              if not os.path.exists(duration_file):
                print(f"Extracting durations from speaker - {spk}")
                get_durations(speaker_folder, duration_file)
            
              manifest_savename = os.path.join(args.manifest_folder, spk + ".tsv")            
              if not os.path.exists(manifest_savename): 
                txt_path=os.path.join(args.syspin_raw_data_path, spk)
                txt_path=os.path.join(txt_path, "txt")
                txt_files = get_files(txt_path, extension='.txt')
                for i in range(len(txt_files)):
                   txt_files[i] = str(txt_files[i])
                with open(duration_file, "r") as f:
                    lines = f.readlines()
                duration_per_spk = 0
                for line in lines:
                    duration = line.split("\t")[-1].strip("\n")
                    id = Path(line.split("\t")[0]).stem
                    id="/home/vijay/Desktop/All_Files/ALL_TTS/DIS_Vector_Embedding_Integration/Data/"+spk+'/'+"txt/"+id+".txt"
                    if id not in txt_files: continue
                    with open(id, "r") as f:
                        text = f.read().strip("\n").strip()
                    if duration_per_spk < 1 * 3600:
                        duration_per_spk += float(duration) 
                        manifest_files.append(
                            "\t".join((line.split("\t")[0],
                                    spk,
                                    text
                        )))
            
            
                with open(manifest_savename, "w") as f:
                    for line in manifest_files:
                        f.write(line + "\n")
                        
              with open(manifest_savename, "r") as f:
                data = f.readlines()
              for line in data:
                charecters.update(set(line.split("\t")[-1].strip("\n")))
            if args.generate_charecter_set:
              print(f"{len(charecters)} charecters")
              with open(os.path.join(args.manifest_folder, "charecters.txt"), "w") as f:
                  for ch in charecters:
                     f.write(ch)

def main():
    globals()[args.run_mode]()

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.duration_folder):
        os.makedirs(args.duration_folder)
    if not os.path.exists(args.manifest_folder):
        os.makedirs(args.manifest_folder)
    main()