# Explaining venture teams’ opportunity identification through multimodal data

*Authors : Alessio Desogus, Haoyu Wen and Jiewei Li*

> **_🔔 IMPORTANT NOTE:_** As agreed with the [ENTC](https://www.epfl.ch/labs/entc/) supervisor of this project, *Dr. Davide Bavato*, we have signed a non-disclosure agreement (NDA) and won't be able to provide the data we received in this repository. However, our three notebooks with their results are available and are sufficient to compare the results displayed in the report.
 
>
## Abstract 📝



## Setup ⚙️



### WhisperX Setup :
- To install `WhisperX`, all the instructions can be found [here](https://github.com/m-bain/whisperX/blob/main/README.md). 
-  Before running (for reproducing our results) the `transcribing.ipynb` be sure to have added you personal [hugging face](https://huggingface.co) token `YOUR_HF_TOKEN`: 
```bash
diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)
```
- To run the `transcribing.ipynb` notebook on [Google Colab](https://colab.research.google.com) as we did for more computational power, don't forget to run the following cell: 
```bash
# Import the Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```bash
pip install git+https://github.com/m-bain/whisperx.git
```

### Python Packages Setup : 
- For this project, in addition to `WhisperX` and the basics `pandas`, `numpy`, `matplotlib` and others packages, we used the following specific Python packages that can be directly used on [Google Colab](https://colab.research.google.com) or installed in a [conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html) environment locally :
1) `nltk`: natural language processing (NLP) for German language
2) `statsmodels.api`: negative binomial regression model
3) `scikit-learn`: machine learning models and functions

### Files Path Setup : 
- For each notebook, the only thing necessary for running it, is to add your base file path (from the Google Drive if running on [Google Colab](https://colab.research.google.com) or the local one) at the following command:
```bash
base_path = 'drive/MyDrive/...'  # on Google Colab
base_path = '.'                  # locally
```

## Files Input-Ouput Overview 🔍

| File name             | File Input Directory | File Input Description | File Output Directory | File Output Description | 
| --------------        | ----------     | ----------   | ---------- | ---------- |
| `transcribing.ipynb`     | `/wavs` | Audio file of each team meeting `.wav` | `/json`| Diarized transcript from *WhisperX* `.json`           
| `cleaning.ipynb`    | `/json`| Diarized transcript from *WhisperX*  `.json` | `/csv` | `transcripts_teams.csv`: Organized at the team level, this dataset is crafted for team-specific analyses. It provides the team identification numbers, initial transcripts, filtered transcripts, and final clean transcripts.       
| `cleaning.ipynb`    | `/json`| Diarized transcript from *WhisperX* `.json` | `/csv` | `transcripts_speakers.csv`: Organized at the speaker level is created for speaker-specific analyses. It offers a view of individual speaker contributions.
| `cleaning.ipynb`    | `/json`| Diarized transcript from *WhisperX* `.json` | `/csv` | `speaking_time.csv`: Provides a temporal perspective on team speakers continuous speaking duration.          
| `main.ipynb`    | `/csv` | `dataset.csv`, `transcripts_teams.csv`, `transcripts_speakers.csv`, `speaking_time.csv` | cell output | `WhisperX Benchmark`, `Negative Binomial Regression` Results and `Classification` Results
| `main.ipynb`    | `/jsons` | Diarized transcript from *Deepgram* `.json` | cell output | `Deepgram Benchmark` 
| `main.ipynb`    | `/rttms` | Diarized transcript from *Pyannote* `.rttm` | cell output | `Pyannote Benchmark`     


## Audio Transcription and Diarization `transcribing.ipynb`

> **_🚀 BEST PRACTICE:_**  If you want to reproduce the work done, the best practice is to run the `transcribing.ipynb` on [Google Colab](https://colab.research.google.com) as it needs some computational power.  

- We used approximately 50 units of computational power from [Google Colab](https://colab.research.google.com) with a [Tesla V100 GPU](https://colab.research.google.com/github/d2l-ai/d2l-tvm-colab/blob/master/chapter_gpu_schedules/arch.ipynb#scrollTo=PyGInfembT2s), and it took us approximately 5 hours to transcript and diarize the 116 audios files with `WhisperX`.

## Transcript Processing and Cleaning `processing.ipynb`

> **_📌 NOTE:_** This notebook takes approximately 2 minutes to run in its entirety, it can be done locally.

give the head of all the csv dataset

## Main Notebook Overview `main.ipynb`
> **_📌 NOTE:_** This notebook takes approximately 5 minutes to run in its entirety, it can be done locally.



just give the architecture of the notebook and see to check the regression.txt for the regression result and the notebbok outpur for the classification result
















