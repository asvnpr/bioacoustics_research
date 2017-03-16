Here I include:

* convert_to_wav.sh: bash script to convert your dataset of recordings to wavs
* specgram_gen.ipynb: Python 3 Jupyter Notebook to generate spectrogram and functions to obtain ROI of a spectrogram
* genDataset.ipynb: 
    - Python 3 Jupyter notebook to parse ROI .xlsl file and save dataset in yaml file  
    -  I'm including two sample output files: dataset.yaml, and simplified dataset.yaml
Requirement are:
* numpy
* matplotlib
* ffmpeg (convert_to_wav)
* openpyxl
* pyyaml

Run: 
`brew install ffmpeg`
or use your OS's package manager
