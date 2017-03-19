Here I include:

* convert_to_wav.sh: bash script to convert your dataset of recordings to wavs
* specgram_gen.ipynb: Python 3 Jupyter Notebook to generate spectrogram and functions to obtain ROI of a spectrogram
* genDataset.ipynb: 
    - Python 3 Jupyter notebook to parse ROI .xlsl file and save dataset in yaml file  
    -  I'm including a sample output file: dataset.yaml
    - functions to create a dataset of species' ROI spectrogram from the dictionary contained in dataset.yaml
    - functions to serialize and package this data  

Requirement are:
* numpy
* matplotlib
* ffmpeg (convert_to_wav)
* openpyxl
* pyyaml

Run: 
`brew install ffmpeg`
or use your OS's package manager
