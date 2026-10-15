## CC-Time: Cross-Model and Cross-Modality Learning with LLMs for Time Series Forecasting


## Requirements
To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
<!-- You can access the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view), then place the downloaded contents under ./dataset. -->
You can access the well pre-processed datasets from Google Drive, then place the downloaded contents under ./dataset.


## Quick Demos
1. Download datasets and place them under ./datasets.
2. Create a folder called gpt2 and download the pre-trained parameters of GPT2 from hugging_face.
    ```
    mkdir gpt2
    python load_gpt2.py
    ```
3. Training and testing. Training and testing. Using the ETTm1 dataset as an example, run the following script:
    ```
    python main.py
    ```
