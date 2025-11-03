# MISIS Neychev Loss

## Team:
1) [**Alexander Gruzdev**](https://github.com/gruzdev-as) - ML
2) [**Kirill Ryzhichkin**](https://github.com/l1ghtsource) - ML
3) [**Maksim Litvinov**](https://github.com/maksimlitvinov39kg) - ML
4) [**Maksim Aksenovskiy**](https://github.com/Solitum26) - ML
5) [**Arina Zamyshevskaya**](https://github.com/Nimbleredsquirrel) - ML

Presentation: [link](https://drive.google.com/file/d/1-L61ooSwS0bSlZIfT1Uk5YO9XHGkVnjc/view?usp=sharing)

## Solution Description

The final solution is an ensemble of gradient boosting models, trained using features generated both without deep learning models and with various modifications of the BERT model. For each test sample, two predictions are made:

1) A prediction from 5 models trained on the entire training dataset with different random seeds to mitigate the effect of randomness.
2) A prediction from 5 models trained on data from a specific second-level category with different random seeds.

The final predictions from the models are summed, averaged, and then blended with weights $w_{full} = 0.6$ and $w_{categories} = 0.4$ to get the final prediction.
Each gradient boosting model uses **109** features, of which **106** are generated from the available data without using machine learning models, as demonstrated in the [**generate_features_train.py**](generate_features_train.py) file. The remaining **3** features are Out-of-Fold (OOF) predictions obtained from fine-tuned open-source models.

![scheme](scheme.png)

### Used Open-Source Models

1) [**rubert-tiny-2**](https://huggingface.co/cointegrated/rubert-tiny2)
2) [**distilbert-base-multilingual-cased**](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)

The **rubert-tiny-2** model was used to generate OOF features from product attributes and descriptions, respectively. The **distilbert-base-multilingual-cased** model was used to generate OOF features from attributes only.

The code for training the models and obtaining OOF predictions can be found here: [**bert_training.py**](bert_training.py)

Model weights used for inference:
1) [**3epoch_768_name_attr_bert_full**](https://drive.google.com/file/d/1GEI0lEi1gitio-aKdn0fdAni-sHMhZlB/view?usp=drive_link)
2) [**3epoch_1024_name_desc_bert_full**](https://drive.google.com/file/d/1vMe_znzoKJjUZ7gRRTQDpbch_5Nx98e6/view?usp=drive_link)
3) [**multi512_attr_bert_full_second_epoch**](https://drive.google.com/file/d/1c9d03-pIwT5HJWfvEQ8PlxW5GtaEkuTB/view?usp=drive_link)

Gradient Boosting Models:
[Download link](https://drive.google.com/drive/folders/1mktUxSWbg1YQHZXdSjQyBoSqwlD2pNdl?usp=drive_link)

### Training the Final Models

The training procedure for the gradient boosting models is described in the [**train_boosting_models.py**](train_boosting_models.py) file for both the general model and the category-specific models.

## Final Pipeline for Reproducing the Solution

During the hackathon, the team used the following resources for model training and feature generation:
1) VK CLOUD SERVER, 128 Gb RAM, 16 CPU Intel Ice Lake;
2) Kaggle GPU accelerated environment (P100);
3) Team members' personal computers (PCs).

Therefore, to reproduce the full pipeline, it is recommended to ensure that your system has sufficient resources. For example, during feature generation on the training dataset, memory consumption can spike significantly, thus affecting the total time required.

*The team apologizes that the code is not perfectly optimized and that feature generation for the training set is time-consuming.*

To reproduce the solution:
1) Create a virtual environment, activate it, and install the required libraries.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2) Ensure that the repository contains a *data* folder with the following structure:
    ```
    /project-root-folder
    ├── data
    │   ├── train
    │   │   ├── train.parquet
    │   │   ├── attributes.parquet
    │   │   ├── text_and_bert.parquet
    │   │   ├── resnet.parquet
    │   ├── test
    │   │   ├── test.parquet
    │   │   ├── attributes_test.parquet
    │   │   ├── text_and_bert_test.parquet
    │   │   ├── resnet_test.parquet
    ```

3) Run the [**generate_features_train.py**](generate_features_train.py) file.
    1) This will start the feature generation process based on the training data (the [**generate_features_train.py**](generate_features_train.py) file will sequentially call [**data_preprocessing.py**](data_preprocessing.py) and [**feature_generation.py**](feature_generation.py)).
    2) The resulting parquet files with data will be saved in *data/train/*.

4) Run the [**bert_training.py**](bert_training.py) file.
    1) This will start the training of the open-source models.
    2) The model weights will be saved in *models/BERT/*.

5) Run the [**train_boosting_models.py**](train_boosting_models.py) file.
    1) This will start the script for training the gradient boosting models on the previously generated parquet files.
    2) The resulting models will be saved in *models/CATBOOST* (the general model) and *models/CATBOOST/categories* (the category-specific models).

6) Run the [**make_submission.py**](make_submission.py) file.
    1) A file named *submission.csv* will be created in the *data* folder, containing the predicted probabilities for each pair from *test.parquet*.

### Other Files, Folders, and Notes

1) For inference via the CI/CD system, the boosting model weights were pushed directly to the repository, while the weights of "heavy" models and trained vectorizers were downloaded during the Docker container build. Therefore, for inference, you can download the necessary weights and models directly. Also, for local inference of the open-source models, you need to either download their base pretrained versions manually from the links or use the following commands:
```bash
    huggingface-cli download cointegrated/rubert-tiny2 --local-dir='./models/basemodel/rubert' && \
    huggingface-cli download distilbert-base-multilingual-cased --local-dir='./models/basemodel/distilbert'
```
If the command fails, install `huggingface-cli`:
```bash
    pip install huggingface-hub
```

2) The *notebooks* folder contains notebooks that may have code snippets for feature training and/or model training that were not included in the final solution. They are provided for informational purposes only and may contain errors:

- [Exploratory Data Analysis](notebooks/0.%20Exploratory%20Data%20Analysis.ipynb) - Contains exploratory data analysis.
- [Preprocessing & Feature Generation for Train](notebooks/1.1%20Preprocessing%20&%20Feature%20Generation%20for%20Train.ipynb) - Generation of main features for the training set.
- [Preprocessing & Feature Generation for Test](notebooks/1.2%20Preprocessing%20&%20Feature%20Generation%20for%20Test.ipynb) - Generation of main features for the test set.
- [Bert (Name + Attrs) OOF Predictions](notebooks/2.1%20Bert%20(Name%20+%20Attrs)%20OOF%20Predictions.ipynb) - Obtaining OOF predictions from BERT on attributes.
- [Bert (Name + Attrs) Full Training](notebooks/2.2%20Bert%20(Name%20+%20Attrs)%20Full%20Training.ipynb) - Full training of the BERT model on attributes.
- [Bert (Name + Desc) OOF Predictions](notebooks/3.1%20Bert%20(Name%20+%20Desc)%20OOF%20Predictions.ipynb) - Obtaining OOF predictions from BERT on descriptions.
- [Bert (Name + Desc) Full Training](notebooks/3.2%20Bert%20(Name%20+%20Desc)%20Full%20Training.ipynb) - Full training of the BERT model on descriptions.
- [Bert Inference](notebooks/4.%20Bert%20Inference.ipynb) - Example of BERT inference.
- [[UNUSED] FastText Training](notebooks/%5BUNUSED%5D%20FastText%20Training.ipynb) - Training a FastText model on descriptions.
- [[UNUSED] Sampling from Train Dataset](notebooks/%5BUNUSED%5D%20Sampling%20from%20Train%20Dataset.ipynb) - Sampling a subset from the training data that is similar to the test set in its distribution of three-level categories.
- [[UNUSED] Siamese Bert (Name + Attrs) Training](notebooks/%5BUNUSED%5D%20Siamese%20Bert%20(Name%20+%20Attrs)%20Training.ipynb) - Training representations for product attributes using a Siamese neural network.
- [[UNUSED] Siamese Bert (Name + Desc) Training](notebooks/%5BUNUSED%5D%20Siamese%20Bert%20(Name%20+%20Desc)%20Training.ipynb) - Training representations for product descriptions using a Siamese neural network.
- [[UNUSED] Transitive Chains](notebooks/%5BUNUSED%5D%20Transitive%20Chains.ipynb) - Augmenting the training set using transitive chains.

## Running via Docker Container

It is possible to run the solution using Docker. To build the solution's image, run:

```bash
docker build . -t matching_object
```

After building the container, you can run inference using the command:

```bash
docker run -it --network none --shm-size 2G --name matching_object -v ./data:/app/data matching_object python make_submission.py
```

You can also run other .py files from the solution inside the Docker container, like so:

```bash
docker run -it --network none --shm-size 2G --name matching_object -v ./data:/app/data matching_object python {scriptname.py}
```

However, please note that because all weights and libraries are downloaded during the container build, this process can be time-consuming.
```
