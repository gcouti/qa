#!/usr/bin/env bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/;

SCRIPT=".env/bin/python -m experiment.qa_experiment" 
COMMON_PARAMETERS="model_default_cfg.verbose=True model_cfg.batch_size=512 model_cfg.epochs=500 dataset_default_cfg.reader=babi"
TASKS=(
  "model_default_cfg.model=pypagai.models.model_n2nmemory.N2NMemory"
  "model_default_cfg.model=qa.models.model_rn.ConvInputsRN dataset_default_cfg.strip_sentences=True dataset_cfg.only_supporting=True"
  "model_default_cfg.model=qa.models.model_rn.ConvRN dataset_default_cfg.strip_sentences=True dataset_cfg.only_supporting=True"
  "model_default_cfg.model=qa.models.model_rn.ConvStoryRN dataset_cfg.only_supporting=True  dataset_default_cfg.strip_sentences=True"
  "model_default_cfg.model=qa.models.model_rn.ConvQueryRN dataset_cfg.only_supporting=True dataset_default_cfg.strip_sentences=True"
  "model_default_cfg.model=pypagai.models.model_rn.RN dataset_cfg.only_supporting=True dataset_default_cfg.strip_sentences=True"
  "model_default_cfg.model=pypagai.models.model_encoder.EncoderModel"
  "model_default_cfg.model=pypagai.models.model_lstm.SimpleLSTM"
  "model_default_cfg.model=pypagai.models.model_lstm.EmbedLSTM"
  "model_default_cfg.model=pypagai.models.model_lstm.ConvLSTM model_cfg.batch_size=32"
  "model_default_cfg.model=pypagai.models.model_rnn.RNNModel"

  # Fix-it
  # "model_default_cfg.model=pypagai.models.model_dmn.DMN"

  # Memory problem
  # "model_default_cfg.model=qa.models.model_rn.RNNoLSTM dataset_default_cfg.strip_sentences=True dataset_cfg.only_supporting=True"
)


DATA_SETS=(
    "dataset_cfg.task=1"
    "dataset_cfg.task=2"
    "dataset_cfg.task=3"
    "dataset_cfg.task=4"
    "dataset_cfg.task=5"
    "dataset_cfg.task=6"
    "dataset_cfg.task=7"
    "dataset_cfg.task=8"
    "dataset_cfg.task=9"
    "dataset_cfg.task=10"
    "dataset_cfg.task=11"
    "dataset_cfg.task=12"
    "dataset_cfg.task=13"
    "dataset_cfg.task=14"
    "dataset_cfg.task=15"
    "dataset_cfg.task=16"
    "dataset_cfg.task=17"
    "dataset_cfg.task=18"
    "dataset_cfg.task=19"
    "dataset_cfg.task=20"
)

for t in "${TASKS[@]}"; do
    for d in "${DATA_SETS[@]}"; do
        COMMAND="$SCRIPT with $d $t $COMMON_PARAMETERS -n $t"

        echo "#########################################################################################################"
        echo "[START] Invoke new experiment $COMMAND"
        echo "#########################################################################################################"

        ${COMMAND}
        echo "#########################################################################################################"
        echo "[FINISH] $COMMAND"
        echo "#########################################################################################################"


    done
done
