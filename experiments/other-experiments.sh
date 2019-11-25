#!/usr/bin/env bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/;

SCRIPT=".env/bin/python -m experiment.qa_experiment" 
COMMON_PARAMETERS="model_default_cfg.verbose=True model_cfg.batch_size=512 model_cfg.epochs=1"
MODELS=(
  "model_default_cfg.model=pypagai.models.model_n2nmemory.N2NMemory"
  "model_default_cfg.model=pypagai.models.model_encoder.EncoderModel"
  "model_default_cfg.model=pypagai.models.model_lstm.SimpleLSTM"
  "model_default_cfg.model=pypagai.models.model_lstm.EmbedLSTM"
  "model_default_cfg.model=pypagai.models.model_rnn.RNNModel"
  "model_default_cfg.model=pypagai.models.model_rn.RN"
  "model_default_cfg.model=qa.models.model_model_deepn2nmemory.DeepN2NMemory"


  # Fix-it
  # "model_default_cfg.model=pypagai.models.model_dmn.DMN"

  # Memory problem
  # "model_default_cfg.model=qa.models.model_rn.RNNoLSTM dataset_default_cfg.strip_sentences=True dataset_cfg.only_supporting=True"
)


DATA_SETS=(
  "dataset_default_cfg.reader=cbt dataset_cfg.task=V"
  "dataset_default_cfg.reader=cbt dataset_cfg.task=CN"
  "dataset_default_cfg.reader=cbt dataset_cfg.task=NE"
  "dataset_default_cfg.reader=cbt dataset_cfg.task=P"
)

for m in "${MODELS[@]}"; do
    for d in "${DATA_SETS[@]}"; do
        COMMAND="$SCRIPT with $m $d $COMMON_PARAMETERS -n $m"

        echo "#########################################################################################################"
        echo "[START] Invoke new experiment $COMMAND"
        echo "#########################################################################################################"

        ${COMMAND}
        echo "#########################################################################################################"
        echo "[FINISH] $COMMAND"
        echo "#########################################################################################################"


    done
done
