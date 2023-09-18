# ICASSP-2024

## Install s3prl toolkit
```
conda create -n s3prl python=3.8 \
conda activate s3prl \
conda install -c conda-forge dtw-python==1.1.6 \
git clone https://github.com/s3prl/s3prl.git \
cd s3prl \
pip install -e ".[all]"
```

## Step 1: Add downstream task to s3prl toolkit

Move ```content_preserving_hubert``` or ```content_preserving_wavlm``` to ```s3prl/s3prl/downstream/``` to add as a task

## Step 2: Modify runner.py
As s3prl does not provide any layerwise control for fine-tuning, we need to modify the ```s3prl/downstream/runner.py``` to freeze the layers that we don't want to train.

Take the code snippet from ```runner_part_freeze_layers.py``` and put it to the ```runner.py``` inside ```_get_upstream_modules()``` function, after the model is loaded, i.e.  

```
model = Upstream(\
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        ).to(self.args.device)
> PASTE THE SCRIPT HERE (copied from ```runner_part_freeze_layers.py)
```

## Step 3: SCORE-Finetuning
```python3 run_downstream.py -m train -p /path_to_experiment -u hubert_base -d content_preserving_hubert -f -l -1``` \
or \
```python3 run_downstream.py -m train -p /path_to_experiment -u wavlm_base -d content_preserving_wavlm -f -l -1``` 

## Step 4: Evaluate the SCORE finetuned model on QbE,ASR, and PR for SUPERB benchmark
Download the needed data, set data paths etc for the respective tasks. More info at [S3PRL/SUPERB](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md) \
Comment the layer freezing part that has been added in Step 2 \
Add the code from ```runner_part_load_custom.py``` to the same place. Set path that has saved checkpoints from Step 3. \

### QbE
No training is required for this task. Please have a look at [S3PRL/SUPERB](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md) for more details.
Run DTW with specific layers, WavLM: wavlm_base, HuBERT: hubert_base \

Note: -1 stands for the last layer (i.e. 12th)

For the test set

```python3 run_downstream.py -m evaluate -t "test" -u hubert_base -l -1 -d quesst14_dtw -p /path_to_experiment ``` \
or \
```python3 run_downstream.py -m evaluate -t "test" -u wavlm_base -l -1 -d quesst14_dtw -p /path_to_experiment``` \ 

Scoring \
```
cd /your_path/quesst14Database/scoring/
bash ./score-TWV-Cnxe.sh /path_to_experiment groundtruth_quesst14_eval -10
```

### PR
Note: Make sure lr is 5.0e − 4 \
Training: \
```python3 run_downstream.py -p /path_to_experiment -m train -u hubert_base -d ctc -c downstream/ctc/libriphone.yaml``` \
or \
```python3 run_downstream.py -p /path_to_experiment -m train -u wavlm_base -d ctc -c downstream/ctc/libriphone.yaml```
Test: \
```python3 run_downstream.py -m evaluate -e /path_to_experiment/dev-best.ckpt```

### ASR
Training: \
```python3 run_downstream.py -p /path_to_experiment -m train -u hubert_base -d asr``` \
or \
```python3 run_downstream.py -p /path_to_experiment -m train -u wavlm_base -d asr ```\

Test: \
```python3 run_downstream.py -m evaluate -t "test-clean" -e /path_to_experiment/dev-clean-best.ckpt``` 
