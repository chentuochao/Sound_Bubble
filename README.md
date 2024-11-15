# Sound bubbles on hearables
The source code for Nature Electronics Paper "Sound bubbles on hearables" [paper_link](https://www.nature.com/articles/s41928-024-01276-z.epdf?sharing_token=UBEcecaT-LOl28bRvGeUHNRgN0jAjWel9jnR3ZoTv0MI_xnp258ClQYuXnq4ROtijwv-jc3byDg5-F2vswnmmIlGNpSLsyxq4V72UEIWzmeKFbUI8XzLC8-DV5LB4nhalyrysnXenNHfMmw9RATWLqfYyBJys5frFCnFNmAiaU8%3D)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/VaGnfQcpopQ/0.jpg)](https://www.youtube.com/watch?v=VaGnfQcpopQ)

## Abstract 
The human auditory system has a limited ability to perceive distance and distinguish speakers in crowded settings. A headset technology that can create a sound bubble in which all speakers within the bubble are audible, but speakers and noise outside the bubble are suppressed, could augment human hearing. However, developing such technology is challenging. Here we report an intelligent headset system capable of creating sound bubbles. The system is based on real-time neural networks that use acoustic data from up to six microphones integrated into noise-cancelling headsets and are run on-device, processing 8 ms audio chunks in 6.36 ms on an embedded central processing unit. Our neural networks can generate sound bubbles with programmable radii between 1 and 2 meters, and with output signals that reduce the intensity of sounds outside the bubble by 49 decibels. With previously unseen environments and wearers, our system can focus on up to two speakers within the bubble with one to two interfering speakers and noise outside the bubble.


## Code structure

### datasets
The files for dataset splitting. 

### src 
It includes the training and testing scripts for our model. 

### edge 
Convert the pytorch model to ONNX model

### helpers
Some utility functions

###  real_experiments
Experiment/Model configuration files for real-world experiment

### syn_experiments
Experiment/Model configuration files for synthetic experiment

### generate_adaptive_dataset.py
The script to generate the synthetic dataset


## Running instruction

### Setting up
Please note that most scripts require you to use a GPU with CUDA capabilities. We cannot guarantee the scripts will work on CPU out-of-the-box. 

If this is your first time running the code, create the environment and install the required modules.

```
conda create --name speech_bubble python=3.8
conda activate speech_bubble
pip install -r requirements2.txt
```
After initial setup, source.sh should activate the environment and add the working directory to your python path. (May not work on Windows/Mac, instead, simply copy and paste the commands inside into the terminal).
```
source setup.sh
``` 

### Test some examples
Run the code to test some synthetic samples for differennt bubble sizes:
```
python src/test_samples.py ./test_samples/syn_1m/ ./TFG_S_big_newdis_v3_pt_fix_MutiLoss/  --distance_threshold 1 --use_cuda
python src/test_samples.py ./test_samples/syn_1_5m/ ./TFG_S_big_newdis_v3_pt_fix_MutiLoss/  --distance_threshold 1.5 --use_cuda
python src/test_samples.py ./test_samples/syn_2m/ ./TFG_S_big_newdis_v3_pt_fix_MutiLoss/  --distance_threshold 2 --use_cuda
```

### Synthentic dataset generation
Download the [VCTK](https://datashare.ed.ac.uk/handle/10283/2651), [LIBRITTS](http://www.openslr.org/60), and [WHAM!](http://wham.whisper.ai) dataset. The splitting files (./datasets/WHAM_split.json and ./datasets/vctk_split.json) split the VCTK and WHAM! into non-overlapped training, validation and testing sets. The LibriTTS is already split. Then to generate our synthetic dataset, run
```
python generate_adaptive_dataset.py VCTK_DATASET_DIRECTORY FOLDER_TO_SAVE --dis_threshold bubble_size --n_outputs_test 2000 --n_outputs_train 10000 --n_outputs_val 2000 --seed 12 --bg_voice_dir WHAM_DATASET_DIRECTORY --tts_dir LIBRITTS_DATASET_DIRECTORY
```

### Model training 
To train the our model, we need (1) unzip the dataset we provide and specify the dataset path in the experiment json file (2) prepare the experiment json files for training configuration (example in ./syn_experiments and ./real_experiments) (3)  train with the following scripts:
```
python src/train_pt.py --run_dir DIRECTORY_TO_SAVE_MODEL --config PATH_TO_CONFIG_FILE 
```

### Model testing 
First unzip the test dataset, then run the testing script.
For synthetic data testing,
```
python src/eval_syn.py TEST_SET_PATH DIRECTORY_TO_SAVE_MODEL DIRECTORY_TO_SAVE_RESULT --use_cuda 
```

For real-world data testing,
```
python src/eval.py TEST_SET_PATH DIRECTORY_TO_SAVE_MODEL DIRECTORY_TO_SAVE_RESULT --use_cuda 
```
