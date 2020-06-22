# Defense-on-adv-Mal-JS
This work implement three defense strategies, adversarial training, randomized smoothing, and L1 regularization, against JS adversarial examples with Bag of words model.

## User Guide
Please follow this guide to utilize the code. </br>
</br>
**STEP 1**: Download naive and malicious Javascript from any database (e.g. from https://github.com/HynekPetrak/javascript-malware-collection)
</br>
**STEP 2**: Make two folder, good_script and mal_script, to save those JS data, respectively. </br>
```console
$mkdir good_script
$mkdir mal_script
```
**STEP 3**: (Preprocessing) Generate RAW data by 
```console
$python3 RAWgenerator.py
```
**STEP 4**: (Preprocessing) Generate Bag of words data by 
```console
$python3 BOWgenerator.py
```
**STEP 5**: Train the basic model by
```console
$python3 BASICtrain.py
```
**STEP 6**: Select a good model from ./Model/ folder. You can select it based on the plot of training result, which would be generated during training.</br>
**STEP 7**: Make a folder name "Adv_example" and generate the adversarial examples by
```console
$mkdir Adv_example
$python3 ADVgenerator.py -M model_path
```
**STEP 8**: Train the models with adversarial training by 
```console
$python3 ADVtrain.py
```
**STEP 9**: Train the models with L1 regularization by 
```console
$python3 L1train.py
```
**STEP 10**: Train the models with randomized smoothing by 
```console
$python3 GAUtrain.py
```
**STEP 11**: Finally, attack each model with all the adversarial examples generated perviously to demonstrate how robust the models trained by different strategies are. 
```console
$python3 MUTUALattak.py
```
**STEP 12**: You can inspect all the results by the plot automatically generated during mutually attacking.
