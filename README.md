# Two-Tower Model and DIN model(without Dice) use TF-estimator API at Amazon Electronics dataset

| batch_size| Model | max-AUC|
| ------ | ------ | ------ |
|32 |Two-Tower|0.877 | 
|32 |DIN(without Dice)| 0.893|

## Requirements
* Python 3.6
* Numpy 1.18.5
* Pandas 1.1.3
* TensorFlow 2.3.1

## Amazon Electronics dataset download 
```sh
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz  
gzip -d reviews_Electronics_5.json.gz  
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz  
gzip -d meta_Electronics.json.gz
```  

## Training and Evaluation
* Step 1: generate tfrecord dataset
```sh
python generate_tfrecord.py
```
or use spark to generate tfrecord  
(I use zeppelin and code save in generate_tfrecord.scala)
* Step 2: training and evaluation
```sh
python main.py
```
you need confirm tfrecord dataset path and param "data_gen_method"("spark" or "python")  
* you can change Two-Tower model to DIN model in main.py's model_fn

## TFServing
```
docker pull tensorflow/serving:1.15.0
docker run -t --rm -p 8501:8501 -v xxx/saved_model/:/models/test-model \
  -e MODEL_NAME=test-model tensorflow/serving:1.15.0 &
```

## Reference: 
https://github.com/zhougr1993/DeepInterestNetwork.git  
https://zhuanlan.zhihu.com/p/129700488  
https://github.com/cdj0311/two_tower_recommendation_system.git
