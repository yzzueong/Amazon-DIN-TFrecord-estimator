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
docker pull tensorflow/serving

docker run -t --rm -p 8501:8501 \
    -v  xxx/saved_model/:/models/model \
    -e MODEL_NAME=model \
    tensorflow/serving &

curl localhost:8501/v1/models/model/metadata

curl -X POST -i 'http://localhost:8501/v1/models/model:predict' --data '
{
  "signature_name":  "predict",
  "instances": [
    {
      "hist" :[48286, 12353, 49240, 26542, 47881, 44980, 36691, 54084, 9091, 60606, 24126, 42664, 8622, 2845, 28137, 21689, 9790, 3742, 46807, 25980, 416, 9358, 53047, 38638, 33061, 29065, 34054, 5383, 41023, 50676, 18606, 20428, 18530, 7092, 41364, 13874, 35322, 28369, 60598, 16183],
      "hist_cate": [744, 183, 20, 504, 611, 401, 614, 42, 757, 171, 286, 643, 50, 737, 456, 797, 280, 245, 612, 596, 68, 265, 537, 633, 161, 45, 391, 124, 470, 310, 699, 104, 446, 398, 466, 318, 1, 758, 305, 687],
      "item_id": [49240],
      "pad_category_ids": [3]
    }
  ]
}'
```

## Reference: 
https://github.com/zhougr1993/DeepInterestNetwork.git  
https://zhuanlan.zhihu.com/p/129700488  
https://github.com/cdj0311/two_tower_recommendation_system.git  
https://github.com/tensorflow/serving.git
