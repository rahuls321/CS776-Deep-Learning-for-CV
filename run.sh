epochs=500
batch_size=128
learning_rate=0.01
dataset='./cifar-10-batches-py'
model_weights='./model_weights'
feat_vector='./feat_vector'
out_folder='./out_folder'
isModelWeightsAvailable=0


echo "Assignment 1 - DLCV"
CUDA_VISIBLE_DEVICES=1 python main.py -d $dataset -f $feat_vector -w $model_weights -o $out_folder -e $epochs -b $batch_size -l $learning_rate -m $isModelWeightsAvailable