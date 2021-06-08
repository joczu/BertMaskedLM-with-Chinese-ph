# ====== train
python train.py -d=./train_data/train --usegpu=True --device=1 --model_name="originmodel"

python train.py -d=./train_data/SougouTrain.txt -b=8 -e=5 --each_steps=100000 --usegpu=True --device=0,1 --model_name="Sougoumodel" --vocab_path=./vocab/SougouBertVocab.txt

python train.py -d=./train_data/SougouTrain.txt -b=32 -e=5 --each_steps=100000 --usegpu=True --device=0,1 --model_name="Sougoumodel" --vocab_path=./vocab/SougouBertVocab.txt --load_model=./model/Sougoumodel_epoch_2.bin

# ======# test
python test.py -d='./test/SougouTest.txt' --batch_size=4  --each_steps=10000 --vocab_path=./vocab/SougouBertVocab.txt  --load_model=./model/Sougoumodel_epoch_2.bin --log_path='./test/logger.txt'  # if we don't use gpu 

python test.py -d='./test/SougouTest.txt' --batch_size=4  --usegpu --each_steps=10000 --vocab_path=./vocab/SougouBertVocab.txt  --load_model=./model/Sougoumodel_epoch_2.bin # if we use gpu