# ====== train
python train.py -d=./train_data/train --usegpu=True --device=1 --model_name="originmodel"

python train.py -d=./train_data/SougouTrain.txt -b=8 -e=5 --each_steps=100000 --usegpu --device=0,1 --model_name="Sougoumodel" --vocab_path=./vocab/SougouBertVocab.txt

python train.py -d=./train_data/SougouTrain.txt -b=32 -e=5 --each_steps=100000 --usegpu --device=0,1 --model_name="Sougoumodel" --vocab_path=./vocab/SougouBertVocab.txt --load_model=./model/Sougoumodel_epoch_2.bin

# ====== finetune by jobdata.txt
python train.py -d=./train_data/jobdata_train.txt -b=16 -e=10 --each_steps=100000 --usegpu --device=0,1 --model_name="Jobdatamodel" --vocab_path=./vocab/SougouBertVocab.txt --load_model=./model/Sougoumodel_epoch_9.bin  --each_steps=1000 --log_path=./train_data/job_logger.txt


# ======# test
python test.py -d='./test/SougouTest.txt' --batch_size=4  --each_steps=10000 --vocab_path=./vocab/SougouBertVocab.txt  --load_model=./model/Sougoumodel_epoch_2.bin --log_path='./test/logger.txt'  # if we don't use gpu 

python test.py -d='./test/SougouTest.txt' --batch_size=32  --usegpu --each_steps=100 --vocab_path=./vocab/SougouBertVocab.txt  --load_model=./model/Sougoumodel_epoch_9.bin  --device=0,1  --log_path=./test/logger.txt  --perplexity=./para_ph.npz   # if we use gpu

python test.py -d='./test/jobdata_test.txt' --batch_size=16  --usegpu --each_steps=10 --vocab_path=./vocab/SougouBertVocab.txt  --load_model=./model/Jobdatamodel_epoch_9.bin  --device=0,1  --log_path=./test/jobdata_logger.txt --perplexity=./para_ph.npz

python test.py -d='./test/jobdata_test_jiebal5.txt' --batch_size=2  --usegpu --each_steps=10 --vocab_path=./vocab/SougouBertVocab.txt  --load_model=./model/Jobdatamodel_epoch_9.bin  --device=0,1  --log_path=./test/jobdata_logger.txt --perplexity=./para_ph.npz --lastone_token

# =======# inference
python inference.py --mode=1 --model=./model/Sougoumodel_epoch_9.bin
python inference.py --mode=1 --model=./model/Jobdatamodel_epoch_9.bin
