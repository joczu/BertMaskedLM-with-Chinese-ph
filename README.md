# BERT_MASKED_LanguageModel

# 描述
该模型基于Transformers中的BertForMaskLM , 该模型通过jieba分词，下载了搜狗新闻数据，使用结巴分词对语料进行按词切分，然后使用Bert模型进行训练，依据bert模型的mask规则，一个语料数据中，除去特殊词汇，'[CLS]','[SEP]','[UNK]','[PAD]'以外，对剩下的（即能够有效利用）的词汇进行mask，mask的规则是15%的概率被mask，而对于所有的被mask的词汇，采用下述规则\
- 80%的词汇使用特殊token[MASK]
- 10%的词汇使用词典中另一词随机替代
- 10%的词汇保留原词\
基于这样的规则，我们建立了此项目。

# 训练
训练的方法已经放在了文件howuse.sh中，可以使用python train.py -d=./train_data/SougouTrain.txt -b=32 -e=5 --each_steps=100000 --usegpu=True --device=0,1 --model_name="Sougoumodel" --vocab_path=./vocab/SougouBertVocab.txt --load_model=./model/Sougoumodel_epoch_2.bin\
进行训练，先对一些参数进行说明:
- '-d', '--train_data_path', type=str, default="./train_data/train", help='training data path'
- '-s', '--model_save_path', type=str, default='./model', help='model file save path '
- '-v', '--vocab_path', type=str, default='./vocab/vocab.txt', help='vocab file path'
- '-b', '--batch_size', type=int, default=4, help='batch_size'
- '-e', '--epoch', type=int, default=2, help='epoch'
- '--each_steps', type=int, default=100, help='save model after train N steps '
- '--model_name', type=str, help='model name'
- '--usegpu', action='store_true',default=False, help="use gpu or not"
- '--device', type=str, default=0
- '--load_model', type=str, default=''
- '--curepoch',type=int, help="what epoch you want to begin train model"
- '--curstep',type=int, help="where step you want to begin train model"\
根据我的经验，使用双卡（皆为RTX 2080Ti），训练全部搜狗语料一轮（batch=32）需要22小时。特别解释一下`curepoch`参数，如果你之前训练了一个model_epoch2，表示训练了三个epoch的模型（0，1，2），现在要训练第4个epoch模型，那么curepoch应该设置为3.curstep同理，如果之前有一个训练好的model_step_10000，现在要从10000开始继续训练，那么应该设置curstep=10000。但是由于data处理的某些问题，curstep的功能目前有问题，不建议使用。

# 推理
推理使用`inference.py`文件，执行 `python inference.py --mode=1 --model=./model/Sougoumodel_epoch_2.bin`对其中的参数进行说明:
- "-m","--mode",type=int,default=1,help="模式为0，表示使用随机mask进行推断，模式为1，表示使用指定mask进行推断"
- "-p","--p_mask",type=float,default=0.15,help="当mode=0时,此参数生效，用来表示随机生成mask的概率"-
- "--model",type=str,help="载入训练好的模型"

# 测试
测试使用'test.py',执行`python test.py -d='./test/SougouTest.txt' --batch_size=4  --usegpu --each_steps=10000 --vocab_path=./vocab/SougouBertVocab.txt  --load_model=./model/Sougoumodel_epoch_2.bin`,对参数进行说明：
- '-d', '--test_data_path', type=str, default="./test/SougouTest.txt", help='test data path'
- '-v', '--vocab_path', type=str, default='./vocab/SougouBertVocab.txt', help='vocab file path'
- '-b', '--batch_size', type=int, default=4, help='batch_size'
- '--each_steps', type=int, default=100, help='show accuracy after train N steps '
- '--usegpu', action='store_true',default=False, help='use GPU or not'
- '--device', type=str, default=1
- '--load_model', type=str, default=''
- '--log_path', type=str, default=''\
这里的`load_model`需要载入训练好的模型，是必要参数。`log_path`是用来保存文件的输入日志的，它可以记录这次测试精确率，时间等信息。

# 词汇
使用的词汇见文件'./vocab/SougouBertVocab.txt' 它的长度达到68000+，我们统计了Sougou语料的词汇，过滤词频小于6*1E-7的词，剩下的词汇在于BertChinese的词汇取并集而得到。

# 模型推理效果展示

**shizi.zhuang@shizi8-7f7b8cd57d-tkpw5:~/Bert/languagemodel$ python inference.py --mode=1 --model=./model/Sougoumodel_epoch_2.bin**\
`输入:  tensor([[  101, 22964,  8492,  1962,   103,   102]])`\
`真实:  共产党 就是 好 [MASK]`\
`预测:  [['的', '了', '朋友', '同志', '人']]`\
`输入:  tensor([[  101,   103,   100, 59849, 12655,   100,  8436,   102]])`\
`真实:  [MASK] 就像 园丁 一样 抚育着 学生`\
`预测:  [['“', '像', '与', '这些', '但']]`\
`输入:  tensor([[ 101,  100, 2397,  103,  102]])`\
`真实:  你想要 干 [MASK]`\
`预测:  [['露露', '啥', '了', '什么', '下来']]`\
`输入:  tensor([[ 101, 8102,  103,  102]])`\
`真实:  中国 [MASK]`\
`预测:  [['认为', '代表团', '媒体报道', '经济网', '证券报']]`\
`输入:  tensor([[  101, 38276,   103,   102]])`\
`真实:  涂鸦 [MASK]`\
`预测:  [['批评', '文字', '印', '举动', '滋生']]`\
`输入:  tensor([[  101, 11742,   103,   102]])`\
`真实:  人工 [MASK]`\
`预测:  [['老窖', '抽检', '增雨', '改造', '为例']]`\
`输入:  tensor([[ 101, 2769, 4263,  103,  102]])`\
`真实:  我 爱 [MASK]`\
`预测:  [['你', '吗', '我', '什么', '她']]`\
