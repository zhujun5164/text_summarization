use limit word

python train_limit_vocab.py 
--folder_path=E:\data\test_for_UNILM 
--task_name=acc_mask 
--config_name=bert-base-chinese-config.json 
--vocab_name=bert-base-chinese-vocab.txt 
--model_name=bert-base-chinese-pytorch_model.bin
--use_UNILM
--use_summary_loss
--limit_vocab 
--limit_vocabulary_name=new_vocab_0.txt 
--limit_vocab_model_name=bert-0-chines-pytorch_model.bin 
--use_cuda


use base BERT

python train_limit_vocab.py 
--folder_path=E:\data\test_for_UNILM 
--task_name=BERT_summary_limit 
--config_name=bert-base-chinese-config.json 
--vocab_name=bert-base-chinese-vocab.txt 
--model_name=bert-base-chinese-pytorch_model.bin
--use_summary_loss
--limit_vocab 
--limit_vocabulary_name=new_vocab_0.txt 
--limit_vocab_model_name=bert-0-chines-pytorch_model.bin 
--use_cuda