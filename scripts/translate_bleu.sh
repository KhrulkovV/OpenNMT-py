MODEL=$(ls model/_step_* |sort -n -t _ -k 3 | tail -1)


CUDA_VISIBLE_DEVICES=7 python OpenNMT-py/translate.py -gpu 0 -replace_unk -alpha 0.6 -beta 0.0 -beam_size 5 -length_penalty wu -coverage_penalty wu -model $MODEL -src data news-commentary-v11.de-en.en.sp —output data/prediction.txt -verbose


spm_decode --model data/wmtende.model data/news-commentary-v11.de-en.de.sp --output data/news-commentary-v11.de-en.de.detok
spm_decode --model data/wmtende.model data/prediction.txt --output data/prediction.detok


perl OpenNMT-py/tools/multi-bleu-detok.perl data/news-commentary-v11.de-en.de.detok < data/prediction.detok
