python create_fasttext_inputs.py
cd fastText/

./fasttext skipgram -input ../data/fasttext-inputs.txt -output skip100_min1_neg15_ws10_epoch7 -minCount 1  -neg 15 -ws 10 -epoch 7 -dim 100
./fasttext skipgram -input ../data/fasttext-inputs.txt -output skip20_min2_neg15_ws20_epoch7  -minCount 2  -neg 15 -ws 20 -epoch 7 -dim 20
./fasttext cbow -input     ../data/fasttext-inputs.txt -output cbow100_min1_neg15_ws10_epoch7 -minCount 1  -neg 15 -ws 10 -epoch 7 -dim 100
./fasttext cbow -input ../data/fasttext-inputs.txt -output cbow100_min2_neg15_epoch_7_ws_20 -minCount 2 -neg 15 -ws 20 -epoch 7 -dim 100

rm skip100_min1_neg15_ws10_epoch7.bin skip20_min2_neg15_ws20_epoch7.bin cbow100_min1_neg15_ws10_epoch7.bin
