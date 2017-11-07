python create_fasttext_inputs.py
cd fastText/

./fasttext skipgram -input ../data/fasttext-inputs.txt -output skip100_min1_neg15_ws5_epoch7  -minCount 1 -neg 15 -ws 5  -epoch 7 -dim 100
./fasttext skipgram -input ../data/fasttext-inputs.txt -output skip100_min1_neg15_ws20_epoch7 -minCount 1 -neg 15 -ws 20 -epoch 7 -dim 100
./fasttext cbow -input     ../data/fasttext-inputs.txt -output cbow100_min1_neg15_ws5_epoch7  -minCount 1 -neg 15 -ws 5  -epoch 7 -dim 100
./fasttext cbow -input     ../data/fasttext-inputs.txt -output cbow100_min1_neg15_ws20_epoch7 -minCount 1 -neg 15 -ws 20 -epoch 7 -dim 100

rm *.bin
