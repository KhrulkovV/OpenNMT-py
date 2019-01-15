apt-get update
apt-get install cmake pkg-config libprotobuf9v5 protobuf-compiler libprotobuf-dev libgoogle-perftools-dev
apt-get install libprotobuf9v5

git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
make install
ldconfig -v

cd ../..


mkdir data
bash OpenNMT-py/scripts/prepare_data.sh raw_data

python OpenNMT-py/preprocess.py -train_src data/train.en
    -train_tgt data/train.de
    -valid_src data/valid.en -valid_tgt data/valid.de 
    -save_data data/processed/
    -src_seq_length 100 -tgt_seq_length 100 
    -max_shard_size 200000000 -share_vocab:q

