#THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=5000 python wtrans.py 13 263653 search_model_ch2en/params_e13_upd263653.npz $1 $2 $3
#THEANO_FLAGS=device=cpu,floatX=float32 python wtrans.py 13 263653 search_model_ch2en/params_e13_upd263653.npz $1 $2 $3
#THEANO_FLAGS=device=cpu,floatX=float32 python wtrans.py 13 263653 search_model_ch2en/params_e13_upd263653.npz $1 $2 $3
#THEANO_FLAGS=device=cpu,floatX=float32 python wtrans.py 13 263653 search_model_ch2en/params_e13_upd263653.npz $1 $2 $3
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 10 1
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 20 1
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 50 1
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 100 1
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 500 1
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python wtrans.py \
    --epoch $1 \
    --batch $2 \
    --model-name $3 \
    --search-mode $4 \
    --beam-size $5 \
    --use-norm $6\
    --use-batch $7\
    --use-score $8\
    --use-valid $9\
    --use-mv ${10}
