#!/bin/sh

./libmf convert ../../data/yelp_train_triple.dat ./yelp_train.bin
./libmf convert ../../data/yelp_test_triple.dat ./yelp_test.bin
./libmf train --tr-rmse --obj -k 10 -t 200 -s 1 -p 0.1 -q 0.01 -g 0.01 -blk 1x1 -ub 1 -ib 1 --no-use-avg --no-rand-shuffle ./yelp_train.bin ./yelp_svd.model
./libmf predict ./yelp_test.bin ./yelp_svd.model ../../results/svd_result1.dat
#awk -F" " '{print $1, $2}' ../../data/yelp_test_triple.dat > ../../data/tmp_yelp_uid_pid.dat
#paste ../../data/tmp_yelp_uid_pid.dat ../../results/tmp_svd_result1.dat -d' ' > ../../results/svd_result1.dat
#rm ../../results/tmp_svd_result1.dat
#rm ../../data/tmp_yelp_uid_pid.dat

#./libmf convert ../../data/amazonfood_train_triple.dat ./amazonfood_train.bin
#./libmf convert ../../data/amazonfood_test_triple.dat ./amazonfood_test.bin
#./libmf train --tr-rmse --obj -k 20 -t 200 -s 1 -p 0.1 -q 0.1 -g 0.03 -blk 1x1 -ub 0.1 -ib 0.1 --no-use-avg --no-rand-shuffle ./amazonfood_train.bin ./amazonfood_svd.model
#./libmf predict ./amazonfood_test.bin ./amazonfood_svd.model ../../results/svd_result2.dat
#awk -F" " '{print $1, $2}' ../../data/amazonfood_test_triple.dat > ../../data/tmp_amazonfood_uid_pid.dat
#paste ../../data/tmp_amazonfood_uid_pid.dat ../../results/tmp_svd_result2.dat -d' ' > ../../results/svd_result2.dat
#rm ../../results/tmp_svd_result2.dat
#rm ../../data/tmp_amazonfood_uid_pid.dat
