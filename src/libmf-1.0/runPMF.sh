#!/bin/sh

#./libmf convert ../../data/yelp_train_triple.dat ./yelp_train.bin
#./libmf convert ../../data/yelp_test_triple.dat ./yelp_test.bin
#./libmf train --tr-rmse --obj -k 40 -t 400 -s 1 -p 0.1 -q 0.1 -g 0.001 -blk 1x1 -ub -1 -ib -1 --no-user-bias --no-item-bias --no-use-avg --no-rand-shuffle ./yelp_train.bin ./yelp_pmf.model
#./libmf predict ./yelp_test.bin ./yelp_pmf.model ../../results/pmf_result1.dat
#awk -F" " '{print $1,$2}' ../../data/yelp_test_triple.dat > ../../data/tmp_yelp_uid_pid.dat
#paste ../../data/tmp_yelp_uid_pid.dat ../../results/tmp_pmf_result1.dat -d' ' > ../../results/pmf_result1.dat
#rm ../../results/tmp_pmf_result1.dat
#rm ../../data/tmp_yelp_uid_pid.dat

./libmf convert ../../data/amazonfood_train_triple.dat ./amazonfood_train.bin
./libmf convert ../../data/amazonfood_test_triple.dat ./amazonfood_test.bin
./libmf train --tr-rmse --obj -k 40 -t 400 -s 1 -p 0.1 -q 0.1 -g 0.01 -blk 1x1 -ub -1 -ib -1 --no-user-bias --no-item-bias --no-use-avg --no-rand-shuffle ./amazonfood_train.bin ./amazonfood_pmf.model
./libmf predict ./amazonfood_test.bin ./amazonfood_pmf.model ../../results/pmf_result2.dat
#awk -F" " '{print $1,$2}' ../../data/amazonfood_test_triple.dat > ../../data/tmp_amazonfood_uid_pid.dat
#paste ../../data/tmp_amazonfood_uid_pid.dat ../../results/tmp_pmf_result2.dat -d' ' > ../../results/pmf_result2.dat
#rm ../../results/tmp_pmf_result2.dat
#rm ../../data/tmp_amazonfood_uid_pid.dat
