#/bin/sh

make train_target=cmr
if [ $? -ne 0 ]; then
    exit 1
fi

#echo "CMR --> Yelp"
#./run -d 0 -r True -tm 0

echo "CMR --> AmazonFood"
./run -d 1 -r True -tm 0

#echo "CMR --> Arts"
#./run -d 3 -r True -tm 0

#echo "CMR --> Sports"
#./run -d 5 -r True -tm 0

#valgrind --tool=memcheck --leak-check=full --show-reachable=yes --error-limit=no --log-file=valgrind.log ./demo.sh
