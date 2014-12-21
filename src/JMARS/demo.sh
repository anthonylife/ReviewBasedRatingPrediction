#/bin/sh

make train_target=jmars
if [ $? -ne 0 ]; then
    exit 1
fi

#echo "mfwdcatm --> Yelp"
#./run -d 0 -r True -tm 0

echo "mfwdcatm --> AmazonFood"
./run -d 1 -r True -tm 0

#echo "mfwdcatm --> Arts"
#./run -d 3 -r True -tm 0

#echo "mfwdcatm --> Sports"
#./run -d 5 -r True -tm 0

#valgrind --tool=memcheck --leak-check=full --show-reachable=yes --error-limit=no --log-file=valgrind.log ./demo.sh
