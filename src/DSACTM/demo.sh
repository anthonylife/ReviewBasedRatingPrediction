#/bin/sh

make
#./main -d 0 -r True -tm 0
#./main -d 0 -r True -tm 1
#./main -d 0 -r True -tm 3
./main1 -d 0 -r True -tm 3


#valgrind --tool=memcheck --leak-check=full --show-reachable=yes --error-limit=no --log-file=valgrind.log ./demo.sh
