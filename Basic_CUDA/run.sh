#!/usr/bin/sh
user_id= 5481 # change to your ID

make datagen
./build/datagen $user_id

make all

# Loop through directories 0 to 9
for dir in {0..9}
do
    echo "Running test for directory $dir"
    ./build/main -e data/$dir/output.raw -i data/$dir/input0.raw,data/$dir/input1.raw -t vector
done
