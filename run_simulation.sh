
num_steps=50

for i in {1..10}
do
  /Users/sarinali/anaconda3/bin/python /Users/sarinali/Projects/VectorAdam/demo1.py $i $num_steps >> output_readable.txt 2>> loss_logs.txt
done
