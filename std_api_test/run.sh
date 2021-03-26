COUNT=0
iter=1
start_index=0
end_index=2500
while [ $COUNT -lt $iter ]; do

    echo processing.. $iter
    echo $start_index - $end_index
    /home/tarun/anaconda3/envs/pytorch17_102/bin/python testing_cars24_with_std_api.py $COUNT $start_index $end_index 
    let COUNT=COUNT+1
    let start_index=end_index
    let end_index=end_index+2500
done 

