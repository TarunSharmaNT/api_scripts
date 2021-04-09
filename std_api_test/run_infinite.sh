COUNT=0
iter=100
while [ $COUNT -lt $iter ]; do
    echo processing..
    echo count $iter
    /root/anaconda3/envs/pytorch17_102/bin/python defect_api_test_akhil.py $COUNT  
    let COUNT=COUNT+1
done 

