if [ "$#" -ne 3 ]; then
    echo "Usage: run_i3_reco <cuda_device> <input_file_list> <output_dir>"
    exit
fi

cuda_device=$1
input_file_list=$2
output_dir=$3

if [ ! -d $output_dir ]; then
    echo "output directory $output_dir does not exist."
    exit
fi

echo "Starting the service"
tmpfile=`mktemp`
python service_control.py --cuda_device $1 | tee $tmpfile &

while ((1)); do
    grep "tcp://" $tmpfile >/dev/null 2>&1 && break
    sleep 0.1
done
ctrl_addr=$(grep "tcp://" $tmpfile | sed "s/^.*tcp/tcp/")
echo "Service ready at $ctrl_addr"

# keep track of PIDs so we wait only for the clients to finish
# and not the service
pids=()

input_files=($(cat $input_file_list))
for file in ${input_files[@]}; do
    output_file=`basename $file`
    output_file=`readlink -m $output_dir/${output_file%.i3.zst}_reco.i3.zst`
    ./i3_reco.py --n_frames 12 --input_files $file --output_file $output_file --service_addr "$ctrl_addr" &
    pids+=($!)
done

echo "Jobs launched"

for pid in ${pids[@]}; do
    wait $pid
done

echo "Jobs done. Killing the LLH services"
python service_control.py --ctrl_addr "$ctrl_addr" --kill &

wait

rm $tmpfile
