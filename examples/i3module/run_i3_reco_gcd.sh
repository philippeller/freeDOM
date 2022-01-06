if [ "$#" -ne 4 ]; then
    echo "Usage: run_i3_reco <cuda_device> <input_file_list> <output_dir> <gcd_file>"
    exit
fi

cuda_device=$1
input_file_list=$2
output_dir=$3
gcd_file=$4

if [ ! -d $output_dir ]; then
    echo "output directory $output_dir does not exist."
    exit
fi

service_ports=( 9887 9888 9889 9890 )
service_addr="tcp://127.0.0.1:${service_ports[$cuda_device]}"

echo "Starting the service"
llh_pids=()
python service_control.py --cuda_device $1 &
sleep 10

# keep track of PIDs so we wait only for the clients to finish
# and not the service
pids=()

#--n_frames 50
input_files=($(cat $input_file_list))
for file in ${input_files[@]}; do
    output_file=`basename $file`
    output_file=`readlink -m $output_dir/${output_file%.i3.zst}_reco.i3.zst`
    ./i3_reco.py --gcd_file $gcd_file --input_files $file --output_file $output_file --service_addr $service_addr &
    pids+=($!)
done

echo "Jobs launched"

for pid in ${pids[@]}; do
    wait $pid
done

echo "Jobs done. Killing the LLH services"
python service_control.py --cuda_device $cuda_device --kill &

wait
