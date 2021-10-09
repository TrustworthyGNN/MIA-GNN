# shellcheck disable=SC2006
py_path=`which python`
start_epoch=$2
dataset=$3
run() {
    number=$1
    shift
    for i in $(seq $number); do
      # shellcheck disable=SC2068
      $@
      epoch=`expr $i \* $start_epoch`
      $py_path code/main_SPs_graph_classification.py --dataset $dataset --config 'configs/SPS/superpixels_graph_classification_GCN_'$dataset'_100k.json' --epochs $epoch
    done
}

# shellcheck disable=SC2046
# shellcheck disable=SC2006
#echo $epoch
run "$1"
