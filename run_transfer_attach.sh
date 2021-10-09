# shellcheck disable=SC2006
py_path=`which python`

run() {
    number=$1
    shift
    for i in $(seq $number); do
      # shellcheck disable=SC2068
      $@
      $py_path transfer_based_attack.py
    done
}

# shellcheck disable=SC2046
# shellcheck disable=SC2006
#echo $epoch
run "$1"
