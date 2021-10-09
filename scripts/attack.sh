# shellcheck disable=SC2034
# shellcheck disable=SC2068
# shellcheck disable=SC2006
# define run function
python_path=`which python`
run() {
    number=$1
    shift
    for i in $(seq $number); do
      $@
    done
}
# $1 defines the number will be repeat
# $2 defines the attack python file will be executed
run "$1" $python_path -W ignore $2