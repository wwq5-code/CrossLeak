#!/bin/bash

set -u

MAX_JOBS=4
pids=()
logs=()

wait_for_slot() {
    while (( ${#pids[@]} >= MAX_JOBS )); do
        local next_pids=()
        local next_logs=()
        local freed=0
        local i

        for i in "${!pids[@]}"; do
            if kill -0 "${pids[$i]}" 2>/dev/null; then
                next_pids+=("${pids[$i]}")
                next_logs+=("${logs[$i]}")
            else
                if ! wait "${pids[$i]}"; then
                    echo "Task failed: ${logs[$i]}" >&2
                fi
                freed=1
            fi
        done

        pids=("${next_pids[@]}")
        logs=("${next_logs[@]}")

        if (( freed == 0 )); then
            sleep 5
        fi
    done
}

run_task() {
    local log_file="$1"
    shift
    wait_for_slot
    echo "$log_file"
    (
        "$@"
    ) > "$log_file" 2>&1 &
    pids+=("$!")
    logs+=("$log_file")
}

wait_for_all() {
    local i

    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "Task failed: ${logs[$i]}" >&2
        fi
    done
}


run_task "On_CIFAR10/IB_FL_local_unlearn_retrain_630_range5" \
    env CIFAR10_retrain_Unlearning_Class_Range=5 \
    python On_CIFAR10/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR10/IB_FL_local_unlearn_retrain_630_range5_ratio_02" \
    env CIFAR10_retrain_Unlearning_Class_Range=5 CIFAR10_retrain_Unlearning_Ratio=0.02 \
    python On_CIFAR10/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR10/IB_FL_local_unlearn_retrain_630_range5_ratio_03" \
    env CIFAR10_retrain_Unlearning_Class_Range=5 CIFAR10_retrain_Unlearning_Ratio=0.03 \
    python On_CIFAR10/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR10/IB_FL_local_unlearn_retrain_630_range5_ratio_04" \
    env CIFAR10_retrain_Unlearning_Class_Range=5 CIFAR10_retrain_Unlearning_Ratio=0.04 \
    python On_CIFAR10/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR10/IB_FL_local_unlearn_retrain_630_range5_ratio_05" \
    env CIFAR10_retrain_Unlearning_Class_Range=5 CIFAR10_retrain_Unlearning_Ratio=0.05 \
    python On_CIFAR10/IB_FL_local_unlearn_retrain.py





run_task "On_CIFAR100/IB_FL_local_unlearn_retrain_630_range5" \
    env CIFAR100_retrain_Unlearning_Class_Range=5 \
    python On_CIFAR100/IB_FL_local_unlearn_retrain.py



run_task "On_CIFAR100/IB_FL_local_unlearn_retrain_630_range5_ratio_02" \
    env CIFAR100_retrain_Unlearning_Class_Range=5 CIFAR100_retrain_Unlearning_Ratio=0.02 \
    python On_CIFAR100/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR100/IB_FL_local_unlearn_retrain_630_range5_ratio_03" \
    env CIFAR100_retrain_Unlearning_Class_Range=5 CIFAR100_retrain_Unlearning_Ratio=0.03 \
    python On_CIFAR100/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR100/IB_FL_local_unlearn_retrain_630_range5_ratio_04" \
    env CIFAR100_retrain_Unlearning_Class_Range=5 CIFAR100_retrain_Unlearning_Ratio=0.04 \
    python On_CIFAR100/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR100/IB_FL_local_unlearn_retrain_630_range5_ratio_05" \
    env CIFAR100_retrain_Unlearning_Class_Range=5 CIFAR100_retrain_Unlearning_Ratio=0.05 \
    python On_CIFAR100/IB_FL_local_unlearn_retrain.py





run_task "On_CIFAR10/IB_FL_local_unlearn_retrain_630_range1" \
    env CIFAR10_retrain_Unlearning_Class_Range=1 \
    python On_CIFAR10/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR10/IB_FL_local_unlearn_retrain_630_range2" \
    env CIFAR10_retrain_Unlearning_Class_Range=2 \
    python On_CIFAR10/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR10/IB_FL_local_unlearn_retrain_630_range3" \
    env CIFAR10_retrain_Unlearning_Class_Range=3 \
    python On_CIFAR10/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR10/IB_FL_local_unlearn_retrain_630_range4" \
    env CIFAR10_retrain_Unlearning_Class_Range=4 \
    python On_CIFAR10/IB_FL_local_unlearn_retrain.py




run_task "On_CIFAR100/IB_FL_local_unlearn_retrain_630_range1" \
    env CIFAR100_retrain_Unlearning_Class_Range=1 \
    python On_CIFAR100/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR100/IB_FL_local_unlearn_retrain_630_range3" \
    env CIFAR100_retrain_Unlearning_Class_Range=3 \
    python On_CIFAR100/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR100/IB_FL_local_unlearn_retrain_630_range7" \
    env CIFAR100_retrain_Unlearning_Class_Range=7 \
    python On_CIFAR100/IB_FL_local_unlearn_retrain.py

run_task "On_CIFAR100/IB_FL_local_unlearn_retrain_630_range9" \
    env CIFAR100_retrain_Unlearning_Class_Range=9 \
    python On_CIFAR100/IB_FL_local_unlearn_retrain.py


wait_for_all
echo "All finished."
