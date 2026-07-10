#!/bin/bash



echo "On_CIFAR10/IB_FL_local_unlearn_rfu_626_range1"

export CIFAR10_RFU_Unlearning_Class_Range=1
python On_CIFAR10/IB_FL_local_unlearn_rfu.py > On_CIFAR10/IB_FL_local_unlearn_rfu_626_range1 2>&1


echo "On_CIFAR10/IB_FL_local_unlearn_rfu_626_range2"

export CIFAR10_RFU_Unlearning_Class_Range=2
python On_CIFAR10/IB_FL_local_unlearn_rfu.py > On_CIFAR10/IB_FL_local_unlearn_rfu_626_range2 2>&1


echo "On_CIFAR10/IB_FL_local_unlearn_rfu_626_range3"

export CIFAR10_RFU_Unlearning_Class_Range=3
python On_CIFAR10/IB_FL_local_unlearn_rfu.py > On_CIFAR10/IB_FL_local_unlearn_rfu_626_range3 2>&1


echo "On_CIFAR10/IB_FL_local_unlearn_rfu_626_range4"

export CIFAR10_RFU_Unlearning_Class_Range=4
python On_CIFAR10/IB_FL_local_unlearn_rfu.py > On_CIFAR10/IB_FL_local_unlearn_rfu_626_range4 2>&1

echo "On_CIFAR10/IB_FL_local_unlearn_rfu_626_range5"

export CIFAR10_RFU_Unlearning_Class_Range=5
python On_CIFAR10/IB_FL_local_unlearn_rfu.py > On_CIFAR10/IB_FL_local_unlearn_rfu_626_range5 2>&1




echo "On_CIFAR10/IB_FL_local_unlearn_rfu_626_range5_ratio_02"
export CIFAR10_RFU_Unlearning_Class_Range=5
export CIFAR10_RFU_Unlearning_Ratio=0.02
python On_CIFAR10/IB_FL_local_unlearn_rfu.py > On_CIFAR10/IB_FL_local_unlearn_rfu_626_range5_ratio_02 2>&1


echo "On_CIFAR10/IB_FL_local_unlearn_rfu_626_range5_ratio_03"
export CIFAR10_RFU_Unlearning_Class_Range=5
export CIFAR10_RFU_Unlearning_Ratio=0.03
python On_CIFAR10/IB_FL_local_unlearn_rfu.py > On_CIFAR10/IB_FL_local_unlearn_rfu_626_range5_ratio_03 2>&1


echo "On_CIFAR10/IB_FL_local_unlearn_rfu_626_range5_ratio_04"
export CIFAR10_RFU_Unlearning_Class_Range=5
export CIFAR10_RFU_Unlearning_Ratio=0.04
python On_CIFAR10/IB_FL_local_unlearn_rfu.py > On_CIFAR10/IB_FL_local_unlearn_rfu_626_range5_ratio_04 2>&1


echo "On_CIFAR10/IB_FL_local_unlearn_rfu_626_range5_ratio_05"
export CIFAR10_RFU_Unlearning_Class_Range=5
export CIFAR10_RFU_Unlearning_Ratio=0.05
python On_CIFAR10/IB_FL_local_unlearn_rfu.py > On_CIFAR10/IB_FL_local_unlearn_rfu_626_range5_ratio_05 2>&1







echo "On_CIFAR100/IB_FL_local_unlearn_rfu_626_range1"

export CIFAR100_RFU_Unlearning_Class_Range=1
python On_CIFAR100/IB_FL_local_unlearn_rfu.py > On_CIFAR100/IB_FL_local_unlearn_rfu_626_range1 2>&1



echo "On_CIFAR100/IB_FL_local_unlearn_rfu_626_range3"

export CIFAR100_RFU_Unlearning_Class_Range=3
python On_CIFAR100/IB_FL_local_unlearn_rfu.py > On_CIFAR100/IB_FL_local_unlearn_rfu_626_range3 2>&1


echo "On_CIFAR100/IB_FL_local_unlearn_rfu_626_range5"

export CIFAR100_RFU_Unlearning_Class_Range=5
python On_CIFAR100/IB_FL_local_unlearn_rfu.py > On_CIFAR100/IB_FL_local_unlearn_rfu_626_range5 2>&1



echo "On_CIFAR100/IB_FL_local_unlearn_rfu_626_range7"

export CIFAR100_RFU_Unlearning_Class_Range=7
python On_CIFAR100/IB_FL_local_unlearn_rfu.py > On_CIFAR100/IB_FL_local_unlearn_rfu_626_range7 2>&1


echo "On_CIFAR100/IB_FL_local_unlearn_rfu_626_range9"

export CIFAR100_RFU_Unlearning_Class_Range=9
python On_CIFAR100/IB_FL_local_unlearn_rfu.py > On_CIFAR100/IB_FL_local_unlearn_rfu_626_range9 2>&1



echo "On_CIFAR100/IB_FL_local_unlearn_rfu_626_range5_ratio_02"
export CIFAR100_RFU_Unlearning_Class_Range=5
export CIFAR100_RFU_Unlearning_Ratio=0.02
python On_CIFAR100/IB_FL_local_unlearn_rfu.py > On_CIFAR100/IB_FL_local_unlearn_rfu_626_range5_ratio_02 2>&1


echo "On_CIFAR100/IB_FL_local_unlearn_rfu_626_range5_ratio_03"
export CIFAR100_RFU_Unlearning_Class_Range=5
export CIFAR100_RFU_Unlearning_Ratio=0.03
python On_CIFAR100/IB_FL_local_unlearn_rfu.py > On_CIFAR100/IB_FL_local_unlearn_rfu_626_range5_ratio_03 2>&1


echo "On_CIFAR100/IB_FL_local_unlearn_rfu_626_range5_ratio_04"
export CIFAR100_RFU_Unlearning_Class_Range=5
export CIFAR100_RFU_Unlearning_Ratio=0.04
python On_CIFAR100/IB_FL_local_unlearn_rfu.py > On_CIFAR100/IB_FL_local_unlearn_rfu_626_range5_ratio_04 2>&1


echo "On_CIFAR100/IB_FL_local_unlearn_rfu_626_range5_ratio_05"
export CIFAR100_RFU_Unlearning_Class_Range=5
export CIFAR100_RFU_Unlearning_Ratio=0.05
python On_CIFAR100/IB_FL_local_unlearn_rfu.py > On_CIFAR100/IB_FL_local_unlearn_rfu_626_range5_ratio_05 2>&1




echo "On_TinyImageNet/IB_FL_local_unlearn_rfu_626_range1"
export TinyImageNet_RFU_Unlearning_Class_Range=1
python On_TinyImageNet/IB_FL_local_unlearn_rfu.py > On_TinyImageNet/IB_FL_local_unlearn_rfu_626_range1 2>&1



echo "On_TinyImageNet/IB_FL_local_unlearn_rfu_626_range3"
export TinyImageNet_RFU_Unlearning_Class_Range=3
python On_TinyImageNet/IB_FL_local_unlearn_rfu.py > On_TinyImageNet/IB_FL_local_unlearn_rfu_626_range3 2>&1


echo "On_TinyImageNet/IB_FL_local_unlearn_rfu_626_range5"
export TinyImageNet_RFU_Unlearning_Class_Range=5
python On_TinyImageNet/IB_FL_local_unlearn_rfu.py > On_TinyImageNet/IB_FL_local_unlearn_rfu_626_range5 2>&1

echo "On_TinyImageNet/IB_FL_local_unlearn_rfu_626_range7"
export TinyImageNet_RFU_Unlearning_Class_Range=7
python On_TinyImageNet/IB_FL_local_unlearn_rfu.py > On_TinyImageNet/IB_FL_local_unlearn_rfu_626_range7 2>&1


echo "On_TinyImageNet/IB_FL_local_unlearn_rfu_626_range9"
export TinyImageNet_RFU_Unlearning_Class_Range=9
python On_TinyImageNet/IB_FL_local_unlearn_rfu.py > On_TinyImageNet/IB_FL_local_unlearn_rfu_626_range9 2>&1



echo "All finished."