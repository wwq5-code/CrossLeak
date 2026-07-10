#!/bin/bash


echo "On_TinyImageNet/IB_FL_local_unlearn_retrain_702_range4"
export TinyImageNet_retrain_Unlearning_Class_Range=4
python On_TinyImageNet/IB_FL_local_unlearn_retrain.py > On_TinyImageNet/IB_FL_local_unlearn_retrain_702_range4 2>&1


echo "On_TinyImageNet/IB_FL_local_unlearn_retrain_702_range5"
export TinyImageNet_retrain_Unlearning_Class_Range=5
python On_TinyImageNet/IB_FL_local_unlearn_retrain.py > On_TinyImageNet/IB_FL_local_unlearn_retrain_702_range5 2>&1




echo "All finished."