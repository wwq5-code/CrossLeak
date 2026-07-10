#!/bin/bash




echo "On_TinyImageNet/IB_FL_local_unlearn_retrain_702_range2"
export TinyImageNet_retrain_Unlearning_Class_Range=2
python On_TinyImageNet/IB_FL_local_unlearn_retrain.py > On_TinyImageNet/IB_FL_local_unlearn_retrain_702_range2 2>&1

echo "On_TinyImageNet/IB_FL_local_unlearn_retrain_702_range3"
export TinyImageNet_retrain_Unlearning_Class_Range=3
python On_TinyImageNet/IB_FL_local_unlearn_retrain.py > On_TinyImageNet/IB_FL_local_unlearn_retrain_702_range3 2>&1


echo "All finished."