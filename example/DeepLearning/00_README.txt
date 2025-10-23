# Example run for reproduction of RPTK Deep LEarning results from RPTK paper 

# Predict
python3 run_DeepLearning.py --csv /path/to/Predict.csv  --output_dir /path/to/Predict --project_name RPTK_paper_final --max_epoch 200  --run_name Predict --data_aug True --batch_norm True --batch_size 15 --folds 5 --num_classes 2 --n_cpu 4 --cropping True

# LIDC
python3 run_DeepLearning.py --csv/path/to//LIDC.csv  --output_dir /path/to/LIDC --project_name RPTK_paper_final --max_epoch 200  --run_name LIDC --data_aug True --batch_norm True --batch_size 15 --folds 5 --num_classes 2 --n_cpu 4 --cropping True

# Melanoma
python3 run_DeepLearning.py --csv /path/to/Melanoma.csv  --output_dir /path/to/Melanoma --project_name RPTK_paper_final --max_epoch 200  --run_name Melanoma --data_aug True --batch_norm True --batch_size 15 --folds 5 --num_classes 2 --n_cpu 4 --cropping True

# GIST
python3 run_DeepLearning.py --csv /path/to/GIST.csv  --output_dir /path/to/GIST --project_name RPTK_paper_final --max_epoch 200  --run_name GIST --data_aug True --batch_norm True --batch_size 15 --folds 5 --num_classes 2 --n_cpu 4 --cropping True

# CRLM
python3 run_DeepLearning.py --csv /path/to/CRLM.csv  --output_dir /path/to/CRLM --project_name RPTK_paper_final --max_epoch 200  --run_name CRLM --data_aug True --batch_norm True --batch_size 15 --folds 5--num_classes 2 --n_cpu 4 --cropping True

# Lipo 
python3 run_DeepLearning.py --csv  /path/to/Lipo.csv  --output_dir  /path/to/Lipo --project_name RPTK_paper_final --max_epoch 200  --run_name Lipo --data_aug True --batch_norm True --batch_size 15 --folds 5 --num_classes 2 --n_cpu 4 --cropping True

# Liver
python3 run_DeepLearning.py --csv  /path/to/Liver.csv  --output_dir  /path/to/Liver --project_name RPTK_paper_final --max_epoch 200  --run_name Liver --data_aug True --batch_norm True --batch_size 15 --folds 5 --num_classes 2 --n_cpu 4 --cropping True
 
 # Desmoid
python3 run_DeepLearning.py --csv  /path/to/Desmoid.csv  --output_dir  /path/to/Desmoid --project_name RPTK_paper_final --max_epoch 200  --run_name Desmoid --data_aug True --batch_norm True --batch_size 15 --folds 5 --num_classes 2 --n_cpu 4 --cropping True
