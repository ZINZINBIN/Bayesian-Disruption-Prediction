
for seq_len in 10 20 30 40
do  
    for dist in 0 1 2
    do
        for mode in 'TQ' 'CQ'
        do
            for loss_type in 'CE' 'Focal'
            do
                # Normal
                python3 train_transformer.py --gpu_num 2 --scaler 'Robust' --loss_type "$loss_type" --mode "$mode" --seq_len "$seq_len" --dist "$dist"
                
                # Resampling
                python3 train_transformer.py --gpu_num 2 --scaler 'Robust' --loss_type "$loss_type" --mode "$mode" --seq_len "$seq_len" --dist "$dist" --use_sampling 1
                
                # Reweighting
                python3 train_transformer.py --gpu_num 2 --scaler 'Robust' --loss_type "$loss_type" --mode "$mode" --seq_len "$seq_len" --dist "$dist" --use_weighting 1
                
                # Label smoothing
                python3 train_transformer.py --gpu_num 2 --scaler 'Robust' --loss_type "$loss_type" --mode "$mode" --seq_len "$seq_len" --dist "$dist" --use_label_smoothing 1
                
                # Resampling + Label smoothing
                python3 train_transformer.py --gpu_num 2 --scaler 'Robust' --loss_type "$loss_type" --mode "$mode" --seq_len "$seq_len" --dist "$dist" --use_sampling 1 --use_label_smoothing 1
                
                # Reweighting + Label smoothing
                python3 train_transformer.py --gpu_num 2 --scaler 'Robust' --loss_type "$loss_type" --mode "$mode" --seq_len "$seq_len" --dist "$dist" --use_weighting 1 --use_label_smoothing 1
            done
        done
    done
done