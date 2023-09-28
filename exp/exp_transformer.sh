
for seq_len in 10 20 30
do  
    for dist in 1 2 3 4 5
    do
        for mode in 'CQ' 'TQ'
        do
            for loss_type in 'Focal'
            do
                # Normal
                python3 train_transformer.py --gpu_num 1 --scaler 'Robust' --loss_type "$loss_type" --mode "$mode" --seq_len "$seq_len" --dist "$dist"
                
                # Resampling
                python3 train_transformer.py --gpu_num 1 --scaler 'Robust' --loss_type "$loss_type" --mode "$mode" --seq_len "$seq_len" --dist "$dist" --use_sampling 1
                
                # Reweighting
                python3 train_transformer.py --gpu_num 1 --scaler 'Robust' --loss_type "$loss_type" --mode "$mode" --seq_len "$seq_len" --dist "$dist" --use_weighting 1
                
                # Label smoothing
                python3 train_transformer.py --gpu_num 1 --scaler 'Robust' --loss_type "$loss_type" --mode "$mode" --seq_len "$seq_len" --dist "$dist" --use_label_smoothing 1
                
                # Resampling + Label smoothing
                python3 train_transformer.py --gpu_num 1 --scaler 'Robust' --loss_type "$loss_type" --mode "$mode" --seq_len "$seq_len" --dist "$dist" --use_sampling 1 --use_label_smoothing 1
                
                # Reweighting + Label smoothing
                python3 train_transformer.py --gpu_num 1 --scaler 'Robust' --loss_type "$loss_type" --mode "$mode" --seq_len "$seq_len" --dist "$dist" --use_weighting 1 --use_label_smoothing 1
            done
        done
    done
done