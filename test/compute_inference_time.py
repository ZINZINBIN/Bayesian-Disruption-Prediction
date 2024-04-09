import torch, time
import numpy as np
import argparse
from src.models.predictor import Predictor
from src.config import Config

config = Config()

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="Compute the inference time of the model")
    
    # random seed
    parser.add_argument("--random_seed", type = int, default = 42)

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--seq_len_efit", type = int, default = 100)
    parser.add_argument("--seq_len_ece", type = int, default = 1000)
    parser.add_argument("--seq_len_diag", type = int, default = 1000)
    parser.add_argument("--dist_warning", type = int, default = 400)
    parser.add_argument("--dist", type = int, default = 40)
    parser.add_argument("--dt", type = float, default = 0.001)
    parser.add_argument("--n_samples", type = int, default = 32)
    
    args = vars(parser.parse_args())

    return args

# torch device state
print("================= device setup =================")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":

    args = parsing()
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
    
    model = Predictor(config.header_config, config.classifier_config, device)
        
    print("\n==================== model summary ====================\n")
    model.to(device)
    model.summary()
    model.eval()
    
    t_measures = []    
    print("\n================ Computation time check ================\n")
    for n_iter in range(args['n_samples']):
        
        torch.cuda.empty_cache()
        torch.cuda.init()
        
        with torch.no_grad():
            sample_data = {
                "efit":torch.zeros((1, model.header_config['efit']['num_inputs'], model.header_config['efit']['seq_len'])),
                "ece":torch.zeros((1, model.header_config['ece']['num_inputs'], model.header_config['ece']['seq_len'])),
                "diag":torch.zeros((1, model.header_config['diag']['num_inputs'], model.header_config['diag']['seq_len'])),
            }
            
            t_start = time.time()
            sample_output = model(sample_data)
            t_end = time.time()
            dt = t_end - t_start
            t_measures.append(dt)
            
            sample_output.cpu()
        
        del sample_data
        del sample_output
        
    # statistical summary
    t_avg = np.mean(t_measures)
    t_std = np.std(t_measures)
    
    print("t_avg : {:.3f}, t_std : {:.3f}".format(t_avg, t_std))