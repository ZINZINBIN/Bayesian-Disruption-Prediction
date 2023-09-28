import pandas as pd
import numpy as np
import warnings
from tqdm.auto import tqdm
from src.config import Config
from src.utils.compute import compute_GreenWald_density, compute_tau_e

warnings.filterwarnings(action = 'ignore')

def bound(x, value : float):
    return x if abs(x) < value else value * x / abs(x)

def positive(x):
    return x if x >0 else 0

if __name__ == "__main__":
    
    # configuration
    config = Config()
    
    # load data
    df_disrupt = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List_2022.csv', encoding = 'euc-kr')
    df_ece = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_ece.csv")
    df_ts = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_thomson.csv")
    df_efit = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_efit.csv")
    df_diag = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_diag.csv")
    
    # columns
    cols_ece = df_ece.columns[df_ece.notna().any()].drop(['Unnamed: 0','shot','time']).tolist()
    cols_ts = df_ts.columns[df_ts.notna().any()].drop(['Unnamed: 0','shot','time']).tolist()
    cols_efit = df_efit.columns[df_efit.notna().any()].drop(['Unnamed: 0','shot','time']).tolist()
    cols_diag = df_diag.columns[df_diag.notna().any()].drop(['Unnamed: 0','shot','time']).tolist()
    
    # shot selection: ignore experiments containing too much NaN values
    # EFIT data -> no need to select the shot (no NaN in this dataset)
    print('\n',"="*50)
    print("# process: shot selection")
    
    shot_ignore = []
    shot_list = df_ece.shot.unique()
    
    # 1st step: diagnostic data -> ignored shot selection
    feature_list = config.LM + config.DL + config.HCM + config.TCI + config.LV + config.RC + config.HA + config.BOL
    
    for col in feature_list:
        shot_ignore.extend(df_diag[df_diag[col].isna()].shot.unique())
        
    shot_ignore = np.unique(shot_ignore).tolist()
    
    # 2nd step: ece data -> ignored shot selection
    for col in config.ECE:
        shot_ignore.extend(df_ece[df_ece[col].isna()].shot.unique())

    shot_ignore = np.unique(shot_ignore).tolist()
    new_shot_list = [n for n in shot_list if n not in shot_ignore]
    
    # 3rd step: range contraction
    df_disrupt = df_disrupt[df_disrupt.shot.isin(new_shot_list)]
    df_efit = df_efit[df_efit.shot.isin(new_shot_list)]
    df_ece = df_ece[df_ece.shot.isin(new_shot_list)]
    df_diag = df_diag[df_diag.shot.isin(new_shot_list)]
    df_ts = df_ts[df_ts.shot.isin(new_shot_list)]
    
    print("# of shot : ", len(new_shot_list))
    
    # check NaN stil exist
    print("ECE data: ",sum(df_ece[config.ECE].isna().sum().values))
    print("Diagnostic data: ",sum(df_diag[feature_list].isna().sum().values))
    print("EFIT data: ", sum(df_efit[config.EFIT].isna().sum().values))
    
    # refinement
    print('\n',"="*50)
    print("# process: covert inf to nan")
    df_ece.replace([np.inf, -np.inf], np.nan)
    df_ts.replace([np.inf, -np.inf], np.nan)
    df_efit.replace([np.inf, -np.inf], np.nan)
    df_diag.replace([np.inf, -np.inf], np.nan)
    
    print("# preprocessing for specific features : Heating or other diagnostics")
    df_diag[config.ECH] = df_diag[config.ECH].fillna(0)
    df_diag[config.NBH] = df_diag[config.NBH].fillna(0)
    
    df_ts[config.TS_TE_CORE] = df_ts[config.TS_TE_CORE].fillna(0)
    df_ts[config.TS_TE_EDGE] = df_ts[config.TS_TE_EDGE].fillna(0)
    df_ts[config.TS_NE_CORE] = df_ts[config.TS_NE_CORE].fillna(0)
    df_ts[config.TS_NE_EDGE] = df_ts[config.TS_NE_EDGE].fillna(0)
    
    df_ece[config.ECE] = df_ece[config.ECE].fillna(0)
    
    # recheck nan
    print("ECE data: ",sum(df_ece[config.ECE].isna().sum().values))
    print("Diagnostic data: ",sum(df_diag[config.DIAG].isna().sum().values))
    print("EFIT data: ", sum(df_efit[config.EFIT].isna().sum().values))
    print("TS data: ", sum(df_ts[config.TS_TE + config.TS_NE].isna().sum().values))
    
    # re-scaling
    print('\n',"="*50)
    print("# process: re-scaling values")
    print("# EFIT data re-scaling...")
    df_efit['\\ipmhd'] = df_efit['\\ipmhd'].apply(lambda x : x / 1e6) # MA
    df_efit['\\ipmhd'] = df_efit['\\ipmhd'].apply(lambda x : abs(x))
    df_efit['\\bcentr'] = df_efit['\\bcentr'].apply(lambda x : abs(x))
    
    for col in config.EFIT:
        df_efit[col]= df_efit[col].apply(lambda x : bound(x,1e2))
    
    print('# ECE data re-scaling...')
    df_ece[config.ECE] = df_ece[config.ECE].apply(lambda x : x / 1e3) # keV
    for col in config.ECE:
        df_ece[col] = df_ece[col].apply(lambda x : bound(x,1e2))
        df_ece[col] = df_ece[col].apply(lambda x : positive(x))
    
    print("# Thomson data re-scaling...")
    for col in config.TS_TE_CORE:    
        df_ts[col] = df_ts[col].apply(lambda x : x / 1e3) # keV
        df_ts[col] = df_ts[col].apply(lambda x : bound(x,1e2))
        df_ts[col] = df_ts[col].apply(lambda x : positive(x))
    
    for col in config.TS_TE_EDGE:    
        df_ts[col] = df_ts[col].apply(lambda x : x / 1e3) 
        df_ts[col] = df_ts[col].apply(lambda x : bound(x,1e2))
        df_ts[col] = df_ts[col].apply(lambda x : positive(x))
    
    for col in config.TS_NE_CORE:    
        df_ts[col] = df_ts[col].apply(lambda x : x / 1e19)
        df_ts[col] = df_ts[col].apply(lambda x : bound(x,1e2))
        df_ts[col] = df_ts[col].apply(lambda x : positive(x))
    
    for col in config.TS_NE_EDGE:    
        df_ts[col] = df_ts[col].apply(lambda x : x / 1e19)
        df_ts[col] = df_ts[col].apply(lambda x : bound(x,1e2))
        df_ts[col] = df_ts[col].apply(lambda x : positive(x))
    
    print('# Diagnostic data re-scaling...')
    for col in config.DL:
        if col == '\\WTOT_DLM03':
            df_diag[col] = df_diag[col].apply(lambda x : x / 1e3) # kJ or keV
            df_diag[col] = df_diag[col].apply(lambda x : positive(x))
            
    df_diag[config.HCM] = df_diag[config.HCM].apply(lambda x : x / 1e3) # kA
    
    df_diag[config.RC] = df_diag[config.RC].apply(lambda x : x / 1e6) # MV / Mamp
            
    for col in config.TCI:
        df_diag[col] = df_diag[col].apply(lambda x : bound(x, 1e1))
        
    df_diag[config.HA] = df_diag[config.HA].apply(lambda x : x / 1e20)
    
    for col in config.HA:
        df_diag[col] = df_diag[col].apply(lambda x : bound(x,1e1))
    
    df_diag[config.ECH] = df_diag[config.ECH].apply(lambda x : x / 1e3) # kJ / kV / keV
    
    # valid shot selection
    print('\n',"="*50)
    print("# process: valid shot selection")
    
    # 1st: middle value => interpolation (just in case)
    df_efit = df_efit.interpolate(method = 'linear', limit_area = 'inside')
    df_diag = df_diag.interpolate(method = 'linear', limit_area = 'inside')
    df_ece = df_ece.interpolate(method = 'linear', limit_area = 'inside')
    df_ts = df_ts.interpolate(method = 'linear', limit_area = 'inside')

    # 2nd: shot selection again (short length -> flattop region)
    shot_ignore = []
    shot_list = df_ece.shot.unique().tolist()
    
    for shot in tqdm(shot_list, desc='remove short shot'):
        
        df_ece_shot = df_ece[df_ece.shot == shot]
        df_efit_shot = df_efit[df_efit.shot == shot]
        df_diag_shot = df_diag[df_diag.shot == shot]
        df_ts_shot = df_ts[df_ts.shot == shot]
        
        t_srt_list = [df_ece_shot.time.iloc[0], df_efit_shot.time.iloc[0], df_diag_shot.time.iloc[0], df_ts_shot.time.iloc[0]]
        t_end_list = [df_ece_shot.time.iloc[-1], df_efit_shot.time.iloc[-1], df_diag_shot.time.iloc[-1], df_ts_shot.time.iloc[-1]]
        n_point_list = [len(df_ece_shot.time), len(df_efit_shot.time), len(df_diag_shot.time), len(df_ts_shot.time)]
        name_list = ['ECE','EFIT','DIAG','TS']
        
        # 1st filter: flattop condition
        is_short = False
        for t_srt, t_end in zip(t_srt_list, t_end_list):
            if t_end - t_srt < 4.0:
                print("shot : {} - time length issue / too short".format(shot))
                is_short = True
                break
        
        if is_short:
            shot_ignore.append(shot)
            continue
        
        # 2nd filter: label error
        is_error = False
        
        # quench info
        tTQend = df_disrupt[df_disrupt.shot == shot].t_tmq.values[0]
        tftsrt = df_disrupt[df_disrupt.shot == shot].t_flattop_start.values[0]
        tipminf = df_disrupt[df_disrupt.shot == shot].t_ip_min_fault.values[0]
        
        for t_srt, t_end, n_point, name in zip(t_srt_list, t_end_list, n_point_list, name_list):
            
            if t_end < tftsrt:
                print("Invalid shot : {} - loss of data".format(shot))
                is_error = True
                break
            
            elif t_end < 4:
                print("Invalid shot : {} - operation time too short".format(shot))
                is_error = True
                break
            
            elif n_point < 20 and name == 'EFIT':
                print("Invalid shot : {} - few data points".format(shot))
                is_error = True
                break
            
            elif n_point < 100 and name != 'EFIT':
                print("Invalid shot : {} - few data points".format(shot))
                is_error = True
                break
            
            elif t_end < tipminf + 0.5:
                print("Invalid shot : {} - doesn't match with label".format(shot))
                is_error = True
                break
            
            elif t_srt > tftsrt - 0.5:
                print("Invalid shot : {} - some info before flattop region missed".format(shot))
                is_error = True
                break
            
        if is_error:
            shot_ignore.append(shot)
            continue
        
    new_shot_list = [n for n in shot_list if n not in shot_ignore]
    
    df_disrupt = df_disrupt[df_disrupt.shot.isin(new_shot_list)]
    df_efit = df_efit[df_efit.shot.isin(new_shot_list)]
    df_ece = df_ece[df_ece.shot.isin(new_shot_list)]
    df_diag = df_diag[df_diag.shot.isin(new_shot_list)]
    df_ts = df_ts[df_ts.shot.isin(new_shot_list)]
    
    print("# of valid shot : ", len(new_shot_list))
    
    # statistical preprocessing
    print('\n',"="*50)
    print("# process: statistical processing")
    
    # feature engineering
    compute_GreenWald_density(df_efit)
    
    # saving file
    print('\n',"="*50)
    print("# saving files...")
    
    df_efit = df_efit[['shot','time'] + config.EFIT + config.EFIT_FE]
    df_ece = df_ece[['shot','time'] + config.ECE]
    df_diag = df_diag[['shot','time'] + config.DIAG]
    df_ts = df_ts[['shot','time'] + config.TS_TE + config.TS_NE]
    
    df_disrupt.to_csv("./dataset/Bayesian_Disruption_Shot_List.csv", index = False)
    df_efit.to_csv("./dataset/Bayesian_Disruption_efit.csv", index = False)
    df_ece.to_csv("./dataset/Bayesian_Disruption_ece.csv", index = False)
    df_diag.to_csv("./dataset/Bayesian_Disruption_diag.csv", index = False)
    df_ts.to_csv("./dataset/Bayesian_Disruption_thomson.csv", index = False)
    print("# done")