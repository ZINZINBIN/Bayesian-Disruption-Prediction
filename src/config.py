import numpy as np
from src.models.tcn import calc_seq_length, calc_dilation

class Config():
      
      STATE_FIXED = 42
      
      # filepath list
      filepath = {
            "efit":"./dataset/Bayesian_Disruption_efit.csv",
            "ece":"./dataset/Bayesian_Disruption_ece.csv",
            "diag":"./dataset/Bayesian_Disruption_diag.csv",
            "disrupt":'./dataset/Bayesian_Disruption_Shot_List.csv'
      }
      
      # EFIT
      # EFIT = ['\\q0', '\\q95', '\\ipmhd', '\\kappa', '\\tritop','\\tribot', '\\betap', '\\betan', '\\li', '\\bcentr', '\\rsurf', '\\aminor',]
      EFIT = ['\\q0', '\\q95', '\\qmin', '\\ipmhd', '\\kappa', '\\tritop','\\tribot', '\\betap', '\\betan', '\\li', '\\bcentr', '\\rsurf', '\\aminor','\\drsep', '\\rxpt1', '\\zxpt1', '\\rxpt2', '\\zxpt2']
      
      # Feature engineering
      # EFIT_FE = ['\\nG'] # ['\\nG', '\\nG_tci01', '\\nG_tci02', '\\nG_tci03', '\\nG_tci04', '\\nG_tci05']
      EFIT_FE = ['\\nG', '\\troyon']
      
      # ECE
      ECE = ['\\ECE08', '\\ECE13', '\\ECE18', '\\ECE24', '\\ECE26','\\ECE32', '\\ECE37', '\\ECE42', '\\ECE54', '\\ECE63', '\\ECE67', '\\ECE73']
      
      # Dianostics
      LM = ['\\LM01','\\LM02','\\LM03','\\LM04']                                                                  # lock mode
      DL = ['\\BETAP_DLM03','\\DMF_DLM03','\\DLM01','\\DLM03','\\WTOT_DLM03']                                     # Diamagnetic loop
      HCM = ['\\HCMIL01','\\HCMIL02','\\HCMIL03','\\HCMIL04','\\HCMIL05','\\HCMIL06','\\HCMIL07','\\HCMIL08','\\HCMIL09','\\HCMIL10',
            '\\HCMIL11','\\HCMIL12','\\HCMIL13','\\HCMIL14','\\HCMIL15','\\HCMIL16','\\HCMID01','\\HCMID02','\\HCMID03','\\HCMID04',
            '\\HCMID05','\\HCMID06','\\HCMID07','\\HCMID08','\\HCMCD01','\\HCMCD02','\\HCMCD03','\\HCMCD04','\\HCMCD05','\\HCMCD06',
            '\\HCMCD07','\\HCMCD08','\\HCMCD09','\\HCMCD10','\\HCMCD11','\\HCMCD12','\\HCMCD13','\\HCMCD14','\\HCMCD15','\\HCMCD16',
            '\\HCMOD01','\\HCMOD02','\\HCMOD03','\\HCMOD04','\\HCMOD05','\\HCMOD06','\\HCMOD07','\\HCMOD08']      # Halo current monitoring
      TCI = ['\\ne_tci01','\\ne_tci02','\\ne_tci03','\\ne_tci04','\\ne_tci05']                                    # TCI 
      LV = ['\\LV01','\\LV12','\\LV23','\\LV34','\\LV45']                                                         # Flux loop & Loop voltage
      RC = ['\\RC01','\\RC02','\\RC03','\\VCM01','\\VCM02','\\VCM03','\\RCPPU1', '\\RCPPL1']                      # Rogowski coil
      HA = ['\\TOR_HA01','\\TOR_HA02','\\TOR_HA03','\\TOR_HA04','\\TOR_HA05','\\TOR_HA06','\\TOR_HA07','\\TOR_HA08',
            '\\TOR_HA09','\\TOR_HA10','\\TOR_HA11','\\TOR_HA12','\\TOR_HA13','\\TOR_HA14','\\TOR_HA15','\\TOR_HA16','\\TOR_HA17',
            '\\TOR_HA18','\\TOR_HA19','\\TOR_HA20','\\POL_HA01','\\POL_HA02','\\POL_HA03','\\POL_HA04','\\POL_HA05',
            '\\POL_HA06','\\POL_HA07','\\POL_HA08','\\POL_HA09','\\POL_HA10']                                     # H-alpha signal
      ECH = ['\\EC2_PWR','\\EC3_PWR', '\\EC4_PWR']                                                                # EC heating 
      NBH = ['\\nb11_pnb','\\nb12_pnb','\\nb13_pnb']                                                              # NB heating
      BOL = ['\\ax3_bolo02:FOO']
      
      DIAG = LM + DL + HCM + TCI + LV + RC + HA + ECH + NBH + BOL
      
      COLUMN_NAME_SCALER = {
            "efit":EFIT + EFIT_FE,
            "ece":ECE,
            "diag":DIAG
      }
      
      # default
      header_config = {
            "efit":{
                  "num_inputs":len(EFIT + EFIT_FE),
                  "hidden_dim":64,
                  "num_channels":[64] * 4,
                  "kernel_size":2,
                  "dropout":0.1,
                  "dilation_size":calc_dilation(2, 2, 4, 1024),
                  "seq_len":10
            },
            "ece":{
                  "num_inputs":len(ECE),
                  "hidden_dim":64,
                  "num_channels":[64] * 4,
                  "kernel_size":2,
                  "dropout":0.1,
                  "dilation_size":calc_dilation(2, 2, 4, 1024),
                  "seq_len":50
            },
            "diag":{
                  "num_inputs":len(DIAG),
                  "hidden_dim":64,
                  "num_channels":[64]*4,
                  "kernel_size":2,
                  "dropout":0.1,
                  "dilation_size":calc_dilation(2, 2, 4, 1024),
                  "seq_len":50
            }
      }
      
      classifier_config = {
            "cls_dim" : 128,
            "n_classes":2,
            "prior_pi":0.5,
            "prior_sigma1":1.0,
            "prior_sigma2":0.0025,
      }
      
      # hyperparameter optimized 
      optim_header_config = {
            "efit":{
                  "num_inputs":len(EFIT + EFIT_FE),
                  "hidden_dim":64,
                  "num_channels":[64] * 4,
                  "kernel_size":2,
                  "dropout":0.1,
                  "dilation_size":calc_dilation(2, 3, 4, 1024 * 8),
                  "seq_len":10
            },
            "ece":{
                  "num_inputs":len(ECE),
                  "hidden_dim":64,
                  "num_channels":[128] * 4,
                  "kernel_size":2,
                  "dropout":0.1,
                  "dilation_size":calc_dilation(2, 3, 4, 1024 * 8),
                  "seq_len":50
            },
            "diag":{
                  "num_inputs":len(DIAG),
                  "hidden_dim":64,
                  "num_channels":[128]*4,
                  "kernel_size":2,
                  "dropout":0.1,
                  "dilation_size":calc_dilation(2, 3, 4, 1024 * 8),
                  "seq_len":50
            }
      }
      
      optim_classifier_config = {
            "cls_dim" : 256,
            "n_classes":2,
            "prior_pi":0.5,
            "prior_sigma1":1.0,
            "prior_sigma2":0.0025,
      }
      
      # Thomson scattering
      TS_TE_CORE = ['\\TS_CORE1:CORE1_TE', '\\TS_CORE2:CORE2_TE', '\\TS_CORE3:CORE3_TE', '\\TS_CORE4:CORE4_TE', '\\TS_CORE5:CORE5_TE', '\\TS_CORE6:CORE6_TE', '\\TS_CORE7:CORE7_TE', '\\TS_CORE8:CORE8_TE', '\\TS_CORE9:CORE9_TE', '\\TS_CORE10:CORE10_TE', '\\TS_CORE11:CORE11_TE', '\\TS_CORE12:CORE12_TE', '\\TS_CORE13:CORE13_TE', '\\TS_CORE14:CORE14_TE']
      TS_TE_EDGE = ['\\TS_EDGE1:EDGE1_TE', '\\TS_EDGE2:EDGE2_TE', '\\TS_EDGE3:EDGE3_TE', '\\TS_EDGE4:EDGE4_TE', '\\TS_EDGE5:EDGE5_TE', '\\TS_EDGE6:EDGE6_TE', '\\TS_EDGE7:EDGE7_TE', '\\TS_EDGE8:EDGE8_TE', '\\TS_EDGE9:EDGE9_TE', '\\TS_EDGE10:EDGE10_TE', '\\TS_EDGE11:EDGE11_TE', '\\TS_EDGE12:EDGE12_TE', '\\TS_EDGE13:EDGE13_TE', '\\TS_EDGE14:EDGE14_TE']
      TS_NE_CORE = ['\\TS_CORE1:CORE1_NE', '\\TS_CORE2:CORE2_NE', '\\TS_CORE3:CORE3_NE', '\\TS_CORE4:CORE4_NE', '\\TS_CORE5:CORE5_NE', '\\TS_CORE6:CORE6_NE', '\\TS_CORE7:CORE7_NE', '\\TS_CORE8:CORE8_NE', '\\TS_CORE9:CORE9_NE', '\\TS_CORE10:CORE10_NE', '\\TS_CORE11:CORE11_NE', '\\TS_CORE12:CORE12_NE', '\\TS_CORE13:CORE13_NE', '\\TS_CORE14:CORE14_NE']
      TS_NE_EDGE = ['\\TS_EDGE1:EDGE1_NE', '\\TS_EDGE2:EDGE2_NE', '\\TS_EDGE3:EDGE3_NE', '\\TS_EDGE4:EDGE4_NE', '\\TS_EDGE5:EDGE5_NE', '\\TS_EDGE6:EDGE6_NE', '\\TS_EDGE7:EDGE7_NE', '\\TS_EDGE8:EDGE8_NE', '\\TS_EDGE9:EDGE9_NE', '\\TS_EDGE10:EDGE10_NE', '\\TS_EDGE11:EDGE11_NE', '\\TS_EDGE12:EDGE12_NE', '\\TS_EDGE13:EDGE13_NE', '\\TS_EDGE14:EDGE14_NE']

      TS_TE = TS_TE_CORE + TS_TE_EDGE
      TS_NE = TS_NE_CORE + TS_NE_EDGE

      TS_CORE_RADIUS = [1.797, 1.818, 1.841, 1.862, 1.884, 1.908, 1.931, 1.954, 1.979, 2.004, 2.03, 2.056, 2.082, 2.108]
      TS_EDGE_RADIUS = [2.108, 2.120, 2.133, 2.146, 2.153, 2.171, 2.183, 2.190, 2.197, 2.203, 2.209, 2.216, 2.229, 2.243]
      TS_RADIUS = TS_CORE_RADIUS + TS_EDGE_RADIUS[1:]
      
      TS_AVG_COLS = ['\\TS_NE_CORE_AVG', '\\TS_NE_EDGE_AVG', '\\TS_TE_CORE_AVG','\\TS_TE_EDGE_AVG']

      # exceptional columns
      EXCEPT_COLS = ['\\TOR_HA00','\\POL_HA00','\\HCMIL09','\\HCMIL10','\\HCMIL11', '\\HCMIL02', '\\HCMIL04', '\\HCMIL05',
                     '\\RCPPU2:FOO', '\\RCPPU2B:FOO','\\RCPPL2B:FOO', '\\DLM02', '\\TS_CORE13:CORE13_TE', '\\TS_CORE14:CORE14_TE',
                     '\\TS_EDGE13:EDGE13_TE', '\\TS_EDGE14:EDGE14_TE', '\\TS_CORE13:CORE13_NE', '\\TS_CORE14:CORE14_NE', '\\TS_EDGE13:EDGE13_NE', 
                     '\\TS_EDGE14:EDGE14_NE', '\\q0', '\\ne_tci01','\\ne_tci02','\\ne_tci03','\\ne_tci04','\\ne_tci05','\\bcentr']

      # permutation feature importance
      feature_map = {
            '\\q95' : 'q95', 
            '\\ipmhd':'Ip', 
            '\\kappa':'kappa', 
            '\\tritop': 'tri-top', 
            '\\tribot': 'tri-bot',
            '\\betap': 'betap',
            '\\betan': 'betan',
            '\\li': 'li',  
            '\\WTOT_DLM03':'W-tot',
            '\\ne_inter01' : 'Ne-avg', 
            '\\TS_NE_CORE_AVG' : 'Ne-core', 
            '\\TS_TE_CORE_AVG': 'Te-core',
            '\\TS_NE_EDGE_AVG' : 'Ne-edge', 
            '\\TS_TE_EDGE_AVG': 'Te-edge',
            '\\nG' : 'N-Greenwald',
            '\\ne_nG_ratio' : 'NG ratio',
            '\\DLM03': 'DLM03',
            '\\RC03' : 'RC03',
            '\\VCM03' : 'VCM03',
            '\\rsurf': 'Rc', 
            '\\aminor':'a'
      }

      # select input features
      input_features = [
            '\\q95', '\\ipmhd', '\\kappa', '\\tritop', '\\tribot',
            '\\betap','\\li', '\\WTOT_DLM03', '\\ne_inter01', '\\ne_nG_ratio',
            '\\TS_NE_CORE_AVG', '\\TS_TE_CORE_AVG', '\\TS_TE_EDGE_AVG', '\\TS_NE_EDGE_AVG',
            '\\DLM03', '\\RC03','\\VCM03','\\rsurf', '\\aminor'
      ]