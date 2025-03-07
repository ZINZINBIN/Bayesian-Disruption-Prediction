 # Bayesian deep learning for predicting disruption in KSTAR
<a href = "https://arxiv.org/abs/2312.12979" target = "_blank">[Paper: Enhancing Disruption Prediction through Bayesian Neural Network in KSTAR]</a>

## Introduction
<div>
    This is github repository of advanced research for disruption prediction using bayesian deep learning. The previous research for disruption prediction is based on multimodal deep learning. The main issue of disruption prediciton is to prevent the occurrance of false alarms. The bayesian approach can handle this problem since the uncertainty now can be given, which means that the classifier can be trained by the generative approach, not by the discriminative appraoch. The disruption predictor, or the classifier is the generative classifier for our case. We can analyze the disruptive data in more detail using uncertainty and casual anaysis based on bayesian approach.
</div>
<div>
    <img src="/image/IMAGE_01.PNG"  width="540" height="320">
</div>
<div>
    In essence, neural networks following a general approach lack the capability to gauge the uncertainty of their predictions accurately. This gives rise to several pertinent concerns, including overconfidence, which manifests as excessively assured predictions even when neural networks offer a confidence interval, and overfitting, characterized by limited generalization resulting from an inclination to closely adhere to the training data rather than reflecting the broader underlying distribution. These issues can pose significant challenges when applied to tasks involving predictive disruption analysis. To cover these issues, we apply bayesian neural network as an alternative way to predict disruption as well. Bayesian neural network is one of the stochastic neural networks trained by variational inference. In other word, this approach can cover computation of the uncertainties, scalability, and the robustness in case of small dataset. 
</div>
<div>
    <img src="/image/IMAGE_02.PNG"  width="640" height="320">
</div>
<div>
    Now, computation of the uncertainties including aleatoric and epistemic uncertainty can be computed since it was already covered in the paper (Kumar Shridhar et al. 2019). Based on this approach, we utilize these uncertainties to enhance the precision of disruption prediction models. 
</div>

## How to run
<div>
    There are mainly three different types of codes in this repository: (1) train (2) test (3) optimize. The codes starting with 'train' are used for training the disruption predictors without other post-hoc methods. Those starting with 'test' are used for evaluating the model performance including the quantitive metrics and qualitive evaluation such as t-SNE and continuous disruption prediction. Lastly, 'optimize' codes are used for tunning the hyperparameters of models or post-hoc parameters. The details of setting the arguments while excecuting codes are as below.
</div>

### Train disruption predictors
- training a non-bayesian disruption predictor
    ```
    python3 train_model.py
    ```

- training a Bayesian disruption predictor
    ```
    python3 train_bayes_model.py
    ```

### Evaluate disruption predictors
- Evaluating a non-bayesian disruption predictor: qualitive metric(F1,Pre,Rec), t-SNE visualization, continuous disruption prediction
    ```
    python3 test_model.py
    ```

- Evaluating a Bayesian disruption predictor: qualitive metric(F1,Pre,Rec), t-SNE visualization, continuous disruption prediction
    ```
    python3 test_bayes_model.py
    ```

- Evaluating uncertainty: visualized probaility distribution, tables of test prediction and uncertainty
    ```
    python3 test_uncertainty.py
    ```

- Evaluating feature importance: visualized feature importance during disruptive phase , tables of test prediction and feature importance
    ```
    python3 test_feature_importance.py
    ```

- Evaluating disruption predictions for test shots: visualized disruption predictions for test shots
    ```
    python3 test_disruption_prediction.py
    ```

### Optimize hyper-parameters for enhancement
- Optimizing the hyperparameters of model configuration
    ```
    python3 optiminze_hyperparameter.py
    ```

- Optimizing temperature scaling for calibration
    ```
    python3 optimize_calibration.py
    ```

## Overall model performance of our proposed model
<div>
    <p float = 'left'>
        <img src="/image/IMAGE_03.PNG"  width="320" height="200">
        <img src="/image/IMAGE_04.PNG"  width="320" height="200">
    </p>
</div>
<div>
    The overall model performance of predicting disruptions is as shown in above figure. This evaluation was conducted by test disruptive shots from 2019 to 2022 in KSTAR campaign. As a result, over 80% of disruptive shots in KSTAR were predicted before 65 ms from thermal quench, while about half ot disruptive shots were also predicted before 300 ms. 
</div>

## Simulation result of disruption prediction with KSTAR experiment
<div>
    <img src="/image/IMAGE_05.PNG"  width="720" height="320">
</div>
<div>
    <img src="/image/IMAGE_06.PNG"  width="720" height="320">
</div>
<div>
    We proceeded with continuous disruption prediction with KSTAR shot 30312 with uncertainty estimation. The result showed successful disruption prediction before 680 ms from thermal quench in shot 30312 with EFIT and diagnostics. Aleatoric and epistemic uncertainty estimated over time have shown drastic increase near the transition of disruptive phase, indicating that recognizing the transition between normal and disruptive state has high aleatoric and epistemic uncertainty. However, the uncertainties decrease again while approaching to thermal quench, meaning that the distinct data distribution of disruptive phase are detectable. Below figure depicts the detailed information of shot 30312. Drastic decrease in both ECE and TCI at 10.88 (s) from below figure indicates the disruptive phase (thermal quench) while observing the fluctuation in diagnostics and plasma current.  
</div>
<div>
    <img src="/image/IMAGE_07.PNG"  width="720" height="320">
</div>

## Uncertainty computation for various cases
<div>
    Below result shows the histogram of each class (disruption and non-disruption) for several cases (True alarm, Missing alarm, and False alarm). In this figure, true alarm case has low deviation of output probability while the others have high deviation of their output, meaning that the uncertainty estimated from data and the model can provide the notion for identifying wrong alarms. 
</div>
<div>
    <p float = 'left'>
        <img src="/image/IMAGE_08.PNG"  width="240" height="200">
        <img src="/image/IMAGE_09.PNG"  width="240" height="200">
        <img src="/image/IMAGE_10.PNG"  width="240" height="200">
    </p>
</div>

- True alarm: Successful case for predicting disruption before 40ms from TQ with low deviation
- Missing alarm: Failure of predicting disruptions before 40ms from TQ with high deviation
- False alarm: Ealry alarm or Mis-classification of non-disruptive data with high deviation

<div>
    <p float = 'left'>
        <img src="/image/IMAGE_11.PNG"  width="380" height="200">
        <img src="/image/IMAGE_12.PNG"  width="380" height="200">
    </p>
</div>
<div>
    Through the histogram of estimated uncertaintes for true alarm, missing alarm, and false alarm cases, it is shown that false alarm cases and missing alarm cases have higher aleatoric and epistemic uncertainties compared with true alarm cases generally. Therefore, it is plausible to enhance the model's precision based on multipe threshold tuning method for the output probability and the uncertaintes. 
</div>

<div>
    <img src="/image/IMAGE_13.PNG"  width="720" height="320">
</div>
<div>
    <img src="/image/IMAGE_14.PNG"  width="720" height="320">
</div>
<div>
    From the optimized thresholds associated with the model's output and the uncertaintes, we can verify the effectiveness of threshold tuning for enhancing the model's precision in terms of continuous disruption prediction. Above figure shows that the time points where the false alarms occur are all handled in KSTAR shot 28158.
</div>

## Feature importance analysis
<div>
    <p float = 'left'>
        <img src="/image/IMAGE_15.PNG"  width="360" height="300">
        <img src="/image/IMAGE_16.PNG"  width="360" height="300">
    </p>
</div>
<div>
    Finally, we applied Integrated Gradients algorithm to compute the feature importance during the inference process, thereby enabling the cause estimation for predicting disruptions. Above figures show the feature importance for an example of true alarm case and temporal feature importance during the thermal quench process. Through the input feature importance, it may be possible not only to predict the disruption but also to estimate the causes of disruptions, which will provide additional information of mitigating disruptions by estimating the causes. 
</div>

## 📖 Citation
If you use this repository in your research, please cite the following:

### 📜 Research Article
[Enhancing disruption prediction through Bayesian neural network in KSTAR](https://doi.org/10.48550/arXiv.2409.08231)  
Jinsu Kim et al 2024 Plasma Phys. Control. Fusion 66 075001

### 📌 Code Repository
Jinsu Kim (2024). **Bayesian-Disruption-Prediction**. GitHub.  
[https://github.com/ZINZINBIN/Bayesian-Disruption-Prediction](https://github.com/ZINZINBIN/Bayesian-Disruption-Prediction)

#### 📚 BibTeX:
```bibtex
@software{Kim_Bayesian_Deep_Learning_2024,
author = {Kim, Jinsu},
doi = {https://doi.org/10.1088/1361-6587/ad48b7},
license = {MIT},
month = may,
title = {{Bayesian Deep Learning based Disruption Prediction Model}},
url = {https://github.com/ZINZINBIN/Bayesian-Disruption-Prediction},
version = {1.0.0},
year = {2024}
}
```