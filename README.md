 # Bayesian deep learning for predicting disruption in KSTAR
<a href = "https://zinzinbin.notion.site/Research-archive-for-graduate-school-8187bf9b3ee5470d8a8dc05120655ceb" target = "_blank">[Research : Bayesian deep learning for predicting disruption in KSTAR]</a>

## Introduction
<div>
    This is github repository of advanced research for disruption prediction using bayesian deep learning. The previous research for disruption prediction is based on multimodal deep learning. The main issue of disruption prediciton is to prevent the occurrance of false alarms. The bayesian approach can handle this problem since the uncertainty now can be given, which means that the classifier can be trained by the generative approach, not by the discriminative appraoch. The disruption predictor, or the classifier is the generative classifier for our case. We can analyze the disruptive data in more detail using uncertainty and casual anaysis based on bayesian approach.
</div>
<div>
    <img src="/image/IMAGE_01.PNG"  width="640" height="320">
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

## The model performance of disruption prediction
<div>
    <img src="/image/IMAGE_03.PNG"  width="640" height="320">
</div>
<div>
    In this research, we predict the disrutions prior to the thermal quench, which can not be achieved from the previous research. This is mainly due to the fact that we utilized multi-diagnostic data and ECE data. We additionally used Dilated Convolution Network to handle multi-scale time series data effectively. This approach was inspired by Michael Churchil and <a href = "https://github.com/rmchurch/disruptcnn" target = "_blank">[github: disruptioncnn]</a>. Predicting disruption was successfully achieved with the prediction time as 40 ms from thermal quench.
</div>

## Uncertainty computation for various cases
<div>
    Below result shows the histogram of each class (disruption and non-disruption) for several cases (Missing alarm, False alarm, True positive). 
</div>
<div>
    <p float = 'left'>
        <img src="/image/IMAGE_04.PNG"  width="240" height="200">
        <img src="/image/IMAGE_05.PNG"  width="240" height="200">
        <img src="/image/IMAGE_06.PNG"  width="240" height="200">
    </p>
</div>

- Missing alarm: clear mis-prediction, but deviation for each class is somewhat valid.
- False alarm: mis-prediction, but deviation for each class and distance of each distribution are valid.
- True positive: small deviation + clear correct prediction