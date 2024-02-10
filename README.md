# Network Anomaly Detection
Solution to *KDD CUP '99* using Pytorch, Skicit-learn and Pandas
## Data
*KDD CUP '99 Data set*
## Task Discription 
Build network intrusion detection system to detect anomalies and attacks in the
Network. There are two problems:
> 1. Binomial Classification: Activity is normal or attack
> 2. Multinomial classification: Activity is normal or DOS or PROBE or R2L or U2R
Please note that, currently the dependent variable (target variable) is not defined explicitly.
However, you can use attack variable to define the target variable as required.
## Solution
Use a deep neural net to train model to classify activity into 5 groups:
- 1. Normal
- 2. DOS
- 3. PROBE
- 4. R2L
- 5. U2R
