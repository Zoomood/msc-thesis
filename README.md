# Predicting temporal post-editing effort with MTQE

This repository contains the code used for my MSc thesis experiement on MTQE and PE effort.

## **Description**
This MSc study investigates whether sentence level reference free QE scores improve the prediction of relative temporal PE effort beyond a simple baseline across 6 target languages : Arabic, Dutch, Italian, Turkish, Ukranian and Vietnamese.

## **Table of Contents**


- [Research questions](#research)
- [Project structure](#Project)

## **Research questions**


Main RQ: To what extent does adding segment-level machine translation quality estimation scores change the prediction of relative temporal post-editing effort beyond a simple baseline across different target languages? 

Sub-questions:
- SQ1: How does out-of-sample prediction performance change when a segment-level QE score is added to the baseline model?
- SQ2: How do these changes in prediction performance vary across QE approaches and target languages?


## **Project structure**

- Data/
  - dataset_qe/
  - processed/
  - raw/
  
- Notebooks/
  - 1_explore_divemt_raw_file - data inspection notebook 
  - 2_qe_pretrained_cometkiwi - cometkiwi score generation notebook
  - 3_qe_llm - GEMBA-MQM scores generation using open weight and proprietary model notebook
  - 4_data_preprocessing - modelling dataset preparation notebook 
  - 5_model_training_and_evaluation - regression experiment and results evaluation notebook
- Results/
  - figures/
  - result files


## 


