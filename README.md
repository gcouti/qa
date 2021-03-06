# Question Answering Problem

Experiments with several Neural Networks to solve question answering QA problem.

## What is a Question Answering problem

Processing and reasoning information is the main characteristic observed in the behavior of intelligent species. However, it is a hard task to transpose this to computer logic, and it has been proven difficult for machine learning algorithms to learn from it. Regardless, some improvement has been made with deep learning models, which encourages us to continue exploring such problems, although a lot of work is required to achieve satisfactory results. This work attempts to explore a set of neural networks which showed good results reasoning problems. 

Creating a way for machines to reason and understand context is a challenge for Computer Science. The solution for this     problem has a crucial importance in order to build smart conversational pypagAI.agents and could be a large step to singularity. As much as we already have pypagAI.agents that are capable to dialog, they have not been sufficiently smart to pass the Turing Test.

## Get start

First of all, you must run setup.sh to create enviroment varibles

```
setup.sh
```

## Results

* **BaBI**

|Tasks  | Random | TFIDF | RF   | SVM | LSTM | RNN | BERT |
| ---   | :-:    | :-:   | :-:  | :-: | :-:  | :-: | :-:  |
|Task01 | 0.0    | 0.0   | 0.0  | 0.0 | 0.0  | 0.0 | 0.0  |


* **Datasets**

|Tasks   | BaBI | CBT | BT   | LAMBADA | SQuaD1.0 | 
| ---    | :-:  | :-: | :-:  | :-:     | :-:      |
| Human  | 0.0  | 0.0 | 0.0  | 0.0     | 0.0      |
| Random | 0.0  | 0.0 | 0.0  | 0.0     | 0.0      |
| TF-IDF | 0.0  | 0.0 | 0.0  | 0.0     | 0.0      |
| RNN    | 0.0  | 0.0 | 0.0  | 0.0     | 0.0      |
| ---    | :-:  | :-: | :-:  | :-:     | :-:      |
| RF     | 0.0  | 0.0 | 0.0  | 0.0     | 0.0      |
| SVM    | 0.0  | 0.0 | 0.0  | 0.0     | 0.0      |
| LSTM   | 0.0  | 0.0 | 0.0  | 0.0     | 0.0      |
| RN     | 0.0  | 0.0 | 0.0  | 0.0     | 0.0      |


