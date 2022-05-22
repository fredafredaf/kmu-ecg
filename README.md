# kmu-ecg

This is the project to use CNN to classify ECGs and heart sound signals collected at KMU

# Data
Data from two groups of paitents are included:

- normal patients
- patients with acute coronary syndrome (ACS)

# Use in Google Colab

Connect your notebook to a GPU runtime by doing Runtime > Change Runtime type > GPU.

Paste the following to the cell and run:

```
!git clone https://github.com/fredafredaf/kmu-ecg.git
%cd kmu-ecg
!pip3 install -r requirements.txt
%env PYTHONPATH=.:$PYTHONPATH
```
Then, you can train the model by using:

`!python3 training/run_experiment.py --im_type=ecg --epochs=30 `

  
 


