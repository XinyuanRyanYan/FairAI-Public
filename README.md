# Fairness in AI Education Module 

The Fairness in AI E-book is a web-based system that combines text, images, and interactive visualizations to educate non-technical users about fairness in AI, including decision-making by Machine Learning algorithms, quantifying bias in model outcomes, and applying algorithms to mitigate bias. Please refer to our paper [Exploring Visualization for Fairness in AI Education](https://www.sci.utah.edu/~beiwang/publications/Fairness_BeiWang_2024.pdf) for details.


## Dependencies

This software requires Python3, R, Flask and aif360 to run. 
If you do not have these packages installed, please use the following command to intall them.

	pip install 'aif360[all]'
	pip install flask
	pip install flask_session
	pip install flask_assets

### Import Dataset
To use the 4th chapter *Final Project* of this tool, you will need to place the file  `'bank-additional-full.csv'`, which is under this project directory, into the following folder of library `aif360`: `[path to library 'aif360']/aif360/data/raw/bank`.
 
## Installation

Run the following command to launch this tool:

	python3 run.py

After running the above commands, you can launch this tool by visiting http://127.0.0.1:8000/ on the local machine (Chrome is recommended). 

## Cite Our Paper

```bibtex
@inproceedings{yan2024exploring,
  title={Exploring Visualization for Fairness in AI Education},
  author={Yan, Xinyuan and Zhou, Youjia and Mishra, Arul and Mishra, Himanshu and Wang, Bei},
  booktitle={2024 IEEE 17th Pacific Visualization Conference (PacificVis)},
  pages={1--10},
  year={2024},
  organization={IEEE}
}
