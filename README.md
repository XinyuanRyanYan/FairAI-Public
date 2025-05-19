# Fairness in AI E-book 

The Fairness in AI E-book is a web-based system that combines text, images, and interactive visualizations to educate non-technical users about fairness in AI, including decision-making by Machine Learning algorithms, quantifying bias in model outcomes, and applying algorithms to mitigate bias. 

Specifically, we designed six types of interactive visualizations that were utilized independently or collectively to illustrate different aspects of Fairness in AI. This tool has been employed in the "*Fair Algorithm for Business*" course offered by the School of Business.




## Dependencies

This software requires Python3, R, Flask and aif360 to run. 
If you do not have these packages installed, please use the following command to intall them.

	pip install 'aif360[all]'
	pip install flask
	pip install flask_session
	pip install flask_assets

### Import Dataset
To use the 5th chapter *Final Project* of this tool, you will need to place the file  `'bank-additional-full.csv'`, which is under this project directory, into the following folder of library `aif360`: `[path to library 'aif360']/aif360/data/raw/bank`.
 
## Installation

Run the following command to launch this tool:

	python3 run.py

After running the above commands, you can launch this tool by visiting http://127.0.0.1:8000/ on the local machine (Chrome is recommended).
