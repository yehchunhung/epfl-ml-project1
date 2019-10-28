Group: TWN1

Steps to obtain our prediction result of the improved method:
	
	1) Place the raw data sets 'train.csv’ and 'test.csv'  in this folder.
	
	2) Execute the run program with the data processing option to create processed data sets.
	
	3) If you want use the hardcoded hyperparamaters, you can simply ignore the grid search
	option of the run program. However, you can also obtain the same results by including 
	this option (This is an exhaustive search and takes quite a lot of time)
	
	So what you need to do in your computer is to:
		
		a) Open terminal and navigate to this folder
		
		b) to execute 'run.py' type:
			
			i) python run.py -am -> to process and create data sets and train the model using hard 
			coded hyperparameters
			
			ii) python run.py -am -t -> to process and create data sets, tune the hyperparameters
			with a grid search with cross validation and then train the model (takes some time)
			
			Note: You can omit the ‘-am’ option after the processed data sets are created in this
			folder. You should not omit this option in the very first execution, otherwise
			the program will not find the processed data sets and crash.
			
	After running the program a file called 'Submit_TWN1’ will be present in this
	folder.
	
	You can find implementations of the six mandatory algorithms in the 'implementations.py' file,
	helper functions for these algorithms in the 'helper_functions.py' file and the functions required
	for data pre-processing and parameter tuning in the 'processing.py' file.
