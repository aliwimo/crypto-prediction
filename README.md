# Machine learning workspace
A work space to test machine learning models

## How to Use:
### Step 1: Create Virtual Environment
First of all, although it is optional (If you don't want to create virtual environment, just skip to Step 3), but it is recommended to create `Virtual Environment` using to following command:
```
python -m venv .venv
```

### Step 2: Source/Enable Virtual Environment
To enable the created virtual environment, run the following code if you are using `Unix` or `Mac` operating system:
```
source .venv/bin/activate
``` 
If you are using `Windows`, enable it using:
```
.\.venv\Scripts\activate
```


*If you want to deactivate virtual environment, run the following* 
```
deactivate
```



### Step 3: Install Required Packages

After that, start installing required packages list in `requirements.txt` using:
```
pip install -r requirements.txt
```


### Step 4: Run Your Server

After installing requirements, modify your script and run it
```
python app.py
```
