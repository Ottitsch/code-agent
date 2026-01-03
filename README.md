A minimal Python AI agent Inspired by [ampcode](https://ampcode.com/how-to-build-an-agent), using the free Gemma 3 12B model.  
Demonstrates the core agent loop with tool calling and iterative reasoning.

setup:  
go to [aistudio](https://aistudio.google.com/) to get your free api key and create a .env with  
```GEMINI_API_KEY=[Insert Key]```
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd agents
python gemini.py
```

try it out:
```
what tools do you have?
----------
create fizzbuzz.py that I can run with python and that has fizzbuzz in it and executes it
----------
edit fizzbuzz.py so that it only prints until 15
----------
create a congrats.py script that rot13 decodes the following string 'Pbatenghyngvbaf ba ohvyqvat n pbqr rqvgvat ntrag!' and prints it
```

