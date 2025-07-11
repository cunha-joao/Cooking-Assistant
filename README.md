# Cooking-Assistant
## Introduction
This is a project for the curricular unit "Aprendizagem Organizacional". In this project there's a chatbot created using Python 3 and Ollama with the theme: Cooking Assistant.

## Basic Setup
* Setup a virtual environment (not mandatory, but recommended)
* Run ```bash pip install -r requirements.txt```
* Make sure to have Ollama installed by running ```bash ollama pull llama3 nomic-embed-text```
* In case you don't have the `vault.txt` file in the `vaults` folder find a digital recipe book on the internet (preferably PDF or txt) and upload it here using the ```bash python upload.py``` command
* If the PDF needs adjustments you can alter the `crop_pdfs.py` file and use the ```bash python crop_pdfs.py``` command
* To run locally on your terminal use ```bash python localrag.py```
* To run on the browser use ```bash python localrag.py --mode api``` and open the browser on `http://localhost:8000`