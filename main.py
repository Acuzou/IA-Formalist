import os
import json
import pdfplumber
import pathlib
import pprint
from transformers import pipeline, AutoTokenizer

class Formalist_IA_Model():
    #https://www.youtube.com/watch?v=tiZFewofSLM
    def __init__(self, path_token_classification_trained_model, path_sequence_classification_trained_model, model_name="distilbert-base-cased"):
        self.path_tok_clf_model = os.path.join(os.path.dirname(__file__), '..', 'model', 'tok_clf_model')
        self.path_seq_clf_model = os.path.join(os.path.dirname(__file__), '..', 'model', 'seq_clf_model')
        self.path_pdf_input = os.path.join(os.path.dirname(__file__), '..', 'pdf_input')
        self.path_json_output = os.path.join(os.path.dirname(__file__), '..', 'json_output')
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.output_json_name = ""
        self.number_of_pdf = ""
        self.number_of_pages_per_pdf = []
        self.texts_per_pdf = []
    
    def load_pdf(self):
        '''Chargement pdf'''
        paths_pdf = pathlib.Path(self.path_pdf_input).rglob('*.pdf')
        paths_pdf = [str(path_pdf) for path_pdf in paths_pdf]
        self.names_pdf = [os.path.basename(path_pdf) for path_pdf in paths_pdf]
        self.number_of_pdf = len(paths_pdf)
        
        for path_pdf in paths_pdf:
            with pdfplumber.open(path_pdf) as pdf:
                pages = pdf.pages
                self.number_of_pages_per_pdf.append(len(pages))
                self.texts_per_pdf.append([pages[i].extract_text() for i in range(self.number_of_pages_per_pdf[-1])])
                #self.lines = [self.texts[i].split("\num_match") for i in range(self.num_pages)]
            
    def read_pdf(self):
        '''Lecture pdf'''
        for i, texts in enumerate(self.texts_per_pdf):
            print("\nLecture du PDF {} en cours".format(self.names_pdf[i]))
            for num_page, page in enumerate(texts):
                print("\nNumero de Page : {}".format(num_page + 1))
                print("\nPage : {}".format(page))

    def test_token_classification_model(self, example):
        '''Function to test the model'''
        pipe = pipeline(
            'token-classification', #"ner"
            self.path_tok_clf_model,
            tokenizer=self.tokenizer,
            #use_auth_token=api_key
        )
        return pipe(example) # Possibility to passed several inputs pipe(examples) avec examples = list([example1, ..., exemple_n])
    
    def test_sequence_classification_model(self, example):
        '''Function to test the model'''
        pipe = pipeline(
            'text-classification',  #"sequence-classification", #"zero-shot-classification",
            self.path_seq_clf_model,
            tokenizer=self.tokenizer,
            #truncation = true
            #use_auth_token=api_key
        )
        return pipe(example)
    
    def token_classifier(self, path_json_output = os.path.join(os.path.dirname(__file__), '..', 'json_output'), output_json_name = "analysis_report.json"):
        '''Fonction de classification en token'''
        
        token_results = []
        for i, texts in enumerate(self.texts_per_pdf):
            print("\nLecture du PDF {} en cours".format(self.names_pdf[i]))
            token_results_per_pdf = []
            for num_page, page in enumerate(texts):
                #print("\nNumero de Page : {}\n".format(num_page + 1))
                #print("\nPage : {}\n".format(page))
                token_results_per_pdf.append(self.test_token_classification_model(page.replace("\n", " ")))
            token_results.append(token_results_per_pdf)   
            
        return token_results
        
    
if __name__ == "__main__":
    
    output_json_name = "analysis_report.json"
    
    path_token_classification_trained_model = ".\\snips_tok_clf\\results"
    path_sequence_classification_trained_model = ".\\snips_seq_clf\\results"
    
    print("\nFORMALIST IA MODEL\n")
    formalist_ia_model = Formalist_IA_Model(path_token_classification_trained_model, path_sequence_classification_trained_model)
    
    # LOAD PDF TO ANALYSE
    print("Loading PDF...")
    formalist_ia_model.load_pdf()
    #formalist_ia_model.read_pdf()
    
    #example = "Monsieur Pierre TANGAMA"
    #print("\nTEST TOKEN CLASSIFICATION MODEL : {}\n".format(example))
    #results = formalist_ia_model.test_token_classification_model(example)
    #for result in results:
    #    print(f"Token: {result['word']}, Label: {result['entity']}, Score: {result['score']}")

    results = formalist_ia_model.token_classifier()
    for results_per_pdf in results:
        for results_per_page in results_per_pdf:
            for result in results_per_page:
                print(f"Token: {result['word']}, Label: {result['entity']}, Score: {result['score']}")