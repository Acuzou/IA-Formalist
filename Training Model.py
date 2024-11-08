# JSON Dataset builder - Main
import datetime
import os
from dotenv import load_dotenv
import pathlib
import json
import pdfplumber
import re
import numpy as np
import matplotlib.pyplot as plt
from evaluate import load #, evaluator # TO STUDY
#import torch
#print(torch.__version__)

from pprint import pprint
from functools import reduce
#import pandas as pd
from tqdm import tqdm
from datasets import load_metric, Dataset

# IMPORTATION DES MODELES PRE ENTRAINE
from transformers import Trainer, TrainingArguments, TrainerCallback, pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoConfig
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification
#from transformers import DistilBertForSequenceClassification, DistilBertForTokenClassification, DistilBertTokenizerFast

#from scikit-learn.preprocessing import LabelEncoder

# Pour m'authentifier à une API
#from transformers import HfFolder, HfApi
#HfFolder.save_token("hf_api_token")

# Charger le fichier .env
load_dotenv()

# API KEY
api_key = os.environ.get("HUGGINGFACE_API_KEY")

class Classification_Model:
    # https://huggingface.co/mistralai/Mistral-7B-v0.1
    
    # Possible models :
    # - distilbert-base-cased
    # - roberta-large
    # - bert-base-cased
    # - dslim/bert-base-NER # Token Classification Based
    # - HuggingFaceFW/fineweb-edu-classifier
    # - ProsusAI/finbert # Sequence Classification Based
    
    # - TinyLlama/llama-7b-hf
    # - TinyLlama/TinyLlama-1.1B-Chat-v1.0
    # - meta-llama/Meta-Llama-3-8B-Instruct
    # - electra-base-cased
    # - openai/whisper-large-v3
    # - mistralai/Mistral-7B-Instruct-v0.3
    # - mistralai/Codestral-22B-v0.1 # Specific pour le code
    # - mistralai/Mistral-7B-v0.3 -- Require Python 3.10.11
    # - mistralai/Mistral-7B-v0.1

    def __init__(self, model_name="distilbert-base-cased"):
        self.model_name = model_name
        self.path_dataset =  os.path.join(os.path.dirname(__file__), '..', 'data') 
        self.path_loss = os.path.join(os.path.dirname(__file__), '..', 'model', 'logs.json')
        self.path_json = "" #path_dataset + "attestation_propriete_train_dataset.json"
        self.normed_dataset = None
        self.snips_dataset = {"train": [], "test": []}
        self.token_classification_dataset = {"train": [], "test": []}
        self.sequence_classification_dataset = {"train": [], "test": []}
        self.tokenizer = None
        self.dataset_labels = None
        self.token_classification_model = None
        self.sequence_classification_model = None
        self.token_data_collator = None
        self.sequence_data_collator = None
        self.logs = {'Token Classification Model': [], 'Sequence Classification Model': []}
        
        
    # SET DATASET FOR TRAINING
    
    def load_dataset(self, json_name="attestation_propriete_train_dataset.json"):
        '''Function to load the dataset'''
        self.path_json = self.path_dataset + "/" + json_name      
        with open(self.path_json) as json_file:
            self.normed_dataset = json.load(json_file)

    def get_token_classification_tokenized_dataset(self, n_range=0):
        '''Function to get the token classification dataset for training'''
        return self.token_classification_dataset["train"][n_range]
    
    def get_sequence_classification_tokenized_dataset(self, n_range=0):
        '''Function to get the sequence classification dataset for training'''
        return self.sequence_classification_dataset["train"][n_range]

    def set_train_dataset(self):
        
        #On veut travailler avec l'équivalent tokenizé pour chaque sequence selon le modèle de BERT pré-entrainé
        #Celui ci est se trouve dans 'input_ids'
        #Une matrice d'attention 'attention_mask' est calculé de même pour comprendre le contexte
        #'labels' représente l'équivalent de 'token-label' prenant en compte le modèle BERT
        #En effet, certains mots se décompose en plusieurs tokens suivant le dictionnaire de BERT

        # Dataset en forme json
        utterances = []
        tokenized_utterances = []
        labels_for_tokens = []
        lines_utterances = []
        sequence_labels = []

        for keyPDF, valuePDF in self.normed_dataset["Data"].items():
            for keySection, valueSection in valuePDF.items():
                for keyAttribut, valueAttribut in valueSection.items():
                    for keyData, valueData in valueAttribut.items():
                        if type(valueData) == int:
                            valueData = [valueData]
                            #print("valueData : ", valueData)
                        for value in valueData:
                            if keyData == 'tokens':
                                utterances.append(' '.join(value))
                                tokenized_utterances.append(value)
                                sequence_labels.append(keyAttribut)
                            if keyData == 'tokens_labels':
                                labels_for_tokens.append(value)
                            if keyData == 'start_index':
                                lines_utterances.append(value)

        int_example = 50 
        print("\nExample of the dataset :")
        print("Integer example : ", int_example)
        print("Sequence label : ", sequence_labels[int_example], "/ Total Length : ", len(sequence_labels))
        print("Utterance : ", utterances[int_example], "/ Total Length : ", len(utterances))
        print("Tokens : ", tokenized_utterances[int_example], "/ Total Length : ", len(tokenized_utterances))
        print("Token labels : ", labels_for_tokens[int_example], "/ Total Length : ", len(labels_for_tokens))
        print("Line : ", lines_utterances[int_example], "/ Total Length : ", len(lines_utterances))

        self.unique_sequence_labels = list(set(sequence_labels))

        #Convertit les type de sequence de str à int en considérant leur indice
        #Crée un nombre entier unique (token) pour chaque type de sequence
        sequence_labels = [self.unique_sequence_labels.index(l) for l in sequence_labels]
        print(f'\nThere are {len(self.unique_sequence_labels)} unique sequence labels : {self.unique_sequence_labels}')

        #Crée un nombre entier unique (token) pour chaque type de token
        self.unique_token_labels = list(set(reduce(lambda x, y: x + y, labels_for_tokens)))
        labels_for_tokens = [[self.unique_token_labels.index(_) for _ in l] for l in labels_for_tokens]
        print(f'\nThere are {len(self.unique_token_labels)} unique token labels : {self.unique_token_labels}')

        # TODO : Prendre en compte le numero de ligne pour chaque token
        self.snips_dataset = Dataset.from_dict(
            dict(
                utterance=utterances, 
                label=sequence_labels,
                tokens=tokenized_utterances,
                token_labels=labels_for_tokens,
                #line_utterances=lines_utterances,
            )
        )

        #On entraine le dataset en considérant 80% des données chargés
        #Les derniers 20% seront utilisés pour tester les performances de notre modèle
        self.snips_dataset  = self.snips_dataset.train_test_split(test_size=0.2)
        print("\nSnips Dataset For Training [{}] : {}".format(int_example, self.snips_dataset['train'][int_example]))

        # Map our snip dataset to be for token classification
        self.token_classification_dataset = self.snips_dataset.map(self.tokenize_and_align_labels, batched=True)
        print("\nToken Classification Tokenized Snips [{}] : {}".format(int_example, self.token_classification_dataset['train'][int_example]))
        
        #We delete useless informations to improve algorithm efficency for token classification
        self.token_classification_dataset['train'] = self.token_classification_dataset['train'].remove_columns(
            ['utterance', 'label', 'tokens', 'token_labels'] #, 'line_utterances'
        )
        self.token_classification_dataset['test'] = self.token_classification_dataset['test'].remove_columns(
            ['utterance', 'label', 'tokens', 'token_labels'] # , 'line_utterances'
        )
        print("\nToken Classification Tokenized Snips Cleaned Formatted : {}".format(self.token_classification_dataset))

        # Same for Sequence Classification
        self.sequence_classification_dataset = self.snips_dataset.map(self.preprocess_function, batched=True)
        print("\nSequence Classification Tokenized Snips [{}] : {}".format(int_example, self.sequence_classification_dataset['train'][int_example]))

        #We delete useless informations to improve algorithm efficency for sequence classification
        self.sequence_classification_dataset['train'] = self.sequence_classification_dataset['train'].remove_columns(
            ['utterance', 'tokens', 'token_labels'] #, 'line_utterances'
        )
        self.sequence_classification_dataset['test'] = self.sequence_classification_dataset['test'].remove_columns(
            ['utterance', 'tokens', 'token_labels'] # , 'line_utterances'
        )
        print("\nSequence Classification Tokenized Snips Cleaned Formatted : {}".format(self.sequence_classification_dataset))
        print("\nDataset In Place !")
        return

    def tokenize_and_align_labels(self, data):
        
        # The given "token_labels" may not match up with the BERT wordpiece tokenization so
        #  this function will map them to the tokenization that BERT uses
        #  -100 is a reserved for labels where we do not want to calculate logs so BERT doesn't waste time
        #  trying to predict tokens like CLS or SEP
        
        tokenized_inputs = self.tokenizer(data["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(data[f"token_labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:  # Set the special tokens to -100.
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)  # CLS and SEP are labeled as -100
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
   
    def preprocess_function(self, data):
        '''Function to preprocess the dataset for sequence classification'''
        # simple function to batch tokenize utterances with truncation
        return self.tokenizer(data["utterance"], truncation=True)
    
    # SET PRE TRAINED MODEL
    def set_tokenizer(self):
        '''Function to load the pre-trained model for token classification'''   
        # On importe le tokenizer de "Model_Name" pre-entrainé
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) # .encode(), .decode()
        # tokenizer.encode("Madame Pierre DUPONT", truncation=True, padding=True), "labels": [1, 1, 1]})
        prompt = "Monsieur Pierre TANGAMA"
        print("\nExample of Tokenizer : {} became {}".format(prompt, self.tokenizer.encode((prompt)))) 
        return

    def set_token_classification_model(self):
        '''Function to load the pre-trained model for token classification'''
        # On charge le modèle pré-entrainé de "Model_Name" pour la classification de tokens. 
        # self.token_classification_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        print("\nSET PRE-TRAINED TOKEN CLASSIFICATION_MODEL : {}".format(self.model_name))
        self.token_classification_model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, 
            num_labels=len(self.unique_token_labels),
            ignore_mismatched_sizes=True,
            use_auth_token=api_key,
        )
        # On définie les noms de chaque type de token rajouté au modèle
        self.token_classification_model.config.id2label = {
            i: l for i, l in enumerate(self.unique_token_labels)
        }
        #print("ID2Label : {}".format(self.token_classification_model.config.id2label))
        print("\nToken Classification Model : {}".format(self.token_classification_model))
        
        # DataCollator crée un lot de données. Il pad également dynamiquement le texte pour qu'il soit de la même longueur que l'élément le plus long du lot, ce qui rend tous les éléments de la même longueur.
        # Il est possible de pad le texte dans la fonction du tokenizer en utilisant padding=True, mais le padding dynamique est plus efficace.
        # L'attention masquée (attention mask) est utilisée pour ignorer les scores d'attention pour les tokens de remplissage.
        self.token_data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        return
    
    def set_sequence_classification_model(self):
        '''Function to load the pre-trained model for token classification'''
        # On charge le modèle pré-entrainé de "Model_Name" pour la classification de séquences. 
        
        print("\nSET PRE-TRAINED SEQUENCE CLASSIFICATION_MODEL : {}".format(self.model_name))
        self.sequence_classification_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.unique_sequence_labels),
            ignore_mismatched_sizes=True,
            use_auth_token=api_key
        )
        # On définie les noms de chaque type d'attribut (séquence) rajouté au modèle
        self.sequence_classification_model.config.id2label = {
            i: l for i, l in enumerate(self.unique_sequence_labels)
        }
        #print("ID2Label : {}".format(self.sequence_classification_model.config.id2label))
        print("\nSequence Classification Model : {}".format(self.token_classification_model))
        
        # DataCollatorWithPadding crée un lot de données. Il pad également dynamiquement le texte pour qu'il soit de la même longueur que l'élément le plus long du lot, ce qui rend tous les éléments de la même longueur.
        # Il est possible de pad le texte dans la fonction du tokenizer en utilisant padding=True, mais le padding dynamique est plus efficace.
        # L'attention masquée (attention mask) est utilisée pour ignorer les scores d'attention pour les tokens de remplissage.
        self.sequence_data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return

    # TRAIN & TEST 
    
    def train_token_classification_model(self, epochs=1, batch_size=32):
        '''Function to train the model'''

        # epochs = Number of checkpoints
        # batch_size = Size of each batch

        training_args = TrainingArguments(
            output_dir=os.path.join(os.path.dirname(__file__), '..', 'model', 'tok_clf_model'), # Location to save snapshot of checkpoint
            logging_dir=os.path.join(os.path.dirname(__file__), '..', 'logs'),
            num_train_epochs=epochs, # Smaller number of epochs for testing = Smaller time of training
            per_device_train_batch_size=batch_size, # Size of each batch must be around 32 or 64
            per_device_eval_batch_size=batch_size, # Smaller batch size for testing = More chances to update the model
            load_best_model_at_end=True, # To ignore overtraining and only keep the best model

            logging_steps=10, # Number of step to output the log of the step
            log_level='info', # Level of details at highest 'info'
            eval_strategy='epoch',
            save_strategy='epoch'
        )
        
        # Define the loss logger

        #loss_logger = LossLogger()
        #callbacks=[loss_logger]

        # Define the trainer:

        trainer = CustomTrainer(
            model=self.token_classification_model,
            args=training_args,
            train_dataset=self.token_classification_dataset['train'],
            eval_dataset=self.token_classification_dataset['test'],
            #compute_metrics=self.compute_metrics,
            data_collator=self.token_data_collator,
        )

        # Evaluate and train the model

        print("\nTRAIN TOKEN CLASSIFICATION MODEL :\n")
        #for epoch in tqdm(range(epochs), desc="Evaluation before training...", unit="epoch", dynamic_ncols=True):
        #    trainer.evaluate()
            
        #for epoch in tqdm(range(epochs), desc="Training the model..."):
        #    trainer.train()
            
        #for epoch in tqdm(range(epochs), desc="Evaluation after training..."):
        #    trainer.evaluate()

        print("\nEvaluation before training...\n")
        print(trainer.evaluate())
        print("\nTraining the model...\n")
        print(trainer.train())
        print("\nEvaluation after training ...\n")
        print(trainer.evaluate())
        
        self.logs['Token Classification Model'].append(trainer.loss_logs)
        
        trainer.save_model()
        #self.plotLossFunction(loss_logger)
    
        return

    def train_sequence_classification_model(self, epochs=1, batch_size=32):
        '''Function to train the model'''

        # epochs = Number of checkpoints
        # batch_size = Size of each batch

        training_args = TrainingArguments(
            output_dir=os.path.join(os.path.dirname(__file__), '..', 'model', 'seq_clf_model'), # Location to save snapshot of checkpoint
            logging_dir=os.path.join(os.path.dirname(__file__), '..', 'logs'),
            num_train_epochs=epochs, # Smaller number of epochs for testing = Smaller time of training
            per_device_train_batch_size=batch_size, # Size of each batch must be around 32 or 64
            per_device_eval_batch_size=batch_size, # Smaller batch size for testing = More chances to update the model
            load_best_model_at_end=True, # To ignore overtraining and only keep the best model

            logging_steps=10, # Number of step to output the log of the step
            log_level='info', # Level of details at highest 'info' #log_level=logging.ERROR pour minimiser le temps d'execution
            eval_strategy='epoch', #'steps' other possibility
            save_strategy='epoch'
        )

        #Hyper Parameters must be optimised
        # some deep learning parameters that the Trainer is able to take in
        warmup_steps=len(self.sequence_classification_dataset['train']) // 5,  # number of warmup steps for learning rate scheduler,
        weight_decay = 0.05,


        # Define the trainer:

        trainer = CustomTrainer(
            model=self.sequence_classification_model,
            args=training_args,
            train_dataset=self.sequence_classification_dataset['train'],
            eval_dataset=self.sequence_classification_dataset['test'],
            #compute_metrics=self.compute_metrics,
            data_collator=self.sequence_data_collator
        )

        # Evaluate and train the model
        #print("\nTRAIN SEQUENCE CLASSIFICATION MODEL :\n")
        #for epoch in tqdm(range(epochs), desc="Evaluation before training..."):
        #    trainer.evaluate()
        #    
        #for epoch in tqdm(range(epochs), desc="Training the model..."):
        #    trainer.train()
        #    
        #for epoch in tqdm(range(epochs), desc="Evaluation after training..."):
        #    trainer.evaluate()
            
        print("\nTRAIN SEQUENCE CLASSIFICATION MODEL :\n")
        print("\nEvaluation before training\n")
        print(trainer.evaluate())
        print("\nTraining the model\n")
        print(trainer.train())
        print("\nEvaluation after training\n")
        print(trainer.evaluate())
        
        self.logs['Sequence Classification Model'].append(trainer.loss_logs)
        trainer.save_model()

        return
    
    def test_token_classification_model(self, example):
        '''Function to test the model'''
        pipe = pipeline(
            'token-classification', 
            model=self.token_classification_model, 
            tokenizer=self.tokenizer,
        )
        return pipe(example) 
    
    def test_sequence_classification_model(self, example):
        '''Function to test the model'''
        pipe = pipeline(
            'text-classification', 
            model=self.sequence_classification_model, # TODO : Enlever le "model =" 
            tokenizer=self.tokenizer,
        )
        return pipe(example)

        # If model saved in a directory ./snips_seq_clf.results
        ''' pipe = pipeline(
                'text-classification',
                './snips_seq_clf/results',
                tokenizer=self.tokenizer
            )
        '''

    # UTILS
    
    def plotLossFunction(self):
        '''Function to plot the loss function'''
        for loss in self.logs:
            plt.plot(loss)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Evolution de la fonction Coût (Loss Function)")
        plt.show()
    
    def load_losses(self, clean_loss=False):
        '''Function to load the logs'''
        json_loss = {}
        try:
            if (clean_loss):
                with open(self.path_loss, 'w') as json_file:
                    json.dump({}, json_file)
                    print("Le fichier Loss JSON situé à l'emplacement {} a été nettoyé.".format(self.path_loss))
            else:
                print("Import Recorded Losses...")
                with open(self.path_loss, 'r') as json_file:
                    json_loss = json.load(json_file)
        except Exception as e:
            print("Une erreur est survenue:", str(e))
        else:
            print("Création Loss JSON file au :", self.path_loss)
            json_loss["Type"] = "Attestation de propriété"
            json_loss["DateTime"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            
        index = len(json_loss.keys()) - 1
        json_loss["Loss N°" + str(index)] = self.logs
            
        with open(self.path_loss, 'w') as json_file:
            json.dump(json_loss, json_file, indent=4)

        return self.logs
    
    def freeze(model, n):
        '''Function to freeze n encoder layers'''

        #BERT MEDIUM = 12 encoders
        #BERT LARGE = 24 encoders
        #More param freezed = More speed = Less Accuracy

        ref_name = 'encoder.layer.' + str(n)
        for name, param in model.distilbert.parameters():
            if ref_name in name:
                break
            param.requires_grad = False  # disable training in BERT
        return model.distilbert.parameters

    def compute_metrics(self, eval_pred):
        '''Function to compute the metrics of the model'''
        # custom method to take in logits and calculate accuracy of the eval set
        # logits is a matrix of logits where the number of rows is the batch size and the number of columns is the number of sequences labels
        # labels are the true labels as a reference of correctly predicted labels
        # predictions are the predicted labels
        metric = load("precision") ### "accuracy", "squad", average="weighted", "macro", "micro", "samples", "binary"
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels) #, sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4]
        
    def get_parameters(self):
        parameters = [self.token_classification_model.distilbert.parameters, self.sequence_classification_model.distilbert.parameters]
        print("Parameters Token Classification : {}".format(parameters[0]))
        print("Parameters Sequence Classification : {}".format(parameters[1]))
        return parameters
    
    def get_config_model(self):
        '''Function to get the config of the model'''
        config = AutoConfig.from_pretrained(self.model_name)
        print("Configuration Model : {}".format(config))
        
    def generated_text(self, prompt):
        generator = pipeline('text-generation', model=self.token_classification_model, tokenizer=self.tokenizer)
        return generator(prompt, max_length=100)
        
# Class Trainer personnalisée pour stocker les valeurs de loss
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_logs = {}
        self.loss_logs['loss'] = []
        self.loss_logs['learning_rate'] = []
        self.loss_logs['grad_norm'] = []
        self.loss_logs['epoch'] = []
        
    def log(self, logs):
        super().log(logs)
        if 'loss' in logs:
            self.loss_logs['loss'].append(logs['loss']) 
            self.loss_logs['learning_rate'].append(logs['learning_rate'])
            self.loss_logs['grad_norm'].append(logs['grad_norm'])
            self.loss_logs['epoch'].append(logs['epoch'])

class LossLogger(TrainerCallback):
    
    def on_train_begin(self, args, state, control):
        self.logs = []

    def on_step_end(self, args, state, control):
        self.logs.append(state.log_history["loss"][-1])


    def on_train_epoch_end(self, args, state, control):
        print(f"Loss : {state.log_history[-1]['loss']}")

    #def on_train_end(self, args, state, control, **kwargs):
    #    print(f"Loss : {state.log_history[-1]['loss']}")

    #def on_epoch_end(self, args, state, control, **kwargs):
    #    print(f"Loss : {state.log_history[-1]['loss']}")
    
if __name__ == "__main__":

    json_name = "attestation_propriete_train_dataset.json"
    
    classification_model = Classification_Model()
    
    # SET TOKENIZER 
    classification_model.set_tokenizer()
    
    # SET DATASET
    classification_model.load_dataset(json_name=json_name)
    classification_model.set_train_dataset()
    
    # SEQUENCE CLASSIFICATION MODEL
    classification_model.set_sequence_classification_model()
    classification_model.train_sequence_classification_model()
    
    # TEST SEQUENCE CLASSIFICATION
    example = "Monsieur Pierre TANGAMA"
    print("\nTEST SEQUENCE CLASSIFICATION WITH EXAMPLE : {}".format(example))
    classification_model.test_sequence_classification_model(example)#, path_model=path_token_classification_trained_model)
    
    # TOKEN CLASSIFICATION MODEL
    classification_model.set_token_classification_model()
    classification_model.train_token_classification_model()
    
    # TEST TOKEN CLASSIFICATION
    print("\nTEST TOKEN CLASSIFICATION WITH EXAMPLE : {}".format(example))
    classification_model.test_token_classification_model(example)#, path_model=path_sequence_classification_trained_model)

    # UTILS
    #classification_model.freeze(classification_model.token_classification_model, 12)
    #classification_model.get_parameters()
    classification_model.load_losses()
    classification_model.plotLossFunction()
    classification_model.get_config_model()
    
    #prompt = "Salut Mistral ! Peux-tu m'expliquer ce que consiste à fine-tuner un modèle pré-entrainé d'intelligence artificielle ?"
    #print("\nDiscussion avec le modèle :")
    #print("\nPrompt : {}".format(prompt))
    #print(classification_model.generated_text(prompt)[0]['generated_text'])
    
    #classification_model.get_sequence_classification_tokenized_dataset()
    #classification_model.get_token_classification_tokenized_dataset()
    
    # TODO : Compute Metrics (One for Token Classification / Text Classification in Evaluate - HuggingFace)
    # TODO : Callbacks to print loss during training
    # TODO : Test the model on Mistral