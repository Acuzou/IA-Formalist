# JSON Dataset builder - Main
import os
import pathlib
import json
import pdfplumber
import re
from pprint import pprint
#import pandas as pd
#import numpy as np
from tqdm import tqdm

class Classifier_Attestation_Propriete:
  '''Convertit un PDF en fichier json classifiant les données selon leurs sequences et tokens associés pour générer des données d'entrainement'''
  
  def __init__(self, path_pdf):
    # Instanciation des attributs de l'objet PDFToJsonDataset
    self.path_pdf = path_pdf
    self.flatten_text = ""
    self.texts = []
    self.lines = []
    self.num_pages = 0
    self.token_visited = []

    # Importation PDF et Conversion Texte Brut
    with pdfplumber.open(self.path_pdf) as pdf:
      pages = pdf.pages
      self.num_pages = len(pages)
      self.texts = [pages[i].extract_text() for i in range(self.num_pages)]
      self.lines = [self.texts[i].split("\num_match") for i in range(self.num_pages)]

    self.set_flatten_text()

    self.json_dataset = {} # Contient tous les attributs extraits du pdf mise en forme
  
    #for line in self.lines[0]:
    #  print(line)
  
  # CONSTRUCT THE JSON DATASET FOR SEQUENCE AND TOKEN CLASSIFICATION

  def extract(self):
    '''Renvoie en format JSON le Dataset extrait du pdf en format pdf pour ensuite pouvoir être entrainé par l'IA'''

    self.set_flatten_text()
    if len(self.flatten_text) == 0:
       return {}

    self.json_dataset["Sections"] = self.get_json_sections()
    self.json_dataset["Notaire"] = self.get_json_notaire_1()
    self.json_dataset["Qualification Juridique"] = self.get_json_qualif_juridique_2()
    self.json_dataset["Etat Civil Décès"] = self.get_json_etat_civil_D_3()
    self.json_dataset["Etat Civil Heritier"] = self.get_json_etat_civil_H_3()
    self.json_dataset["Designation"] = self.get_json_designation_4()
    self.json_dataset["Evaluation"] = self.get_json_evaluation_5()
    self.json_dataset["Etat Descriptif de Division"] = self.get_json_etat_descriptif_6()
    self.json_dataset["Effet Relatif"] = self.get_json_effet_relatif_7()
    self.json_dataset["Droits Transmis"] = self.get_json_droits_transmis_8()
    self.json_dataset["Requisition Publication"] = self.get_json_requisition_publication_9()
    self.json_dataset["Certification Attestation"] = self.get_json_certification_attestation_10()
    self.json_dataset["Certification Identite"] = self.get_json_certification_identite_11()
    self.json_dataset["Contexte"] = self.get_json_context()
    
    return self.json_dataset

  # Useful Functions

  def set_flatten_text(self):
    '''Réinitialise la valeur du texte aplati entre un String de tout le pdf'''
    self.flatten_text = ""
    for i in range(self.num_pages):
      self.flatten_text = self.flatten_text + self.texts[i]
    return

  def update_flatten_text(self, first_quote = "", last_quote = "", index_before = 0): 
    '''Actualise le text aplati à partir d'une citation'''
    self.set_flatten_text()

    if len(first_quote) >= 1:
        text_split = re.split(first_quote, self.flatten_text, maxsplit=1)
        self.flatten_text = text_split[1]
    if len(last_quote) >= 1:
        text_split_zoom = re.split(last_quote, self.flatten_text, maxsplit=1)
        self.flatten_text = text_split_zoom[0]

    if len(first_quote) == 0:
        last_index = index_before
    else:
        last_index = len(text_split[0].replace("\n", " "))
    return last_index
  
  def get_match(self, pattern, index_before = 0, n_groups = 0, replace = ""):
    '''Renvoie la valeur de l'expression extraite et sa position en ligne si celui-ci existe'''

    # EXPLICATION :
    # re = REGULAR EXPRESSION ou REGEX
    # pattern est un exemple d'Expression Régulière
    # re.search(pattern, self.flatten_text) effectue un recherche d'expression régulière selon le language REGEX (REGular EXpression) pour détecter les expressions à extraire
    # flags=re.DOTALL précise qu'il ne doit pas prendre en compte les sauts de ligne
    # re.sub(r'[\n,.]', ' ', text) pour remplacer les sauts de lignes, les virgules et les points par des espaces
    # match.group(n_groups) renvoie le groupe extrait n° n_groups --> 0 == Tout / 1 == Juste la partie entre 1ère parenthèse du REGEX / 2 == Dans la deuxième parenthèse / etc 
    # text.strip() supprimer les espaces inutiles
    # match.start() est l'indice du caractère au début de l'expression extraite
    # len(text[:match.start()].splitlines()) compte le nombre de ligne avant cette expression extraite

    match = re.search(pattern, self.flatten_text, flags=re.DOTALL)
    if match:
      expression = re.sub(r'[\n]', ' ', match.group(n_groups)).strip()
      
      index_position = [index_before + match.start(n_groups) + 1]# Verifier si les indices correspondent 
      n = len(expression)
      for word in expression.split(" "):
          index_position.append(index_position[-1] + len(word) + 1)
      index_position = index_position[:-1]
        
      # end_index = index_before + match.end(n_groups) # + len(self.flatten_text[:match.start()+1].splitlines()) 
      if(len(replace)>=1):
        expression = expression.replace(replace, " ").strip()
      return [[expression], [index_position]]
    else:
      #print("No Data")
      return [[], []]

  def get_matches(self, pattern, index_before = 0, n_groups = 0, replace = "", selected_groups=[]):
    '''Renvoie la valeur de l'expression extraite et sa position en ligne si celui-ci existe'''

    # EXPLICATION :
    # re = REGULAR EXPRESSION ou REGEX
    # pattern est un exemple d'Expression Régulière
    # re.finditer(pattern, self.flatten_text) effectue un recherche d'expression régulière selon le language REGEX (REGular EXpression) pour détecter les expressions à extraire
    # flags=re.DOTALL précise qu'il ne doit pas prendre en compte les sauts de ligne
    # re.sub(r'[\n,.]', ' ', text) pour remplacer les sauts de lignes, les virgules et les points par des espaces
    # match.group(n_groups) renvoie le groupe extrait n° n_groups --> 0 == Tout / 1 == Juste la partie entre 1ère parenthèse du REGEX / 2 == Dans la deuxième parenthèse / etc
    # text.strip() supprimer les espaces inutiles
    # match.start() est l'indice du caractère au début de l'expression extraite
    # len(text[:match.start()].splitlines()) compte le nombre de ligne avant cette expression extraite

    matches_iter = re.finditer(pattern, self.flatten_text, flags=re.DOTALL)
    matches, positions = [], []

    for match in matches_iter:
        
        expression = ""
        if len(selected_groups) == 0:
            expression = expression + re.sub(r"[\n]", " ", match.group(n_groups)).strip()
            if(len(replace)>=1):
                expression = expression.replace(replace, " ").strip()

        for n_groups in selected_groups:
            expression = expression + re.sub(r"[\n]", " ", match.group(n_groups)).strip()
            if(len(replace)>=1):
                expression = expression.replace(replace, " ").strip()
                
        index_position = [index_before + match.start(n_groups) + 1]# Verifier si les indices correspondent 
        n = len(expression)
        for word in expression.split(" "):
            index_position.append(index_position[-1] + len(word) + 1)
        index_position = index_position[:-1]

        matches.append(expression)
        positions.append(index_position)

    return [matches, positions]
  
  # GET JSON DATASET

  def get_json_sections(self):
    '''Renvoie le JSON Dataset des différentes Clauses de l'acte'''
    self.set_flatten_text()

    # Ces attributs décrivent la structure des multiples parties indépendantes à analyser d'un acte d'une attestation de propriété
    named_label = "Sections"
    subdataset = {}
    subdataset["Texte_Applicable_S"] = self.set_normed_data_default(self.get_match(r"TEXT[E|ES]\sAPPLICABLE", n_groups=0), named_label)
    subdataset["Personne_Decedee_S"] = self.set_normed_data_default(self.get_match(r"(PERSONN[E|ES]\sD[E|É]C[E|É]D[E|É][E|ES])", n_groups=0), named_label)
    subdataset["Devolution_Successorale_S"] = self.set_normed_data_default(self.get_match(r"D[E|É]VOLUTION\sSUCCESSORALE", n_groups=0), named_label)
    subdataset["Qualities_Hereditaires_S"] = self.set_normed_data_default(self.get_match(r"QUALIT[E|ES]\sH[E|É]R[E|É]DITAIR[E|ES]", n_groups=0), named_label)
    subdataset["Visa_des_Actes_S"] = self.set_normed_data_default(self.get_match(r"VISA\sDES\sACTES", n_groups=0), named_label)
    subdataset["Acceptation_de_la_Succession_S"] = self.set_normed_data_default(self.get_match(r"ACCEPTATION\sDE\sLA\sSUCCESSION", n_groups=0), named_label)
    subdataset["Designation_S"] = self.set_normed_data_default(self.get_match(r"D[E|É]SIGNATION", n_groups=0), named_label)
    subdataset["Lotissement_S"] = self.set_normed_data_default(self.get_match(r"Lotissement", n_groups=0), named_label)
    subdataset["Evaluation_S"] = self.set_normed_data_default(self.get_match(r"[E|É]VALUATION", n_groups=0), named_label)
    subdataset["Effet_Relatif_S"] = self.set_normed_data_default(self.get_match(r"EFFET\sRELATIF", n_groups=0), named_label)
    subdataset["Origine_de_Propriete_S"] = self.set_normed_data_default(self.get_match(r"ORIGINE\sDE\sPROPRI[E|É]T[E|É]", n_groups=0), named_label)
    subdataset["Servitude_S"] = self.set_normed_data_default(self.get_match(r"SERVITUDES", n_groups=0), named_label)
    subdataset["Droits_Transmis_S"] = self.set_normed_data_default(self.get_match(r"DROITS\sTRANSMIS", n_groups=0), named_label)
    subdataset["Requisition_Publication_S"] = self.set_normed_data_default(self.get_match(r"R[E|É]QUISITION\s-\sPUBLICATION", n_groups=0), named_label)
    subdataset["Pouvoir_S"] = self.set_normed_data_default(self.get_match(r"POUVOIR", n_groups=0), named_label)
    subdataset["Certification_Attestation_S"] = self.set_normed_data_default(self.get_match(r"CERTIFICATION\sET\sATTESTATION", n_groups=0), named_label)
    subdataset["Autorisation_Destruction_Documents_S"] = self.set_normed_data_default(self.get_match(r"AUTORISATION\sDE\sDESTRUCTION\sDES\s\DOCUMENTS\sET\sPI[E|È]CES", n_groups=0), named_label)
    subdataset["Mention_Protection_Données_Personnelles_S"] = self.set_normed_data_default(self.get_match(r"MENTION\sSUR\sLA\sPROTECTION\sDES\sDONN[E|É]ES\sPERSONNELLES", n_groups=0), named_label)
    subdataset["Certification_Identite_S"] = self.set_normed_data_default(self.get_match(r"CERTIFICATION\sD’IDENTIT[E|É]", n_groups=0), named_label)
    return subdataset

  def get_json_notaire_1(self):
    '''Renvoie le JSON Dataset des informations du notaire associé à l'acte'''
    # L'AN DEUX MILLE VINGT-QUATRE,
    # LE date en lettres
    # A ville (département), adresse, au siège de l’Office Notarial, ci-après nommé,
    # Maître nom, Notaire Associé ou au sein de la Société Civile Professionnelle «dénomination »
    # titulaire d’un Office Notarial à ville (code postal), adresse
    
    self.set_flatten_text()

    subdataset = {} 
    subdataset["Annee_1"] = self.set_normed_data_default(self.get_match(r"(L\s*\'\s*A\s*N.*?,)", n_groups=1), named_label="Annee_Lettre")
    subdataset["Date_1"] = self.set_normed_data_default(self.get_match(r",(.*?)\nA\s", n_groups=1), named_label="Date_Lettre")
    subdataset["Lieu_SON_1"] = self.set_normed_data_lieu(self.get_match(r"(?:\nA\s|le[\s\n]siège[\s\n]est[\s\n]à)(.*?)(?:\,*[\s\n]au[\s\n]siège|\,*\s*titulaire)", n_groups=0))
    subdataset["Lieu_ON_1"] = self.set_normed_data_lieu(self.get_match(r"(titulaire\n*\s*d\'*’*un\n*\s*[oO]ffice\n*\s*[nN]otarial.*?)A\sreçu\n*\s*le\n*\s*présent\n*\s*acte", n_groups=0))
    subdataset["Nom_Maitre_1"] = self.set_normed_data_name(self.get_match(r"(Ma[î|i]tre.*?)[nN]otaire", n_groups=1))
    subdataset["Type_Maitre_1"] = self.set_normed_data_default(self.get_match(r"[nN]otaire\s*\,*(.*?)de\n*\s*la\n*\s*[sS]ociété", n_groups=0), named_label="Type_Maitre")
    
    #print(subdataset["Annee_1"])
    n = self.update_flatten_text(first_quote="[nN]otaire")
    subdataset["Nom_Societe_1"] = self.set_normed_data_nom_societe(self.get_match(r"(?:[\"«](.*?)[»\"]|[A-ZÉÈ\d-]{2,}\,)", index_before=n, n_groups=0))
    
    return subdataset

  def get_json_qualif_juridique_2(self):
    '''Renvoie le JSON Dataset des qualifications juridiques l'acte'''

    # QUALIFICATION JURIDIQUE
    # A reçu le présent acte contenant ATTESTATION IMMOBILIERE APRES DECES à la requête de :…
   
    n = self.update_flatten_text(first_quote=r"pr[ée]sent\sacte", last_quote=r"la\srequ[êe]te")
    subdataset = {}   
    subdataset["Type_Document_2"] = self.set_normed_data_default(self.get_match(r"([A-ZÉÈ]{2,}.*?)à", index_before=n, n_groups=1), named_label="Type_Document")
    
    self.set_flatten_text()

    if re.search(r"à[\s\n]la[\s\n]requ[êe]te[\s\n]de", self.flatten_text):
        #r"(Ci-après\snommés|en\svertu\sdes\spouvoirs)"
        n = self.update_flatten_text(first_quote=r"à[\s\n]la[\s\n]requ[êe]te[\s\n]de", last_quote="TEXTE\sAPPLICABLE", index_before=n)
        subdataset["Requerant_Acte_2"] = self.set_normed_data_qualif_juridique(self.get_matches(r"-(.*?)pr[ée]sent[\s\n]à[\s\n]l\’acte", index_before=n, n_groups=0))
    else:
        subdataset["Requerant_Acte_2"] = {
            "tokens": [],
            "token_labels": [],
            "line": [0]
        }
    return subdataset

  def get_json_etat_civil_D_3(self):
    '''Renvoie le JSON Dataset de la comparutions clients (état civil) des parties prenantes de l'acte'''

    # PERSONNE DECEDEE
    # NOM en majuscules, prénoms en minuscules, domicile, date et lieu de naissance
    # Profession, nom du conjoint ou partenaire PACS, date et lieu du mariage, date divorce, veuvage
    # Nationalité, qualité de résident ou non au sens de la réglementation fiscale.
  
    # r"PERSONN(?:E|ES)\sD(?:E|É)C(?:E|É)D(?:E|É)(?:E|ES)"
    self.set_flatten_text()
    if re.search(r"PERSONNE[S]*\sD[EÉ]C[EÉ]D[EÉ]E[S]*", self.flatten_text):
       n = self.update_flatten_text(first_quote=r"PERSONNE[S]*\sD[EÉ]C[EÉ]D[EÉ]E[S]*", last_quote=r"D[E|É]VOLUTION\sSUCCESSORALE")
    
    #if r"PERSONNE[S]\sD[EÉ]C[EÉ]D[EÉ]E[S]" in self.flatten_text:
    #    print("TEST A")
    #    n = self.update_flatten_text(first_quote=r"PERSONNE[S]\sD[EÉ]C[EÉ]D[EÉ]E[S]", last_quote=r"D[E|É]VOLUTION\sSUCCESSORALE")
    else:
        print("PAS DE DECES") 
        n = self.update_flatten_text(first_quote=r"D[E|É]VOLUTION\sSUCCESSORALE")
    subdataset = {}

    subdataset["Nom_D_3"] = self.set_normed_data_name(self.get_match(r"\b(Monsieur|Madame)\b(.*?)[A-ZÉÈ]{2,}\,", index_before=n, n_groups=0))
    subdataset["Domicile_D_3"] = self.set_normed_data_lieu(self.get_match(r"demeurant\n*\s*à(.*?)\.", index_before=n, n_groups=0))
    subdataset["Lieu_Naissance_D_33"] = self.set_normed_data_lieu(self.get_match(r"(\b(Né|Née)\b\sà.*?)(\.|le)", index_before=n, n_groups=0))
    subdataset["Date_Naissance_D_3"] = self.set_normed_data_date(self.get_match(r"\sle\s(.*?)[\.\n]", index_before=n, n_groups=0))
    subdataset["Lieu_Deces_D_3"] = self.set_normed_data_lieu(self.get_match(r"(Décédé(?:e)?\s*à\s.*?)(\.|le)", index_before=n, n_groups=1))
    subdataset["Date_Deces_D_3"] = self.set_normed_data_date(self.get_match(r"Décédé[e]*\sà(?:.*)(\sle\s.*?[\.\n])", index_before=n, n_groups=1)) 
    subdataset["Profession_D_3"] = self.set_normed_data_common_name(self.get_match(r"([Ee]n\n*\s*son\n*\s*vivant\s)(.*?)\,", index_before=n, n_groups=0), pattern=r"([Ee]n\n*\s*son\n*\s*vivant\s)(.*?)\,", named_label="Profession")
    subdataset["Situation_Maritale_D_3"] = self.set_normed_data_situation_maritale(self.get_match(r"\b(?:Né|Née)\b\sà(?:.*?)\.(.*?)\.", index_before=n, n_groups=1)) #[4:] sur le groupe # TODO: A revoir
    #subdataset["Specification_Maritale_D_3"] = self.set_normed_data_default(self.get_match(r"\b(Né|Née)\b\sà([^\.]*)\.(.*?)\.(.*?)\.", index_before=n, n_groups=4), named_label="Spe_Marital")
    subdataset["Nationalite_D_3"] = self.set_normed_data_common_name(self.get_match(r"([Dd]e[\s\n][Nn]ationalit[ée])(.*?)\.", index_before=n, n_groups=0), pattern=r"([Dd]e[\s\n][Nn]ationalit[ée])(.*?)\.", named_label="Nationalite")
    subdataset["Qualite_Resident_D_3"] = self.set_normed_data_default(self.get_match(r"[Dd]e[\s\n][Nn]ationalit[ée].*?\.\s(.*?)[.(]", index_before=n, n_groups=1), named_label="Qual_Resident")

    ###subdataset["Ville_D_D_3"] = self.get_match(r"Décédé(?:e)? à\s(.*?)\(", index_before=n, n_groups=1)
    ###n = self.update_flatten_text(first_quote="Décédé")
    ###expression = self.get_matches(r"\((.*?)\)", index_before=n, n_groups=1)
    ###subdataset["Departement_D_D_3"] = expression[0][0][0], expression[1][0]
    ###subdataset["Pays_D_D_3"] = expression[0][1][0], expression[1][1]

    return subdataset
  
  def get_json_etat_civil_H_3(self):
    '''Renvoie le JSON Dataset de la comparutions clients (état civil) des parties prenantes de l'acte'''
    # Héritiers
    # NOM en majuscules, prénoms en minuscules, domicile, date et lieu de naissance
    # Profession, nom du conjoint ou partenaire PACS, date et lieu du mariage, date divorce, veuvage
    # Nationalité, qualité de résident ou non au sens de la réglementation fiscale.
    subdataset = {}
    n = self.update_flatten_text(first_quote=r"D[E|É]VOLUTION\sSUCCESSORALE", last_quote=r"\b(Prédécès\ssans\spostérité|QUALIT[E|É]S\sH[E|É]R[E|É]DITAIRES|D[E|É]SIGNATION)\b")

    subdataset["Nom_H_3"] = self.set_normed_data_name(self.get_matches(r"\b(Madame|Monsieur)\b(.*?)[A-ZÉÈ]{2,}\,", index_before=n, n_groups=0))
    subdataset["Profession_H_3"] = self.set_normed_data_common_name(self.get_matches(r"[A-ZÉÈ]{2,}\,([^\n]*?)\,", index_before=n, n_groups=0), pattern=r"([A-ZÉÈ]{2,}\,)([^\n]*?)\,",named_label="Profession", set_prefixe=False)
    subdataset["Domicile_H_3"] = self.set_normed_data_lieu(self.get_matches(r"demeurant\s*\n*à(.*?)\.", index_before=n, n_groups=0))
    subdataset["Lieu_Naissance_H_3"] = self.set_normed_data_lieu(self.get_matches(r"\b(Né|Née)\b\sà\s(.*?)\sle\s", index_before=n, n_groups=0))
    subdataset["Date_Naissance_H_3"] = self.set_normed_data_date(self.get_matches(r"\b(?:Né|Née)\b\sà\s(?:.*?)(\sle\s.*?)[\.\n]", index_before=n, n_groups=1))
    subdataset["Situation_Maritale_H_3"] = self.set_normed_data_situation_maritale(self.get_matches(r"\b(?:Né|Née)\b\sà(?:.*?)\.(.*?)\.", index_before=n, n_groups=1))
    #subdataset["Specification_Maritale_H_3"] = self.set_normed_data_default(self.get_matches(r"\b(Né|Née)\b\s(.*?)\.(.*?)\.(.*?)\.", index_before=n, n_groups=4), named_label="Spe_Marital")
    subdataset["Nationalite_H_3"] = self.set_normed_data_common_name(self.get_matches(r"[Dd]e[\s\n][Nn]ationalit[ée](.*?)\.", index_before=n, n_groups=0), pattern=r"([Dd]e[\s\n][Nn]ationalit[ée])(.*?)\.", named_label="Nationalite")
    subdataset["Qualite_Resident_H_3"] = self.set_normed_data_default(self.get_matches(r"[Dd]e[\s\n][Nn]ationalit[ée].*?\.\s(.*?)[.(]", index_before=n, n_groups=1), named_label="Qual_Resident")
    subdataset["Relation_Familiale_H_3"] = self.set_normed_data_common_name(self.get_matches(r"[Ss](on|a)\s([^\s]*?)\.", index_before=n, n_groups=0), pattern=r"([Ss](?:on|a)\s)([^\s]*?)\.", named_label="Relation_Familiale")
    return subdataset

  def get_json_designation_4(self):
    '''Renvoie le JSON Dataset de la désignation complète des immeubles de l'acte'''

    # Nature du bien (terrain, parcelle, maison…), commune, adresse
    # Tableau cadastral avec préfixe (éventuellement), section, lieudit, surface
    #n = self.update_flatten_text(last_quote=r"[E|É]VALUATION")

    # n est le nombre de lignes au début de la partie filtrée
    # n = self.update_flatten_text(first_quote=r"\b(D[E|É]SIGNATION|IMMEUBLE[S]*\sDE\sCOMMUNAUTE|IMMEUBLE[S]*\sPROPRE[S]*)\b", last_quote=r"[E|É]VALUATION")
    n = self.update_flatten_text(first_quote=r"D[E|É]SIGNATION", last_quote=r"EFFET\sRELATIF")

    subdataset = {}
    pattern_prefixe_TC = r"\b([Ff]igurant\s(?:ainsi\s)*au\scadastre|L\'assiette\sde\sla\svolumétrie\sest\sla\ssuivante|cadastré)\b"
    
    # print("Flatten text : ", self.flatten_text[:1000])
    #subdataset["Total_Descriptif_P_4"] = self.get_match(r"\n*(.*?)" + pattern_prefixe_TC, index_before=n, n_groups=1) 
    subdataset["Adresse_Propriete_4"] = self.set_normed_data_lieu(self.get_match(r"[A-ZÉÈ-]{3,}(.*?)\,\n", index_before=n, n_groups=0))
    subdataset["Nature_P_4"] = self.set_normed_data_default(self.get_match(r"\,\n(.*?)" + pattern_prefixe_TC, index_before=n, n_groups=1), named_label="Nature_P")
    subdataset["Prefixe_TC_4"] = self.set_normed_data_default(self.get_match(pattern_prefixe_TC, index_before=n, n_groups=0), named_label="Prefixe_TC")

    #n = self.update_flatten_text(first_quote=pattern_prefixe_TC, last_n_lines=n)
    # Pattern pour extraire le contenu du tableau cadastral
    pattern = (
        r'([A-Z]+)'                 # Groupe 1: Premier mot en MAJUSCULES
        r'\s+(\d+)'                 # Groupe 2: Premier nombre
        r'\s([a-zA-Zéèîï\d\s]+[a-zA-Zéèîïûü]+)'  # Groupe 3: Entre le premier nombre et un mot en majuscules suivi d'un nombre
        r'\s(\d{2}\s[a-z]{2}\s\d{2}\s[a-z]{1}\s\d{2}\s[a-z]{2})' # Groupe 4: Motif spécifique commençant par un nombre
    )

    #subdataset["Tableau_Cadastral_4"] = self.get_matches(r"([A-Z]+)\s+(\d+)\s([a-zA-Zéèîï\d\s]+[a-zA-Zéèîïûü]+)\s(\d{2}\s[a-z]{2}\s\d{2}\s[a-z]{1}\s\d{2}\s[a-z]{2})", index_before=n, n_groups=0)
    subdataset["Tableau_Cadastral_4"] = self.set_normed_data_tableau_cadastral(self.get_matches(pattern, index_before=n, n_groups=0))
    return subdataset

  def get_json_evaluation_5(self):
    '''Renvoie le JSON Dataset de l'évaluation des immeubles associé à l'attestation de propriété'''
    #Evaluation en pleine propriété
    #Evaluation des quotités transmises (si bien transmis en partie et/ou démembrement)

    # TODO : Evaluation en Lettre

    n = self.update_flatten_text(first_quote=r"[E|É]VALUATION", last_quote=r"EFFET\sRELATIF")
    subdataset = {}
    #r"(\d+[\d\.\s]+\,\d{2}\sEUR)"
    #subdataset["Eval_Pleine_Propriete_5"] = self.get_match(r"\d+[\d\.\s]+\,\d{2}\s*(EUR|€)", index_before=n , n_groups=0)
    Evaluations_5 = self.get_matches(r"(\d+[\d\.\s]+\,\d{2}\s*(EUR|€))", index_before=n, n_groups=1)
    #Evaluations_5[:][-1], Evaluations_5[1][-1]
    subdataset["Eval_Pleine_Propriete_5"] = self.set_normed_data_argent([[Evaluations_5[0][0]], [Evaluations_5[1][0]]])
    subdataset["Eval_Quotites_Transmises_5"] = self.set_normed_data_argent([[Evaluations_5[0][-1]], [Evaluations_5[1][-1]]])
    return subdataset

  def get_json_etat_descriptif_6(self):
    '''Renvoie le JSON Dataset des références de publication de l'état descritif de division  de l'acte'''
    # Référence de publication de l'état descritif de division de ses évent mddificatifs 
    # (Si bien soumis au régime de la copropriété)
    # r"(D[E|É]SIGNATION|IMMEUBLE[S]*\sDE\sCOMMUNAUTE|IMMEUBLE[S]*\sPROPRE[S]*)"
    n = self.update_flatten_text(first_quote=r"D[E|É]SIGNATION")
    subdataset = {}
    #r"\b(nue\n*-\n*propri[é|e]t[é|e]|copropri[é|e]t[é|e]|pleine\s*\n*-*\s*\n*propri[é|e]t[é|e])\b"
    subdataset["Regime_Propriete_6"] = self.set_normed_data_default(self.get_match(r"\b(nue\s*-\s*propri[ée]t[ée]|copropri[ée]t[ée]|pleine\s*-*\s*propri[ée]t[ée])\b", index_before=n, n_groups=1), "Regime_Propriete")
    n = self.update_flatten_text(last_quote=r"EFFET\sRELATIF", index_before=n)
    #print("Flatten text : ", self.flatten_text[:1000])
    # TODO : Corriger le pattern pour cas plusieurs effets relatifs
    subdataset["Reference_Etat_Descritif_Division_1_6"] = self.set_normed_data_reference(self.get_matches(r"volume\s*\d+\,*\s*num[ée]ro\s*\d+\.*", index_before=n, n_groups=0))
    return subdataset

  def get_json_effet_relatif_7(self):
    '''Renvoie le JSON Dataset des références de publication du titre immédiat de l'acte'''

    # Effet Relatif
    # Type d’acte, Notaire ayant reçu l’acte, lieu
    # SPF où l’acte a été publié ou mention « en cours de publication, date de publication,
    # volume, numéro.
    # Exemple : Acquisition suivant acte reçu par Maître XXX, notaire à XXX le XXX en cours de publication au service de la publicité foncière de CORBEIL 1. 
    # / publié au service de la publicité foncière de XXX, le XXX, volume XXX, numéro XXX.
    
    self.set_flatten_text()
    if not re.search(r"EFFET\sRELATIF", self.flatten_text):
        return {}
    
    n = self.update_flatten_text(first_quote=r"EFFET\sRELATIF", last_quote="SERVITUDES")
    subdataset={}
    subdataset["Type_Acte_7"] = self.set_normed_data_default(self.get_matches(r"([^\n][^\n]*)\s*suivant\n*\s*acte\n*\s*reçu\n*\s*par", index_before=n, n_groups=1, replace="-"), named_label="Type_Acte")
    subdataset["Notaire_Effet_Relatif_7"] = self.set_normed_data_name(self.get_matches(r"suivant\n*\s*acte\n*\s*reçu\n*\s*par(.*?)\,", index_before=n, n_groups=1))
    subdataset["Lieu_Reception_7"] = self.set_normed_data_lieu(self.get_matches(r"lors\n*\s*[nN]otaire\n*\s*à(.*?)\d{4}\,", index_before=n, n_groups=0))
    subdataset["Lieu_Publicite_Fonciere_7"] = self.set_normed_data_lieu(self.get_matches(r"(au\s*\n*service\s*\n*de\n*\s*la\n*\s*publicité\n*\s*foncière\n*\s*de.*?)\,*\s*volume", index_before=n, n_groups=1))
    subdataset["Reference_Effet_Relatif_7"] = self.set_normed_data_reference(self.get_matches(r"volume\s*\d+\,*\s*num[ée]ro\s*\d+\.*", index_before=n, n_groups=0))
    subdataset["Date_Reception_7"] = self.set_normed_data_date(self.get_matches(r"[nN]otaire\n*\s*à.*?\b(le|les)\b(.*?),", index_before=n, n_groups=2, selected_groups=[2]))
    subdataset["Date_Publicite_Fonciere_7"] = self.set_normed_data_date(self.get_matches(r"publicité\n*\s*foncière\n*\s*de\n*\s*.*?\b(le|les)\b(.*?)\,*\svolume", index_before=n, n_groups=2, selected_groups=[2]))
    subdataset["Statut_Publication_7"] = self.set_normed_data_default(self.get_matches(r"\b(publié|en[\s\n]cours[\s\n]de[\s\n]publication)", index_before=n, n_groups=0), "Statut_Publication")
    return subdataset
  
  def get_json_droits_transmis_8(self):
    '''Renvoie le JSON Dataset des droits transmis de la succession'''

    # Le notaire soussigné atteste que, par suite du décès , les biens et droits immobiliers dont la désignation précède se sont trouvés transmis aux ayants droit en leur qualité
    # Madame XXX recueille « quotité »
    # Madame XXX recueille « quotité » 
    
    n = self.update_flatten_text(first_quote=r"DROITS\sTRANSMIS", last_quote=r"R[E|É]QUISITION-PUBLICATION")
    subdataset={}
    subdataset["Nom_Droits_Transmis_8"] = self.set_normed_data_name(self.get_matches(r"([^\n][^\n]*)recueille", index_before=n, n_groups=0))
    subdataset["Valeur_Transmis_8"] = self.set_normed_data_valeur_transmis(self.get_matches(r"(recueille.*?)\b(Monsieur|Madame|REQUISITION\s-\sPUBLICATION)", index_before=n, n_groups=1))
    #subdataset["Droits_Transmis_Quotites_8"] = self.get_matches(r"recueille(.*?)\n", index_before=n, n_groups=1)
    return subdataset

  def get_json_requisition_publication_9(self):
    '''Renvoie le JSON Dataset de la Réquisition - Publication de l'acte'''
    # L’“ ayant droit ” requiert le notaire soussigné de dresser la présente attestation de propriété pour la faire publier.
    # La présente attestation de propriété sera publiée au service de la publicité foncière de XXX.
    # En fonction des dispositions à publier au fichier immobilier, la contribution de sécurité immobilière s'élève à la somme de XXX.
    # La taxe fixe sera perçue par ce service de la publicité foncière.

    n = self.update_flatten_text(first_quote=r"R[E|É]QUISITION\s*-\s*PUBLICATION", last_quote=r"POUVOIRS")
    subdataset = {}
    subdataset["Lieu_Requisition_PF_9"] = self.set_normed_data_lieu(self.get_match(r"Au[\n\s]service[\n\s]de[\n\s]la[\n\s]publicit[e|é][\n\s]fonci[e|è]re[\n\s]d(.*?)\.", index_before=n, n_groups=0))
    subdataset["Contribution_Securite_Immobiliere_9"] = self.set_normed_data_argent(self.get_match(r"contribution[\n\s]de[\n\s]s[e|é]curit[e|é][\n\s]immobili[e|è]re[\n\s]s\'[e|é]l[e|è]ve[\n\s]à[\n\s]la[\n\s]somme[\n\s]de(.*?)\.", index_before=n, n_groups=0))
    #subdataset["Taxe_Publicite_Fonciere_9"] = self.set_normed_data_argent(self.get_match(r"(TOTAL\s\d+[\d\.]+*\,\d{2})\n", index_before=n, n_groups=0))
    #subdataset["Contribution_Securite_Immobiliere_9"] = self.get_match(r"(\d+[\d\.]+\,\d{2})([^\d]*?)", index_before=n, n_groups=1)
    
    return subdataset

  def get_json_certification_attestation_10(self): 

    # PAR SUITE DES FAITS ET ACTES SUS-ENONCES, le notaire soussigné certifie et atteste que les biens immobiliers faisant l’objet des présentes, appartiennent à :
    # Madame XXX à concurrence de XXX.
    # Madame XXX à concurrence de XXX.

    n = self.update_flatten_text(first_quote=r"CERTIFICATION\sET\sATTESTATION", last_quote=r"CERTIFICATION\sD’IDENTIT[E|É]")
    subdataset = {} 
    subdataset["Personnes_Certifiees_10"] = self.set_normed_data_name(self.get_matches(r"\b(Madame|Monsieur)\b(.*?)[A-ZÉÈ-]{2,}", index_before=n, n_groups=0))
    subdataset["Limites_Concurrence_10"] = self.set_normed_data_valeur_transmis(self.get_matches(r"(à[\s\n]concurrence[\s\n]d.*?)(?:\b(Madame|Monsieur|AUTORISATION)\b)", index_before=n, n_groups=1))
    return subdataset

  def get_json_certification_identite_11(self):
    '''Renvoie le JSON Dataset de la Certification d'Identité devant apparaitre à la fin de l'acte'''

    # Le notaire soussigné certifie que l’identité complète des parties dénommées dans le présent document telle qu'elle est indiquée en tête des présentes à la suite de leur nom ou dénomination lui a été régulièrement justifiée.

    n = self.update_flatten_text(first_quote=r"CERTIFICATION\sD’IDENTIT[E|É]")
    subdataset = {}
    subdataset["Certification_Identite_11"] = self.set_normed_data_default(self.get_match(r"(.*?)\.", index_before=n, n_groups=0), named_label="Cert_Id_11")
    return subdataset

  def get_json_context(self):
    '''Renvoie le JSON Dataset d'autres informations'''
    subdataset = {}
    self.set_flatten_text()
    words = self.flatten_text.replace("\n", " ").split(" ")
    subdataset["Context"] = {
        "tokens": [], 
        "tokens_labels": [], 
        "start_index": []
    }
    
    index_position = 0
    for word in words:
        if word not in self.token_visited:
            subdataset["Context"]["tokens"].append(word)
            subdataset["Context"]["tokens_labels"].append("O")
            subdataset["Context"]["start_index"].append(index_position)
        index_position = index_position + len(word) + 1
    return subdataset
    
  # Token Labeling for Classification

  def set_token_labels(self, token, named_label):
      '''Génère la liste des types de tokens selon la liste des tokens extraits'''
      token_labels = []
      token_labels.append("B-" + named_label)
      self.token_visited.append(token[0])
      num_words = len(token)
      if num_words >= 2:
          for j in range(num_words-2):
              token_labels.append("I-" + named_label)
              self.token_visited.append(token[j+1])
          token_labels.append("E-" + named_label)
          self.token_visited.append(token[-1])
      return token_labels
  
  def set_normed_data_default(self, found_match, named_label):
    '''Met en bonne forme les données extraites'''
    #Ex : {"token": ["Ceci", "est", "une", "phrase"], "token_labels": ["Pronom", "Verbe", "Déterminant", "Nom_Commun"], "line": 100}
    num_match, tokens, tokens_labels = len(found_match[0]), [], []
    for i in range(num_match):
        token_list = found_match[0][i].split(" ")
        tokens.append(token_list)
        token_labels = self.set_token_labels(token_list, named_label)
        tokens_labels.append([token_label for token_label in token_labels])
    return {
        "tokens": tokens, 
        "tokens_labels": tokens_labels, 
        "start_index": found_match[1]
    }
  
  def set_normed_data_lieu(self, found_match):
    '''Met en bonne forme les données extraites'''
    #Ex : {"token": ["Ceci", "est", "une", "phrase"], "token_labels": ["Pronom", "Verbe", "Déterminant", "Nom_Commun"], "start_index": 100}
    
    num_match, tokens, tokens_labels = len(found_match[0]), [], []
    for i in range(num_match):
        token, token_labels = [], []
        prefixe_match = re.search(r"(.*?)[A-ZÉÈ]{2,}", found_match[0][i])
        token = token + prefixe_match.group(1).strip().split(" ") if prefixe_match else token
        token_labels = token_labels + self.set_token_labels(prefixe_match.group(1).strip().split(" "), "Prefixe_Ville") if prefixe_match else token_labels

        ville_match = re.search(r"([A-ZÉÈ-]{2,})", found_match[0][i])
        token = token + ville_match.group(1).strip().split(" ") if ville_match else token
        token_labels = token_labels + self.set_token_labels(ville_match.group(1).strip().split(" "), "Ville") if ville_match else token_labels

        parenthese_match = re.findall(r"(\(.*?\)(?:\s\d{5})*)", found_match[0][i])
        num_match = len(parenthese_match)

        if num_match >= 1:
            departement_token = str(parenthese_match[0])
            token.append(departement_token)
            token_labels = token_labels + self.set_token_labels([departement_token], "Departement")

        if num_match >= 2:
            pays_token = str(parenthese_match[1])
            token.append(pays_token)
            token_labels = token_labels + self.set_token_labels([pays_token], "Pays")

        #[\)\,]\,*\s([^\,]*?)(?:\,|\.|A\sreçu|$)
        #adresse_match = re.search(r"[\)\,]\,*\s(?:\(.*\))*(?:\s*\d{5})*\,*\s(.*?)(?:\,|\.|A\sreçu|$)", found_match[0][i])
        adresse_match = re.search(r"[\)\,]\,*\s([^\,]*?)(?:\,|\.|A\sreçu|$)", found_match[0][i])
        token = token + adresse_match.group(1).strip().split(" ") if adresse_match else token
        token_labels = token_labels + self.set_token_labels(adresse_match.group(1).strip().split(" "), "Adresse") if adresse_match else token_labels

        suffixe_match = re.search(r"[\)\,]\,*\s(?:[^\,]*)((\,|\.|A\sreçu).*?)", found_match[0][i])
        token = token + suffixe_match.group(1).strip().split(" ") if suffixe_match else token
        token_labels = token_labels + self.set_token_labels(suffixe_match.group(1).strip().split(" "), "Suffixe_Ville") if suffixe_match else token_labels

        tokens.append(token)
        tokens_labels.append(token_labels)

    return {
        "tokens": tokens, 
        "tokens_labels": tokens_labels, 
        "start_index": found_match[1]
    }

  def set_normed_data_date(self, found_match):
    '''Met en bonne forme les données extraites'''
    #Ex : {"token": ["Ceci", "est", "une", "phrase"], "token_labels": ["Pronom", "Verbe", "Déterminant", "Nom_Commun"], "start_index": 100}
    #print(found_match)
    
    num_match, tokens, tokens_labels = len(found_match[0]), [], []
    for i in range(num_match):
        token, token_labels = [], []
        prefixe_match = re.search(r"(le\s)", found_match[0][i])
        token = token + prefixe_match.group(1).strip().split(" ") if prefixe_match else token
        token_labels = token_labels + self.set_token_labels(prefixe_match.group(1).strip().split(" "), "Prefixe_Date") if prefixe_match else token_labels

        jour_match = re.search(r"(\d{1,2})", found_match[0][i])
        token = token + jour_match.group(1).strip().split(" ") if jour_match else token
        token_labels = token_labels + self.set_token_labels(jour_match.group(1).strip().split(" "), "Jour") if jour_match else token_labels

        mois_match = re.search(r"([A-Za-zéèû]+)\s", found_match[0][i])
        token = token + mois_match.group(1).strip().split(" ") if mois_match else token
        token_labels = token_labels + self.set_token_labels(mois_match.group(1).strip().split(" "), "Mois") if mois_match else token_labels

        annee_match = re.search(r"(\d{4})", found_match[0][i])
        token = token + annee_match.group(1).strip().split(" ") if annee_match else token
        token_labels = token_labels + self.set_token_labels(annee_match.group(1).strip().split(" "), "Annee") if annee_match else token_labels

        tokens.append(token)
        tokens_labels.append(token_labels)

    return {
        "tokens": tokens, 
        "tokens_labels": tokens_labels, 
        "start_index": found_match[1]
    }
  
  def set_normed_data_name(self, found_match):
    '''Met en bonne forme les données extraites'''
    #Ex : {"token": ["Ceci", "est", "une", "phrase"], "token_labels": ["Pronom", "Verbe", "Déterminant", "Nom_Commun"], "start_index": 100}
    num_match, tokens, tokens_labels = len(found_match[0]), [], []
    for i in range(num_match):
        match = re.search(r"\b(Monsieur|Madame|Ma[î|i]tre)\b(.*?)([A-ZÉÈ\\/-]{3,})", found_match[0][i])
        if match:
            token = match.group(1).strip().split(" ") + match.group(2).strip().split(" ") + match.group(3).strip().split(" ")
            civilite = self.set_token_labels(match.group(1).strip().split(" "), "Civilite")
            prenom = self.set_token_labels(match.group(2).strip().split(" "), "Prenom")
            nom = self.set_token_labels(match.group(3).strip().split(" "), "Nom")
            tokens.append(token), 
            tokens_labels.append(civilite + prenom + nom)
    return {
        "tokens": tokens,
        "tokens_labels": tokens_labels,
        "start_index": found_match[1]
    }
  
  def set_normed_data_nom_societe(self, found_match):
    '''Met en bonne forme les données extraites'''
    #Ex : {"token": ["Ceci", "est", "une", "phrase"], "token_labels": ["Pronom", "Verbe", "Déterminant", "Nom_Commun"], "start_index": 100}
    if len(found_match[0]) == 0:
        return {"tokens": [], "tokens_labels": [], "start_index": 0}
    else:
        notaires = re.findall(r"([A-Za-z-éèîïÉÈ]+\,*)\s([A-ZÉÈ]{2,}\,*)\s", found_match[0][0])
        majuscules = re.findall(r"([A-ZÉÈ\d-]{2,}\,*)", found_match[0][0])
        prenom_labels = [notaire[0] for notaire in notaires]
        nom_labels = [notaire[1] for notaire in notaires]
        token = found_match[0][0].split(" ") 
        token_labels = []
        for word in token:
            if word in prenom_labels:
                token_labels.append("Prenom_Societe")
            elif word in nom_labels:
                token_labels.append("Nom_Societe")
            elif word in ["\"", "«", "»"]:
                token_labels.append("Encadrement_Societe")
            elif word in majuscules:
                token_labels.append("Nom_Societe_Maj")
            else:
                token_labels.append("Nom_Societe_Min")
            self.token_visited.append(word)
        return {
            "tokens": [token],
            "tokens_labels": [token_labels],
            "start_index": found_match[1]
        }

  def set_normed_data_qualif_juridique(self, found_match):   
    '''Met en bonne forme les données extraites'''
    #Ex : {"token": ["Ceci", "est", "une", "phrase"], "token_labels": ["Pronom", "Verbe", "Déterminant", "Nom_Commun"], "start_index": 100}
    num_match = len(found_match[0])
    if num_match == 0:
        return {"tokens": [], "tokens_labels": [], "start_index": 0}
    else:
        tokens = []
        tokens_labels = []
        for i in range(num_match):
            requerant = re.search(r"-\s*([Mm]adame|[Mm]onsieur)(.*?)([A-ZÉÈ]{2,})\,*", found_match[0][i])
            tokens.append([requerant.group(1).strip(), requerant.group(2).strip(), requerant.group(3).strip()])
            civilite_requerant = self.set_token_labels(requerant.group(1).strip().split(" "), "Civilite")
            prenom_requerant = self.set_token_labels(requerant.group(2).strip().split(" "), "Prenom")
            nom_requerant = self.set_token_labels(requerant.group(3).strip().split(" "), "Nom")
            tokens_labels.append(civilite_requerant  + prenom_requerant + nom_requerant)
        return {
            "tokens": tokens,
            "tokens_labels": tokens_labels,
            "start_index": found_match[1]
        }
     
  def set_normed_data_nationalite(self, found_match):
      '''Met en bonne forme les données extraites'''
      num_match, tokens, tokens_labels = len(found_match[0]), [], []
      for i in range(num_match):
          token, token_labels = found_match[0][i].split(" "), []
          prefixe_match = re.search(r"[Dd]e[\s\n][Nn]ationalit[ée]", found_match[0][i])
          if prefixe_match:
              prefixe = prefixe_match.group(0).strip().split(" ")
              token = token + prefixe
              token_labels = token_labels + self.set_token_labels(prefixe, "Prefixe_Nationalite")
        
          profession_match = re.search(r"[Dd]e[\s\n][Nn]ationalit[ée](.*?)$", found_match[0][i])
          if profession_match:
              profession = profession_match.group(1).strip().split(" ")
              token = token + profession
              token_labels = token_labels + self.set_token_labels(profession, "Nationalite")

          tokens.append(token)
          tokens_labels.append(token_labels)

      return {
          "tokens": tokens,
          "tokens_labels": tokens_labels,
          "start_index": found_match[1]
      }
      
  def set_normed_data_situation_maritale(self, found_match):
    
    '''Met en bonne forme les données extraites'''
    #Ex : {"token": ["Ceci", "est", "une", "phrase"], "token_labels": ["Pronom", "Verbe", "Déterminant", "Nom_Commun"], "start_index": 100}
    #print(found_match)
    num_match, tokens, tokens_labels = len(found_match[0]), [], []
    for i in range(num_match):
        token, token_labels = [], []
        situation_match = re.search(r"(.*?)\b(Monsieur|Madame)\b", found_match[0][i])
        token = token + situation_match.group(1).strip().split(" ") if situation_match else token
        token_labels = token_labels + self.set_token_labels(situation_match.group(1).strip().split(" "), "Situation_Marital") if situation_match else token_labels

        civilite_conjoint_match = re.search(r"(\b(Monsieur|Madame)\b)", found_match[0][i])
        token = token + civilite_conjoint_match.group(1).strip().split(" ") if civilite_conjoint_match else token
        token_labels = token_labels + self.set_token_labels(civilite_conjoint_match.group(1).strip().split(" "), "Civilite") if civilite_conjoint_match else token_labels

        prenom_conjoint_match = re.search(r"\b(?:Monsieur|Madame)\b(.*?)[A-ZÉÈ]{2,}", found_match[0][i])
        token = token + prenom_conjoint_match.group(1).strip().split(" ") if prenom_conjoint_match else token
        token_labels = token_labels + self.set_token_labels(prenom_conjoint_match.group(1).strip().split(" "), "Prenom_Conjoint") if prenom_conjoint_match else token_labels

        nom_conjoint_match = re.search(r"([A-ZÉÈ]{2,})", found_match[0][i])
        token = token + nom_conjoint_match.group(1).strip().split(" ") if nom_conjoint_match else token
        token_labels = token_labels + self.set_token_labels(nom_conjoint_match.group(1).strip().split(" "), "Nom_Conjoint") if nom_conjoint_match else token_labels

        suffixe_match = re.search(r"[A-ZÉÈ]{2,}\s(.*?)", found_match[0][i])
        token = token + suffixe_match.group(1).strip().split(" ") if suffixe_match and len(suffixe_match.group(1).strip()) > 1 else token
        token_labels = token_labels + self.set_token_labels(suffixe_match.group(1).strip().split(" "), "Suffixe_SM") if suffixe_match and len(suffixe_match.group(1).strip()) > 1 else token_labels

        tokens.append(token)
        tokens_labels.append(token_labels)

    return {
        "tokens": tokens, 
        "tokens_labels": tokens_labels, 
        "start_index": found_match[1]
    }

  def set_normed_data_common_name(self, found_match, pattern = "", named_label = "", set_prefixe = True):
      '''Met en bonne forme les données extraites'''
      #Ex : {"token": ["Ceci", "est", "une", "phrase"], "token_labels": ["Pronom", "Verbe", "Déterminant", "Nom_Commun"], "start_index": 100}
      num_match, tokens, tokens_labels = len(found_match[0]), [], []
      for i in range(num_match):
          token, token_labels = [], []
          match = re.search(r"{}".format(pattern), found_match[0][i])
          if match :
              if set_prefixe:
                  prefixe = match.group(1).strip().split(" ")
                  token = token + prefixe
                  token_labels = token_labels + self.set_token_labels(prefixe, f"Prefixe_{named_label}")

                  #profession_match = re.search(r"{}(.*?)$".format(pattern[:-7]), found_match[0][i])
                  #match = re.search(r"{}".format(pattern), found_match[0][i])
                  common_name = match.group(2).strip().split(" ")
                  token = token + common_name
                  token_labels = token_labels + self.set_token_labels(common_name, named_label)
              else:
                  common_name = match.group(1).strip().split(" ")
                  token = token + common_name
                  token_labels = token_labels + self.set_token_labels(common_name, named_label)
        
          tokens.append(token)
          tokens_labels.append(token_labels)
      
      return {
          "tokens": tokens, 
          "tokens_labels": tokens_labels, 
          "start_index": found_match[1]
      }
  
  def set_normed_data_argent(self, found_match):
      '''Met en bonne forme les données extraites'''
      #Ex : {"token": ["Ceci", "est", "une", "phrase"], "token_labels": ["Pronom", "Verbe", "Déterminant", "Nom_Commun"], "start_index": 100}
      num_match, tokens, tokens_labels = len(found_match[0]), [], []
      for i in range(num_match):
        token, token_labels = [], []

        prefixe_match = re.search(r"([^\d]+?)\d+[\.\s]\d+\,\d{2,}[\s\n\t]", found_match[0][i])
        token = token + prefixe_match.group(1).strip().split(" ") if prefixe_match else token
        token_labels = token_labels + self.set_token_labels(prefixe_match.group(1).strip().split(" "), "Prefixe_Argent") if prefixe_match else token_labels

        millier_euro_match = re.search(r"(\d+?[\.\s])\d+\,\d{2,}[\s\n\t]", found_match[0][i]) # TODO : Formule à revoir
        token = token + millier_euro_match.group(1).strip().split(" ") if millier_euro_match else token
        token_labels = token_labels + self.set_token_labels(millier_euro_match.group(1).strip().split(" "), "Millier_Euro") if millier_euro_match else token_labels

        euro_match = re.search(r"[\.\s\(](\d+?\,)\d{2,}[\s\n\t]", found_match[0][i])
        token = token + euro_match.group(1).strip().split(" ") if euro_match else token
        token_labels = token_labels + self.set_token_labels(euro_match.group(1).strip().split(" "), "Euro") if euro_match else token_labels

        centime_euro_match = re.search(r"[\.\s\(]\d+\,(\d{2,})[\s\n\t]", found_match[0][i])
        token = token + centime_euro_match.group(1).strip().split(" ") if centime_euro_match else token
        token_labels = token_labels + self.set_token_labels(centime_euro_match.group(1).strip().split(" "), "Centime_Euro") if centime_euro_match else token_labels

        devise_match = re.search(r"\b(EUR|€|eur\)|eur)\b", found_match[0][i])
        token = token + devise_match.group(1).strip().split(" ") if devise_match and len(devise_match.group(1).strip()) > 1 else token
        token_labels = token_labels + self.set_token_labels(devise_match.group(1).strip().split(" "), "Devise") if devise_match else token_labels
        
        suffixe_match = re.search(r"\,\d{2,}\s*[EUReur€]+(.*?)$", found_match[0][i])
        token = token + suffixe_match.group(1).strip().split(" ") if suffixe_match and len(suffixe_match.group(1).strip()) > 1 else token
        token_labels = token_labels + self.set_token_labels(suffixe_match.group(1).strip().split(" "), "Suffixe_Argent") if suffixe_match else token_labels

        tokens.append(token)
        tokens_labels.append(token_labels)

      return {
          "tokens": tokens, 
          "tokens_labels": tokens_labels, 
          "start_index": found_match[1]
      }

  def set_normed_data_reference(self, found_match):
      '''Met en bonne forme les données extraites'''
      #Ex : {"token": ["Ceci", "est", "une", "phrase"], "token_labels": ["Pronom", "Verbe", "Déterminant", "Nom_Commun"], "start_index": 100}
      num_match, tokens, tokens_labels = len(found_match[0]), [], []
      for i in range(num_match):
        token, token_labels = [], []
        volume_match = re.search(r"(volume.*?)num", found_match[0][i])
        token = token + volume_match.group(1).strip().split(" ") if volume_match else token
        token_labels = token_labels + self.set_token_labels(volume_match.group(1).strip().split(" "), "Volume_Ref") if volume_match else token_labels

        numero_match = re.search(r"(num.*?)$", found_match[0][i])
        token = token + numero_match.group(1).strip().split(" ") if numero_match else token
        token_labels = token_labels + self.set_token_labels(numero_match.group(1).strip().split(" "), "Numero_Ref") if numero_match else token_labels

        tokens.append(token)
        tokens_labels.append(token_labels)

      return {
          "tokens": tokens, 
          "tokens_labels": tokens_labels, 
          "start_index": found_match[1]
      }
  
  def set_normed_data_valeur_transmis(self, found_match):
      '''Met en bonne forme les données extraites'''
      #Ex : {"token": ["Ceci", "est", "une", "phrase"], "token_labels": ["Pronom", "Verbe", "Déterminant", "Nom_Commun"], "start_index": 100}
      num_match, tokens, tokens_labels = len(found_match[0]), [], []
      for i in range(num_match):
        token, token_labels = [], []

        prefixe_match = re.search(r"\b(recueille|[àa][\s\n]concurrence)\b", found_match[0][i])
        token = token + prefixe_match.group(0).strip().split(" ") if prefixe_match else token
        token_labels = token_labels + self.set_token_labels(prefixe_match.group(1).strip().split(" "), "Prefixe_Part") if prefixe_match else token_labels

        part_en_lettre_match = re.search(r"\b(?:recueille|[àa][\s\n]concurrence)\b(.*?)\b(?:\(|\sen\s)\b", found_match[0][i])
        token = token + part_en_lettre_match.group(1).strip().split(" ") if part_en_lettre_match else token
        token_labels = token_labels + self.set_token_labels(part_en_lettre_match.group(1).strip().split(" "), "Part_En_Lettre") if part_en_lettre_match else token_labels

        part_en_chiffre_match = re.search(r"(\(.*?\))", found_match[0][i])
        token = token + part_en_chiffre_match.group(1).strip().split(" ") if part_en_chiffre_match else token
        token_labels = token_labels + self.set_token_labels(part_en_chiffre_match.group(1).strip().split(" "), "Part_En_Chiffre") if part_en_chiffre_match else token_labels

        type_propriete_match = re.search(r"\)*(\sen\s.*?)$", found_match[0][i])
        token = token + type_propriete_match.group(1).strip().split(" ") if type_propriete_match else token
        token_labels = token_labels + self.set_token_labels(type_propriete_match.group(1).strip().split(" "), "Type_Propriete") if type_propriete_match else token_labels

        tokens.append(token)
        tokens_labels.append(token_labels)

      return {
          "tokens": tokens, 
          "tokens_labels": tokens_labels, 
          "start_index": found_match[1]
      }
  
  def set_normed_data_tableau_cadastral(self, found_match):
      '''Met en bonne forme les données extraites'''
      #Ex : {"token": ["Ceci", "est", "une", "phrase"], "token_labels": ["Pronom", "Verbe", "Déterminant", "Nom_Commun"], "start_index": 100}
      num_match, tokens, tokens_labels = len(found_match[0]), [], []
      for i in range(num_match):
        token, token_labels = [], []

        section_match = re.search(r"([A-Z]+)", found_match[0][i])
        token = token + section_match.group(0).strip().split(" ") if section_match else token
        token_labels = token_labels + self.set_token_labels(section_match.group(1).strip().split(" "), "Section") if section_match else token_labels

        numero_match = re.search(r"\s+(\d+)", found_match[0][i])
        token = token + numero_match.group(1).strip().split(" ") if numero_match else token
        token_labels = token_labels + self.set_token_labels(numero_match.group(1).strip().split(" "), "Numero") if numero_match else token_labels

        lieudit_match = re.search(r"[A-Z]+\s+\d+\s([a-zA-Zéèîï\d\s]+[a-zA-Zéèîïûü]+)\s\d{2}\s[a-z]{2}\s\d{2}\s[a-z]{1}\s\d{2}\s[a-z]{2}", found_match[0][i])
        token = token + lieudit_match.group(1).strip().split(" ") if lieudit_match else token
        token_labels = token_labels + self.set_token_labels(lieudit_match.group(1).strip().split(" "), "Lieudit") if lieudit_match else token_labels

        surface_match = re.search(r"\s(\d{2}\s[a-z]{2}\s\d{2}\s[a-z]{1}\s\d{2}\s[a-z]{2})", found_match[0][i])
        token = token + surface_match.group(1).strip().split(" ") if surface_match else token
        token_labels = token_labels + self.set_token_labels(surface_match.group(1).strip().split(" "), "Surface") if surface_match else token_labels

        tokens.append(token)
        tokens_labels.append(token_labels)

      return {
          "tokens": tokens, 
          "tokens_labels": tokens_labels, 
          "start_index": found_match[1]
      }
  
  # Display JSON Datasets

  def get_json_dataset(self):
    return self.json_dataset

  def display_json_dataset(self):
    #print("\nJSON DATASET : {}".format(self.get_json_dataset()))
    print("\nJSON DATASET : ")
    pprint(self.get_json_dataset())
    return

  def get_json_subdataset(self, num_json):
    '''Renvoie le json associé à son numéro num_json'''
    if num_json == 0:
      return self.get_json_sections()
    if num_json == 1:
      return self.get_json_notaire_1()
    if num_json == 2:
      return self.get_json_qualif_juridique_2()
    if num_json == 3:
      return self.get_json_etat_civil_3()
    if num_json == 4:
      return self.get_json_designation_4()
    if num_json ==5:
      return self.get_json_evaluation_5()
    if num_json==6:
      return self.get_json_etat_descriptif_6()
    if num_json==7:
      return self.get_json_effet_relatif_7()
    if num_json==8:
      return self.get_json_droits_transmis_8()
    if num_json==9:
      return self.get_json_requisition_publication_9()
    if num_json==10:
      return self.get_json_certification_attestation_10()
    if num_json==11:
      return self.get_json_certification_identite_11()
    if num_json==12:
      return self.get_json_context()
    return {}
      
  def display_json_subdataset(self, nums_sections):
    '''Affiche les JSON datasets associés aux numéros de sections en entrée'''
    if 0 in nums_sections:
      print("\n0) JSON Dataset - Sections :\n")
      pprint(self.get_json_sections())
    if 1 in nums_sections:
      print("\n1) JSON Dataset - Notaire :\n")
      pprint(self.get_json_notaire_1())
    if 2 in nums_sections:
      print("\n2) JSON Dataset - Qualification Juridique :\n")
      pprint(self.get_json_qualif_juridique_2())
    if 3 in nums_sections:
      print("\n3) JSON Dataset - Etat Civil :\n")
      pprint(self.get_json_etat_civil_D_3())
      print("\n")
      pprint(self.get_json_etat_civil_H_3())
    if 4 in nums_sections:
      print("\n4) JSON Dataset - Designation :\n")
      pprint(self.get_json_designation_4())
    if 5 in nums_sections:
      print("\n5) JSON Dataset - Evaluation :\n")
      pprint(self.get_json_evaluation_5())
    if 6 in nums_sections:
      print("\n6) JSON Dataset - Références de publication de l'état descriptif de division\n")
      pprint(self.get_json_etat_descriptif_6())
    if 7 in nums_sections:
      print("\n7) JSON Dataset - Références de publication du titre immédiat :\n")
      pprint(self.get_json_effet_relatif_7())
    if 8 in nums_sections:
      print("\n8) JSON Dataset - Droits Transmis :\n")
      pprint(self.get_json_droits_transmis_8())
    if 9 in nums_sections:
      print("\n9) JSON Dataset - Réquisition - Publication :\n")
      pprint(self.get_json_requisition_publication_9())
    if 10 in nums_sections:
      print("\n10) JSON Dataset - Certification et Attestation :\n")
      pprint(self.get_json_certification_attestation_10())
    if 11 in nums_sections:
      print("\n11) JSON Dataset - Certification d'Identité :\n")
      pprint(self.get_json_certification_identite_11())
    if 12 in nums_sections:
      print("\n11) JSON Dataset - Contexte :\n")
      pprint(self.get_json_context())
    return
  
  def display_first_info(self):
    '''Affiche les premières informations extraites'''
    print("\nFirst Info :\n")
    print(self.texts[0])
    return
  
def extract_pdf(path_data_folder, path_json_folder, clean_dataset=False, selected_datasets=[], display_sections=[], num_max_data=0, print_first_info=False):
    '''Extrait tous les pdf présents dans le dossier (+sous-dossier) associé au chemin path_folder'''

    paths_pdf = pathlib.Path(path_data_folder).rglob('*.pdf')
    paths_pdf = [str(path_pdf) for path_pdf in paths_pdf]
    path_json = path_json_folder + "/attestation_propriete_train_dataset.json"

    json_dataset = {}

    try:
        if (clean_dataset):
            with open(path_json, 'w') as json_file:
                json.dump({}, json_file)
            print("Le fichier JSON situé à l'emplacement {} a été nettoyé.".format(path_data_folder))
        else:
            print("Import Trained Dataset...")
            with open(path_json, 'r') as json_file:
                json_dataset = json.load(json_file)
    except Exception as e:
        print("Une erreur est survenue:", str(e))
    else:
        print("Création JSON file au :",path_data_folder)
        json_dataset["Type PDF"] = "Attestation de propriété"
        json_dataset["Data"] = {}

    print("Extraction des données... {} PDF trouvés.".format(len(paths_pdf)))

    for i, path_pdf in tqdm(enumerate(paths_pdf)):
        if (i < num_max_data or num_max_data == 0) :
            if (i in selected_datasets or len(selected_datasets) == 0):
                print("\nAnalysis of {}".format(os.path.basename(path_pdf)))
                sequence_classification = Classifier_Attestation_Propriete(path_pdf)
                #if sequence_classification.is_text():
                if print_first_info:
                    sequence_classification.display_first_info()
                json_data = sequence_classification.extract()
                json_dataset["Data"][os.path.basename(path_pdf)] = json_data
                sequence_classification.display_json_subdataset(display_sections)
        else:
            print("\nBreaking at iteration {}...".format(i))
            break

    with open(path_json, 'w') as json_file:
        json.dump(json_dataset, json_file, indent=4)
    return json_dataset

if __name__ == "__main__":
    
    # Extraction des données PDF localisées dans le dossier data
    path_data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    path_repo_train_dataset = os.path.join(os.path.dirname(__file__), '..', 'data')

    #selected_datasets = range(100, 203) # API TANGAMA = 28 
    #selected_datasets = range(70, 182)
    selected_datasets = [0] #range(182)

    display_sections = [1] #range(13) # Jusqu'à 11 ici
    num_max_data = 0
    print_first_info = False

    json_dataset = extract_pdf(path_data_folder, path_repo_train_dataset, clean_dataset=True, selected_datasets=selected_datasets, display_sections=display_sections, num_max_data=num_max_data, print_first_info=print_first_info)
    # pprint(json_dataset)

    # TODO : Reception_Acte_7
    # TODO : 96 Acte avec plusieurs désignations
    # TODO : Annee_1 pour 1.pdf

    # TODO : A CHECKER
    # TODO : Lieu_Reception_7
    # TODO : Lieu Siege Office Notarial

    # Test Unitaire dataset
    #path_pdf = os.path.join(os.path.dirname(__file__), "API TANGAMA.pdf")
    #json_dataset = Sequence_Classifier_Attestation_Propriete(path_pdf)
    #json_dataset.extract()
    #json_dataset.display_json_dataset()
    #json_dataset.display_json_subdataset(display_sections)
    
    # TODO : Remplacer "lines" par "start_index" de l'expression dans une liste pour matches
    # Concaténer les tokens_labels dans Training Model pour créer du contexte lors de l'entrainement
    # On entraine le modèle, page par page
    # Détecter les expressions a considérer comme inutiles : Tout - Moins ce qu'on a détecter
    
    # TODO : Remplacer les match.group(1).strip().split(" ") par une variable local pour éviter les repetitions dans les sets
  






