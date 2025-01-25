ITA -------------------------------------------------------------------------------------

SISTEMA DI RICONOSCIMENTO MULTI-ALGORITMO BASATO SULL'IRIDE
Università degli Studi di Enna "Kore"
Dipartimento di Ingegneria e Architettura
Corso di Laurea in Ingeneria dell'Intelligenza artificiale e della Sicurezza Informatica 
allievi: Barbera Antonino, Mirotta Mauro
A.A. 2024/2025

 LINGUAGGIO : Python 3.11.2
 COME ESEGUIRE IL PROGETTO:
 - Installare le librerie necessarie :
    da terminale : pip install -r requirements.txt
    o in alternativa, da terminale GitBash o MAC:
    bash prepare.sh

    ATTENZIONE : Se dovessero riscontrarsi problemi con la libreria opencv, eseguire da terminale:
    -pip uninstall opencv-python
    -pip install -r requirements.txt

 - Eseguire il sistema multi-algoritmo al completo:
   da terminale : python main.py
   L'esecuzione potrebbe richiedere diversi minuti inizialmente per il caricamento delle immagini;
   Verranno create la cartella graph e iris_images con le relative immagini all'interno;
   Lo script non funzionerà se non è presente la cartella checkpoints con all'interno i modelli;
   Se la cartella checkpoints non è presente, eseguire da terminale:
      python models_generation.py
    
 - Eseguire l'algoritmo 1 basato su classificatori ML:
   da terminale :  python models_generation.py
   L'operazione potrebbe richiedere alcuni minuti per il caricamento delle immagini;
   Verrà create la cartella graph con all'interno i grafici;

 - Eseguire l'algoritmo 2 basato sul SIFT:
  da terminale :  python sift_test.py
  L'operazione richiede alcuni minuti per il carico computazionale;

Il Dataset utilizzato è il CASIA IRIS v1, fornito dal docente per la realizzazione di questo progetto.


ENG ---------------------------------------------------------------------------------------------------

MULTI-ALGORITHM IRIS RECOGNITION SYSTEM
University of Enna "Kore"
Department of Engineering and Architecture
Bachelor's Degree in Artificial Intelligence Engineering and Information Security
Students: Barbera Antonino, Mirotta Mauro
Academic Year: 2024/2025

Language: Python 3.11.2
Instructions to Run the Project:

- Install the required libraries by running:
  pip install -r requirements.txt
  Or alternatively, from GitBash or MAC terminal:
  bash prepare.sh
  ATTENTION : If you encounter issues with the opencv library, run the following commands:
  pip uninstall opencv-python
  pip install -r requirements.txt

- Running the Complete System
  To execute the multi-algorithm system, run:
  python main.py
  The first run might take a few minutes to load the images.
  The graph and iris_images folders with related images will be automatically generated.
  Important: The script will not work if the checkpoints folder with the required models is missing.
  If the checkpoints folder does not exist, generate it by running:
  python models_generation.py

- Running Algorithm 1: Based on ML Classifiers
  Run the following command:
  python models_generation.py
  This process may take a few minutes to load the images. The graph folder containing the graphs will be generated upon completion.

- Running Algorithm 2: Based on SIFT
  Run the following command:
  python sift_test.py
  This operation might take a few minutes due to computational processing.

Dataset Used
The project uses the CASIA IRIS v1 dataset, provided by the professor for the development of the system.
--------------------------------------------------------------------------------------------------------------------------