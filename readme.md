# Allineamento Trascrizioni - MiM Algorithm
Il metodo permette di allineare la trascrizione di una righa di testo alle relative parole nell'immagine della riga

## Preparazione
1. Crea una cartella ```"data/lines"``` che contiene le immagini delle linee di testo. La cartella è organizzata in sottocartelle una per ogni documento. 

2. Crea la cartelle ```"data/GT"``` che contiene i file txt delle trascrizioni. LA cartella è organizzata in sottocartelle una per ogni documento.

## Eseguire l'allineamento

1. Lancia il file ```alignment.py``` per effettuare l'allineamento e ottenere il file pickle ```"all_align.als"```. All'interno del file si possono settare  parametri per il processo.

2. Puoi correggere le uscite dell'algoritmo di allineamento lanciando il file ```"correction_tool.py"```
   il tool visualizzerà tutte le parole allineate una alla volta. Con il tasto INVIO si può passare alla prossima parola.
   Per correggere un errore di segmentazione si può utilizzare il mouse:
      con un click con il tasto sinistro si imposta un nuovo confine di segmentazione sinistro
      con un click con il tasto destro si imposta un nuovo confine di segmentazione destro
    
   alla fine il tool corregge il file di allineamento ```"all_aligns.als"```
   e genrea nella cartella ```time``` un file dove viene riportato il tempo totale impiegato alla correzione

   Il processo misura anche le performance dell'allineamento:
   verrà salvato un file nella cartella ```Performance```
   dove sono riportati il numero totale di allineamenti e il numero di allineamenti che non hanno necessitato di una correzione


3. lancia il file ```crop_all_words.py``` per generare tutte le immagini delle parole ottenute