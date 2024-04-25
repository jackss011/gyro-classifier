Per prima cosa devi settare la repo 
```bash
# scarica la repo
git clone <link>

# setta la repository remota
git remote add origin <link>
```

Ora metti i tuoi file dentro src poi committa e pusha

```
git add .
git commit -m "messaggio significativo"
git push origin main
```

Da adesso in poi lavora su un branch cosí non rischiamo di fare danni al main

per creare un nuovo branch
```
# la prima volta devi aggiungere -b per crearlo
git checkout -b <nome_branch>

# le prossime volte per cambiare branch basta che fai
git checkout <nome_branch>
```

ora fai le tue cose sul branch dopo di che committi quindi di nuovo

```
git add .
git commit -m "messaggio significativo"
#
git push origin <nome_branch>
```

quando sei arrivato a un punto in cui sei contento del codice e lo vuoi mettere sul main devi fare i seguenti passaggi

```
# vai nel main
git checkout main

# controlla se c'é qualcosa nel main remoto
git fetch origin main

# prendi quello che io ho pushato nel main remoto
git pull origin main

# torna al tuo branch
git checkout <nome_branch>

#ora devi sostanzialmente riallineare il tuo branch con il main, per farlo fai
git rebase main
# supponiamo che i commit sul main siano a-b-c e sul tuo branch sono invece a-b-d, 
# con git rebase il tuo branch diventa a-b-c-d

# ora devi rimettere le tue modifiche sul main quindi
git checkout main
git merge <nome_branch>
# ora anche il main ha i commit a-b-c-d

# ora basta pushare 
git push origin main
```

A questo punto torni sul tuo branch, aggiungi codice, e ripeti

