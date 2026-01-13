## Régression logistique ARCOM 2024 – prédire une forte confiance dans l’information

### Objectif du modèle

L’idée du modèle est simple : on veut prédire si une personne déclare une **forte confiance dans l’information**, à partir de ses réponses au questionnaire ARCOM 2024.

La variable de départ est `CONF1_R`, notée sur une échelle de 1 à 5. Pour passer dans un cadre de classification, on transforme cette variable en une cible binaire :

* **high_conf = 1** si `CONF1_R ≥ 4` (confiance élevée),
* **high_conf = 0** sinon.

On se retrouve donc dans un **problème de classification binaire**, ce qui correspond directement au cadre de la **régression logistique**.

---

### Données utilisées et déséquilibre des classes

Le jeu de données initial contient 3346 individus. Après suppression des observations où `CONF1_R` est manquant, il reste **2949 lignes**. L’échantillon est ensuite séparé en :

* **train : 2211**
* **test : 738**

Un point important est la **répartition des classes**. Après nettoyage, on compte :

* `high_conf = 0` : 2133 individus
* `high_conf = 1` : 816 individus

Donc la classe “forte confiance” représente environ **28%** des individus. Les classes sont donc **déséquilibrées** : la classe 1 est minoritaire.

C’est important parce que, dans ce contexte, l’**accuracy** peut être trompeuse : si on prédisait toujours “0”, on ferait déjà environ **72%** de “bonnes réponses” sans apprendre quoi que ce soit. Le fait d’utiliser `class_weight="balanced"` est donc une bonne pratique, car cela évite que le modèle ignore la classe minoritaire.

---

### Résultats globaux : AUC et accuracy

Les scores obtenus sont :

* **AUC = 0.661**
* **Accuracy = 0.652**

L’AUC (aire sous la courbe ROC) se lit comme en cours :

* 0.5 correspond au hasard,
* 1 correspond à une séparation parfaite.

Ici, **0.661** signifie que le modèle fait mieux que le hasard, mais que la séparation entre “forte confiance” et “faible confiance” reste **moyenne**. En pratique, ça veut dire que le modèle arrive assez souvent à donner une probabilité plus élevée à une personne réellement confiante qu’à une personne non confiante, mais les deux profils restent très proches dans beaucoup de cas.

L’accuracy (65%) semble correcte au premier regard, mais elle doit être interprétée avec prudence à cause du déséquilibre des classes. Elle ne dit pas clairement si le modèle repère bien les individus réellement confiants. Pour ça, la matrice de confusion et les métriques precision/recall sont beaucoup plus informatives.

---

### Lecture de la matrice de confusion

La matrice obtenue est :

[
\begin{pmatrix}
359 & 175 \
82 & 122
\end{pmatrix}
]

Ce qui se lit ainsi :

* **359 vrais négatifs** : le modèle prédit “faible confiance” et c’est correct,
* **175 faux positifs** : le modèle prédit “forte confiance” alors que la personne ne l’est pas,
* **82 faux négatifs** : le modèle prédit “faible confiance” alors que la personne est confiante,
* **122 vrais positifs** : le modèle détecte correctement une forte confiance.

Il y a **beaucoup de faux positifs**, ce qui explique une précision faible.

---

### Precision, recall et F1 

Pour la classe 1 (forte confiance), les métriques sont :

* **Recall ≈ 0.60**
  → le modèle détecte environ **60%** des individus réellement confiants.
  Il en “rate” donc environ 40%.

* **Precision ≈ 0.41**
  → quand le modèle dit “forte confiance”, il n’a raison que dans **41%** des cas.
  Autrement dit, il déclenche beaucoup de “fausses alertes” (cohérent avec FP = 175).

* **F1 ≈ 0.49**
  → c’est un résumé du compromis : le modèle repère une partie de la classe 1, mais il est peu fiable lorsqu’il la prédit.

---

### Pourquoi le modèle n’est pas “tranché” : probabilités prédites

Le graphique des probabilités prédites montre que les distributions des probabilités pour les vrais 0 et les vrais 1 **se chevauchent fortement**. On n’observe pas une séparation nette où les “1” auraient systématiquement des probabilités proches de 1 et les “0” proches de 0.

En clair :

* certaines personnes non confiantes reçoivent quand même des probabilités assez élevées,
* et certaines personnes confiantes reçoivent seulement des probabilités moyennes.

C’est exactement ce qui explique à la fois :

* un **AUC moyen** (0.66),
* et une **precision faible** sur la classe 1.

---

### Le seuil à 0.5 : pas forcément optimal

Par défaut, tu classes “1” si la probabilité prédite est ≥ 0.5. Mais vu le chevauchement des distributions, ce seuil n’est pas forcément le meilleur.

* Si on **augmente** le seuil, on réduira les faux positifs → **precision augmente**, mais on manquera davantage de vrais confiants → **recall baisse**.
* Si on **baisse** le seuil, on détectera plus de confiants → **recall augmente**, mais au prix de plus de fausses alertes → **precision baisse**.

C’est le compromis classique vu en cours, et c’est exactement ce que résume la courbe ROC.

---

### Lecture des coefficients : interprétation prudente

En régression logistique, les coefficients s’interprètent comme des effets sur les **log-odds**, donc sur la probabilité d’appartenir à la classe 1.

* coefficient **positif** : associé à une probabilité plus élevée d’être “confiant”
* coefficient **négatif** : associé à une probabilité plus faible

Le graphique “top coefficients” met en avant certaines variables très marquées. Mais comme on l’a vu en cours, il faut rester prudent : ce sont des **associations statistiques**, pas des relations causales. On ne peut pas dire “ça cause la confiance”, seulement “c’est lié à la confiance dans les données”.

---

## Conclusion 

Au final, la régression logistique apprend bien quelque chose : l’AUC de 0.661 montre qu’elle fait mieux que le hasard, donc les réponses au questionnaire contiennent de l’information utile. Le modèle parvient à détecter environ 60% des personnes réellement confiantes, mais sa **precision est faible** (41%) : beaucoup de personnes prédites “confiantes” ne le sont pas réellement.

En pratique, ce modèle est donc pertinent pour une lecture **exploratoire** : il met en évidence des liens statistiques entre pratiques/réponses et confiance dans l’information, mais il reste limité pour une prédiction fine individu par individu, ce qui reflète la complexité du phénomène et l’existence probable de nombreux facteurs non observés.

---
