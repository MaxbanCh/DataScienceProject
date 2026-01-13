
## Modèle M1 (OLS) – Prédiction de la durée de parole féminine (*female_duration*)

On commence par analyser un premier modèle, noté **M1**, qui repose sur une régression linéaire classique afin de prédire la durée de parole féminine, mesurée en secondes.

### Cadre général du modèle

Le modèle utilise un grand nombre de variables explicatives, essentiellement des variables de contexte (chaîne, temporalité, etc.) ainsi que des variables de volume, parmi lesquelles figure notamment `male_duration`. Après encodage des variables catégorielles, le modèle mobilise **72 variables explicatives**, comme l’indique le *Df Model = 72* dans le résumé OLS.

La variable expliquée est `female_duration`, c’est-à-dire la durée totale de parole des femmes sur une séquence donnée. La relation estimée est celle d’une régression linéaire standard, du type
[
\hat{y} = a + b_1 x_1 + b_2 x_2 + \dots
]

Les tailles d’échantillon sont très importantes : environ **863 000 observations pour l’entraînement** et **216 000 pour le test**. Une telle taille garantit que les résultats ne sont pas dus au hasard et qu’ils sont statistiquement très stables.

---

### Qualité de prédiction du modèle

La performance prédictive du modèle M1 est exceptionnelle. Le **R² sur l’échantillon test est de0.9776**, ce qui signifie que le modèle explique près de **98 % de la variance** de la durée de parole féminine. Autrement dit, lorsque `female_duration` varie, le modèle parvient presque systématiquement à suivre ces variations.

Le **RMSE test est d’environ 80 secondes**, soit une erreur moyenne d’environ une minute vingt. Étant donné que certaines durées de parole peuvent atteindre plusieurs dizaines de minutes, cette erreur reste relativement faible. D’un point de vue strictement prédictif, le modèle est donc extrêmement performant.

Le résumé OLS fourni par *statsmodels* confirme ce constat : le R² reporté est très proche (0.976), ce qui est cohérent puisque ce dernier est généralement calculé sur l’échantillon d’estimation. Le test F global est massivement significatif (p-value ≈ 0), ce qui indique que le modèle, pris dans son ensemble, est bien meilleur qu’un modèle nul ne contenant qu’une constante. Il faut toutefois rappeler qu’avec un échantillon de plus de 800 000 observations, presque tout devient significatif au sens statistique.

---

### Interprétation des coefficients et limites structurelles

Même si de nombreux coefficients apparaissent très significatifs individuellement, l’interprétation économique de ces coefficients pose un problème majeur. Le résumé OLS signale un **Condition Number extrêmement élevé (≈ 10¹⁶)** ainsi qu’une valeur propre minimale quasi nulle. Cela révèle une **multicolinéarité très forte**, voire une quasi-singularité de la matrice des variables explicatives.

Concrètement, cela signifie que certaines variables sont très fortement corrélées entre elles, ou qu’elles contiennent des informations redondantes (par exemple des encodages catégoriels mal contraints ou des relations quasi comptables entre variables). Cette situation entraîne des coefficients parfois énormes et instables, avec des p-values difficiles à interpréter.

Ainsi, même si le modèle prédit très bien, **il ne faut pas interpréter les coefficients individuellement** dans M1. Le modèle est performant du point de vue prédictif, mais fragile et peu fiable du point de vue interprétatif.

---

### Diagnostics graphiques

Les graphiques confirment cette lecture. Le nuage observé versus prédit montre une forte concentration des points autour de la diagonale, ce qui traduit une excellente qualité de prédiction globale. On observe toutefois quelques points très éloignés, correspondant à des cas atypiques ou extrêmes.

Le graphique des résidus en fonction des valeurs ajustées révèle une structure en éventail, signe clair d’**hétéroscédasticité** : l’erreur n’a pas la même variance selon le niveau de la prédiction. Le modèle se trompe davantage pour certaines plages de valeurs, notamment pour les durées très longues.

L’histogramme des résidus est centré autour de zéro, ce qui indique l’absence de biais moyen, mais il présente une asymétrie et des queues épaisses. Le Q-Q plot confirme que la normalité des résidus n’est pas respectée, en particulier dans les extrémités de la distribution. Ces violations sont fréquentes sur des données réelles et ne remettent pas en cause la qualité prédictive du modèle, mais elles limitent encore davantage l’interprétation statistique fine.

---

### Bilan pour le modèle M1

En résumé, le modèle M1 est **un excellent modèle de prédiction**, avec un R² proche de 0.98 et une erreur moyenne relativement faible. En revanche, il souffre de problèmes structurels importants : multicolinéarité extrême, hétéroscédasticité et non-normalité des résidus. Il est donc inadapté à une interprétation causale ou à une lecture coefficient par coefficient.

---




## Modèle M2 – Régression Ridge sans `male_duration`

Le modèle **M2** adopte une approche différente. Il s’agit d’une régression Ridge (pénalisation L2), dont l’objectif est de prédire `female_duration` **sans inclure `male_duration`**, variable extrêmement corrélée à la cible. Ce choix s’inscrit dans une logique plus structurelle, visant à expliquer la durée de parole féminine à partir d’éléments de programmation et de contexte, plutôt que par une relation quasi comptable.

La pénalisation Ridge (avec α = 1.0) est utilisée ici pour stabiliser les coefficients en présence de multicolinéarité et pour éviter l’instabilité observée dans l’OLS.

---

### Performance globale de M2

Les performances de M2 sont nettement plus modestes : le **R² test est d’environ 0.44**, ce qui signifie que le modèle explique un peu moins de la moitié de la variance de `female_duration`. Le **RMSE est d’environ 401 secondes**, soit plus de six minutes d’erreur moyenne.

Cette baisse de performance est attendue et logique. En retirant `male_duration`, on supprime la variable la plus informative du point de vue purement prédictif. Le modèle ne capte donc plus la variabilité fine de la durée de parole, mais seulement une structure moyenne.

---

### Diagnostics graphiques et comportement du modèle

Le graphique observé versus prédit montre un nuage très éloigné de la diagonale, avec une forte compression des prédictions. Le modèle plafonne autour de certaines valeurs et sous-estime systématiquement les durées longues. Ce comportement est typique de la Ridge, qui réduit volontairement l’amplitude des prédictions afin de diminuer la variance.

Les résidus présentent une forte asymétrie, avec une longue queue négative, ce qui traduit une sous-prédiction des durées élevées. Le Q-Q plot montre des écarts importants à la normalité, en particulier dans les queues. Enfin, le graphique résidus versus valeurs ajustées révèle une hétéroscédasticité très marquée, indiquant que le modèle est biaisé selon le niveau de `female_duration`.

Ces diagnostics montrent que, malgré la pénalisation, un modèle linéaire reste insuffisant pour capturer toute la complexité de la distribution de la durée de parole féminine.

---

### Comparaison entre M1 et M2

La comparaison entre les deux modèles est très instructive. M1 affiche un R² proche de 1, mais repose sur une relation quasi mécanique et n’est pas interprétable. M2, au contraire, présente un R² beaucoup plus faible, mais il est méthodologiquement plus honnête et plus proche d’un modèle explicatif ou structurel.

La baisse de performance observée dans M2 n’est donc pas un échec, mais la conséquence logique du retrait d’une variable centrale et du choix d’un modèle pénalisé.

---

### Conclusion générale

En conclusion, la variable male_duration est essentielle pour comprendre les résultats. Sa présence dans M1 permet une prédiction quasi parfaite, mais au prix d’une interprétation très limitée : le modèle décrit surtout une contrainte comptable du temps de parole.

En retirant male_duration, le modèle M2 met en évidence une réalité plus intéressante : la durée de parole féminine ne dépend pas uniquement du contexte de diffusion, mais aussi de facteurs éditoriaux et situationnels non observés.

Ainsi, si l’objectif est de prédire, M1 est imbattable. En revanche, si l’objectif est de comprendre les mécanismes réels de la visibilité des femmes dans les médias, M2 constitue un point de départ plus pertinent, même si ses performances sont plus modestes.

-> En termes concrets :
la parole féminine n’est pas seulement une question d’horaires ou de chaînes, mais de choix éditoriaux, qui dépassent largement ce que capturent les variables disponibles.