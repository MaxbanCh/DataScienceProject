
## Analyse du modèle **JT_OLS** – durée des JT (*duree_sec*)

On commence par analyser un modèle de régression linéaire classique (OLS) dont l’objectif est d’expliquer la durée des sujets de JT, mesurée en secondes.

### Qualité globale de l’ajustement

Les indicateurs de performance montrent que le modèle explique une part importante de la variabilité observée. Le **R² est d’environ 0.79**, et le R² ajusté est quasiment identique, ce qui indique que le nombre de variables explicatives (21 au total) est cohérent et qu’il n’y a pas de sur-ajustement manifeste. Autrement dit, les variables ajoutées apportent bien de l’information utile.

Le **RMSE sur l’échantillon test est d’environ 93,5 secondes**, soit un peu plus d’une minute et demie d’erreur moyenne. Compte tenu de l’ordre de grandeur des durées de sujets de JT, cette erreur reste raisonnable. Avec plus de **214 000 observations**, on dispose en outre d’un échantillon très large, ce qui rend les résultats particulièrement stables.

Dans l’ensemble, le modèle OLS fournit donc un **bon ajustement**, sans être parfait, ce qui est attendu pour un phénomène éditorial complexe.

---

### Significativité globale du modèle

Le test de Fisher confirme cette bonne performance globale. La statistique F est très élevée (de l’ordre de 3,8 × 10⁴) et la p-value est pratiquement nulle. On rejette donc très nettement l’hypothèse selon laquelle tous les coefficients (hors constante) seraient nuls.

Cela signifie que, pris collectivement, les facteurs explicatifs inclus dans le modèle contribuent bien à expliquer la durée des JT.

---

### Lecture et interprétation des coefficients

La constante est estimée à environ **154 secondes**. Elle correspond à une durée de référence lorsque toutes les variables explicatives prennent la valeur zéro, ce qui sert surtout de point d’ancrage au modèle.

Plusieurs variables ont des **effets positifs marqués et très significatifs**. Certaines augmentent la durée du JT de plusieurs dizaines de secondes, et même de plus de deux minutes pour l’une d’entre elles. Ces résultats indiquent que certaines caractéristiques du contexte ou du contenu sont fortement associées à un allongement des sujets.

À l’inverse, quelques variables présentent des **effets négatifs significatifs**, réduisant la durée du JT d’une quinzaine à une vingtaine de secondes. Elles correspondent donc à des facteurs associés à des sujets plus courts.

Une variable apparaît en revanche **non significative au seuil de 5 %**, avec une p-value proche de 0,06. Dans la logique du cours, ce type de variable pourrait être envisagé comme supprimable, car son apport explicatif est incertain.

Un point important à souligner est que, contrairement à certains modèles précédents, les coefficients de ce modèle sont **globalement stables, cohérents et interprétables**, ce qui constitue un avantage majeur.

---

### Diagnostics graphiques et hypothèses OLS

Les graphiques de diagnostic confirment que le modèle capture correctement la structure moyenne des données. Le nuage des valeurs observées contre les valeurs prédites est bien aligné le long de la diagonale, avec une dispersion raisonnable, ce qui montre que la relation linéaire estimée est pertinente.

En revanche, l’analyse des résidus révèle plusieurs écarts aux hypothèses idéales du modèle linéaire. La distribution des résidus est très asymétrique, comme le montrent l’histogramme, le Q-Q plot et les indicateurs de forme (skewness très élevée, kurtosis extrême, test de Jarque–Bera fortement significatif). L’hypothèse de normalité des erreurs est donc clairement violée.

Le graphique des résidus en fonction des valeurs ajustées met également en évidence une **hétéroscédasticité marquée** : la variance des résidus augmente avec la durée prédite. Cela signifie que le modèle commet des erreurs plus importantes lorsque les sujets sont plus longs.

Ces problèmes ne sont toutefois ni surprenants ni rédhibitoires dans un contexte de données réelles de grande taille. Avec plus de 200 000 observations, les tests de normalité deviennent de toute façon extrêmement sensibles, et l’OLS reste un estimateur valide pour les coefficients, même si les écarts-types doivent être interprétés avec prudence.

---

### Multicolinéarité

Le condition number très élevé (de l’ordre de 10¹⁶) et la valeur propre minimale quasi nulle signalent une **multicolinéarité importante** entre certaines variables explicatives. En théorie, cela peut rendre les coefficients instables.

Dans ce modèle toutefois, les statistiques t restent élevées et les signes des coefficients sont cohérents, ce qui suggère que la multicolinéarité, bien que présente, est **moins problématique** que dans d’autres modèles plus instables. L’interprétation globale des effets reste donc possible.

---

### Bilan pour le modèle JT_OLS

En résumé, le modèle OLS appliqué à la durée des JT explique environ **79 % de la variance observée**, ce qui constitue une performance satisfaisante. Le modèle est globalement significatif et la majorité des coefficients sont interprétables. Néanmoins, les diagnostics révèlent des violations claires des hypothèses de normalité et d’homoscédasticité, ainsi qu’une multicolinéarité non négligeable. Ces limites sont attendues compte tenu de la nature des données et de leur taille.

---



## Analyse du modèle **JT_RIDGE** – Régression Ridge sur la durée des sujets de JT

On considère maintenant un modèle de **régression Ridge**, estimé sur les mêmes données et avec les mêmes variables explicatives, afin de comparer son comportement à celui de l’OLS.

### Cadre méthodologique

Comme vu en cours, la régression Ridge introduit une pénalisation de type L2 sur les coefficients. Cette pénalisation vise à réduire la multicolinéarité, à stabiliser les estimations et à améliorer la capacité de généralisation du modèle, au prix d’un léger biais.

---

### Performances globales

Les performances du modèle Ridge sont très proches de celles de l’OLS. Le **R² sur l’échantillon test est d’environ 0.78**, et le **RMSE est proche de 93 secondes**, soit une erreur moyenne comparable à celle obtenue précédemment. Avec un paramètre de pénalisation α = 2.0, le modèle conserve donc un excellent pouvoir explicatif.

Le résultat clé est que la pénalisation Ridge **ne dégrade pas la performance prédictive** par rapport à l’OLS, ce qui est un point très positif.

---

### Observé versus prédit et analyse des résidus

Le graphique observé/prédit montre une forte concentration des points autour de la diagonale, signe que le modèle capte bien la structure moyenne des durées. Comme attendu avec une pénalisation L2, les valeurs extrêmes sont légèrement écrasées : les très longues durées ont tendance à être sous-estimées.

L’analyse des résidus révèle une distribution centrée autour de zéro, mais asymétrique, avec des queues épaisses. Le Q-Q plot montre un bon alignement au centre de la distribution, mais des écarts importants dans les extrémités, ce qui traduit la présence de sujets atypiques ou d’événements exceptionnels.

Le graphique des résidus en fonction des valeurs ajustées confirme la présence d’hétéroscédasticité : plus la durée prédite est élevée, plus l’erreur potentielle augmente. La Ridge ne corrige pas ce problème, mais elle permet d’éviter les instabilités numériques et l’explosion des coefficients.

---

### Comparaison OLS / Ridge

D’un point de vue prédictif, les deux modèles sont quasiment équivalents. En revanche, la Ridge présente un avantage méthodologique clair : elle atténue les effets de la multicolinéarité et produit des coefficients plus stables. Même si l’interprétation fine des coefficients reste limitée, le modèle Ridge est plus robuste dans un contexte de nombreuses variables corrélées.

---

### Conclusion générale

En conclusion, le modèle **JT_RIDGE** explique efficacement la durée des sujets de JT, avec un R² proche de 0.78 et une erreur moyenne inférieure à deux minutes. La pénalisation Ridge permet de stabiliser l’estimation dans un contexte de forte multicolinéarité, sans perte de performance par rapport à l’OLS. Les diagnostics confirment toutefois la présence d’hétéroscédasticité et de résidus non parfaitement gaussiens, ce qui est attendu pour des données éditoriales réelles. Le modèle reste néanmoins pertinent pour analyser et prédire les durées moyennes des sujets de JT.



Que dit vraiment ce modèle sur les JT ?

D’un point de vue concret, ces modèles montrent que la durée des sujets de JT n’est pas improvisée. Environ 80 % de la variation de la durée peut être expliquée par des éléments de contexte relativement simples : la chaîne, le moment de diffusion, le calendrier, ou encore certaines caractéristiques récurrentes de la programmation.

-> Autrement dit, un JT suit une grammaire temporelle :

certaines chaînes ont des formats plus longs que d’autres, certaines heures ou certains jours appellent des sujets plus développés, les périodes particulières (week-ends, vacances, jours fériés) modifient la durée des sujets.Le fait d’obtenir un R² proche de 0.8 signifie que la majorité des choix de durée sont structurels, et non laissés au hasard ou uniquement dictés par l’actualité du jour.