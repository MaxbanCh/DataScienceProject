## Régression Ridge sur *female_share* — Analyse et interprétation

On s’intéresse ici à une régression Ridge appliquée à la variable *female_share*, qui mesure la proportion de parole féminine (entre 0 et 1). L’analyse est réalisée uniquement sur les observations dont le temps de parole total dépasse 60 secondes, ce qui correspond au filtre recommandé en cours afin d’éviter les observations trop courtes, souvent instables et bruitées.

Le modèle utilisé est une régression linéaire pénalisée de type Ridge, avec un paramètre de pénalisation α fixé à 3. Ce choix vise principalement à réduire la variance des estimations et à limiter les problèmes de multicolinéarité entre les variables explicatives. L’échantillon est particulièrement large, avec environ 855 000 observations pour l’entraînement et plus de 213 000 pour le test, ce qui garantit une grande stabilité des résultats.

Il est important de rappeler, comme vu en cours, que la régression Ridge améliore la robustesse des coefficients mais ne permet pas de réaliser une inférence statistique classique : il n’y a donc pas de p-values associées aux coefficients estimés.

---

### Performance globale du modèle

Le coefficient de détermination obtenu sur l’échantillon test est de **R² = 0,178**. Cela signifie que le modèle explique environ 18 % de la variance de la part de parole féminine. Ce niveau de R² peut paraître faible, mais il n’est pas surprenant dans ce contexte. La variable expliquée est une proportion bornée entre 0 et 1, souvent difficile à prédire, et elle dépend très probablement de nombreux facteurs non observés, comme le contenu éditorial, le type d’émission, les invités ou le contexte du débat. Comme souligné en cours, un R² faible n’implique pas nécessairement que le modèle est inutile, surtout lorsque le phénomène étudié est complexe.

Le RMSE sur l’échantillon test est d’environ **0,173**. Concrètement, cela correspond à une erreur moyenne de prédiction de 17 points de pourcentage sur la part de parole féminine. Le modèle peut donc prédire, par exemple, une valeur de 0,45 alors que la valeur réelle est de 0,62, ou encore 0,30 au lieu de 0,47. Ce niveau d’erreur est acceptable pour une analyse exploratoire ou descriptive, mais il reste insuffisant pour des prédictions précises au niveau individuel.

---

### Analyse des diagnostics graphiques

Le graphique comparant les valeurs observées et les valeurs prédites montre une forte dispersion des points autour de la diagonale. Il n’y a pas d’alignement net sur la droite parfaite, ce qui indique que le modèle capte une tendance globale mais reste peu précis individuellement. On observe notamment une sous-estimation des valeurs élevées de *female_share* et, dans certains cas, une surestimation des valeurs faibles. Ce comportement est cohérent avec le R² relativement modeste : le signal existe, mais il est faible.

L’analyse des résidus en fonction des valeurs ajustées met en évidence une structure en éventail, avec une variance des résidus qui augmente ou diminue selon le niveau de la prédiction. Cela suggère une violation de l’hypothèse d’homoscédasticité. Autrement dit, le modèle linéaire ne parvient pas à expliquer correctement la variance conditionnelle de *female_share*. Cette observation indique que la relation entre les variables explicatives et la part de parole féminine n’est probablement pas strictement linéaire et qu’il pourrait manquer des interactions, des effets non linéaires ou des variables explicatives importantes.

L’histogramme des résidus montre une distribution centrée autour de zéro, ce qui suggère l’absence de biais systématique dans les prédictions. En revanche, la distribution n’est pas parfaitement symétrique et présente des queues relativement épaisses. Le Q-Q plot confirme ces écarts par rapport à la normalité théorique, en particulier dans les queues de distribution. Ce résultat est fréquent lorsque l’on modélise une variable bornée entre 0 et 1 avec un modèle linéaire.

Il convient toutefois de rappeler que, dans un cadre de régression Ridge à visée principalement prédictive ou descriptive, la normalité des résidus est moins cruciale que dans un cadre d’inférence classique.

---

### Bilan général

Dans l’ensemble, ce modèle présente plusieurs points positifs. La taille très importante de l’échantillon rend les résultats robustes, et la pénalisation Ridge assure une bonne stabilité des coefficients. Le modèle détecte un signal réel entre les variables explicatives et la part de parole féminine, et le filtrage sur le temps de parole total est méthodologiquement pertinent.

Cependant, ses limites sont claires. Le pouvoir explicatif reste modéré, les résidus sont fortement dispersés et plusieurs hypothèses du modèle linéaire — notamment la linéarité, l’homoscédasticité et la normalité des erreurs — sont imparfaitement respectées. Enfin, l’absence de p-values empêche toute interprétation causale ou inférentielle des coefficients.

---

### Conclusion

En conclusion, la régression Ridge appliquée à la variable *female_share* met en évidence une relation statistique réelle mais limitée entre les variables explicatives et la part de parole féminine. Le R² relativement faible reflète la complexité du phénomène étudié et l’importance probable de facteurs non observés. Les diagnostics graphiques confirment que le cadre linéaire est imparfait pour modéliser ce type de variable bornée.

Ce modèle doit donc être compris avant tout comme un **outil descriptif et exploratoire**, utile pour dégager des tendances générales, mais inadapté à une prédiction fine ou à une interprétation causale au niveau individuel.

