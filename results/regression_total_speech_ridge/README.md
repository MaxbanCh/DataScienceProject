## Régression Ridge sur la durée totale de parole (*total_speech*)

On s’intéresse ici à un modèle de **régression Ridge** dont l’objectif est d’expliquer la durée totale de parole, mesurée en secondes (`total_speech`), uniquement à partir du **contexte de diffusion**. La question sous-jacente est donc la suivante : *dans quelle mesure le volume de parole peut-il être expliqué par la programmation (chaîne, heure, calendrier, etc.), indépendamment du contenu précis ?*

### Cadre du modèle et données utilisées

La variable expliquée est `total_speech`, qui correspond au volume total de parole sur un créneau donné. Les variables explicatives décrivent exclusivement le contexte : la chaîne, le type de média (radio ou télévision), l’heure de diffusion, le jour de la semaine, l’année et le mois, ainsi que plusieurs indicateurs de calendrier (jours fériés, vacances scolaires et zones).

Le modèle estimé est une **régression linéaire pénalisée de type Ridge**, avec un paramètre de régularisation **α = 5.0**. Comme vu en cours, ce paramètre permet d’éviter des coefficients excessifs en présence de nombreuses variables corrélées, au prix d’un léger biais.

L’échantillon est particulièrement large : environ **863 000 observations** sont utilisées pour l’entraînement et **216 000** pour le test, soit une répartition classique de 80 % / 20 %. Cette taille garantit une très grande stabilité des résultats.

---

### Performance globale du modèle

Sur l’échantillon test, le modèle atteint un **R² de 0.772**, ce qui signifie qu’il explique environ **77 % de la variabilité** de la durée totale de parole. Ce niveau de performance est élevé et montre que le volume de parole n’est pas aléatoire : il suit des règles et des habitudes de diffusion très liées à la programmation, comme la chaîne, l’heure ou le calendrier.

Le **RMSE est d’environ 536 secondes**, soit un peu moins de **9 minutes d’erreur moyenne**. Cette valeur peut sembler élevée en absolu, mais elle doit être mise en perspective avec l’ordre de grandeur de `total_speech`, qui peut atteindre des durées importantes. Il est donc normal d’observer des écarts de plusieurs minutes, même lorsque le R² est élevé.

Dans l’ensemble, ces résultats montrent que le modèle explique très bien la structure générale du volume de parole, même si la précision individuelle reste limitée.

---

### Observé versus prédit

Le graphique comparant les valeurs observées et les valeurs prédites montre une **bonne cohérence globale** avec la diagonale : le modèle prédit correctement les ordres de grandeur de `total_speech`. On observe cependant une dispersion importante autour de la diagonale, ainsi que l’existence de **deux grands nuages de points**, correspondant vraisemblablement à des niveaux de volume de parole différents.

Cela suggère que le modèle distingue bien certains “régimes” de diffusion, par exemple des créneaux ou des chaînes où l’on parle beaucoup, et d’autres où l’on parle moins. En revanche, à l’intérieur de ces régimes, une partie de la variabilité reste inexpliquée.

---

### Analyse des résidus

Le graphique des résidus en fonction des valeurs ajustées montre que les erreurs sont bien centrées autour de zéro, ce qui indique l’absence de biais moyen. En revanche, on observe des **structures en éventail** et des amas de points, ce qui révèle que le modèle n’explique pas entièrement la variabilité conditionnelle de `total_speech`.

L’histogramme des résidus confirme cette lecture. La distribution est centrée autour de zéro, mais elle présente des **queues longues**, avec des erreurs très importantes dans certains cas. Autrement dit, la plupart du temps, le modèle se trompe de manière modérée, mais il existe des situations rares où l’erreur est très élevée.

Le Q-Q plot va dans le même sens : les résidus suivent assez bien une loi normale au centre de la distribution, mais s’en écartent fortement dans les extrémités. Cela indique une **non-normalité des résidus**, principalement due à la présence de valeurs extrêmes.

Ces résultats sont cohérents avec la nature de la variable étudiée. Le volume de parole peut être fortement affecté par des événements exceptionnels, des émissions spéciales ou des situations éditoriales particulières, qui ne sont pas entièrement capturées par les variables de contexte disponibles.

---

### Rôle de la pénalisation Ridge

Le choix d’un paramètre **α = 5.0** permet de rendre le modèle plus stable et moins sensible aux corrélations entre variables explicatives. Une pénalisation plus forte conduit à un modèle plus “sage”, avec des coefficients plus modérés, tandis qu’une pénalisation plus faible laisserait davantage de liberté au modèle, au risque d’instabilités. Ici, le compromis choisi permet d’obtenir de bonnes performances sans comportements extrêmes.

---

### Conclusion générale

En conclusion, la régression Ridge appliquée à `total_speech` montre que **le contexte de diffusion explique une grande partie du volume total de parole**. Avec un R² d’environ 0.77, le modèle capture efficacement la structure générale des durées, confirmant que la programmation joue un rôle central. L’erreur moyenne reste toutefois de l’ordre de plusieurs minutes, et les diagnostics révèlent une dispersion importante et des résidus non parfaitement gaussiens, en particulier en présence de situations exceptionnelles.

Le modèle constitue donc un **outil descriptif et prédictif solide** pour comprendre les grandes régularités du volume de parole, tout en laissant apparaître des marges d’amélioration liées à des facteurs non observés, comme le contenu précis des émissions ou des événements éditoriaux particuliers.




Au final, ce modèle montre quelque chose de très clair : le volume total de parole est largement déterminé par la programmation. Le contexte de diffusion impose un cadre fort, dans lequel les émissions s’inscrivent presque toujours.

Cependant, il existe des écarts parfois importants, qui correspondent précisément aux moments où l’actualité ou les choix éditoriaux prennent le dessus sur la routine. Ces écarts ne sont pas des anomalies statistiques : ce sont les moments où le média sort de son format habituel.

-> En résumé :
la programmation fixe le cadre du volume de parole, mais le contenu décide quand et comment ce cadre est dépassé.