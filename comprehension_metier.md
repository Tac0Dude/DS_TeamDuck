# 0. Phase Zéro – Business Understanding

## Value Proposition

### Quel est le problème ?

L'administration fiscale de l'État de l'Iowa fait face à une problématique structurelle : des propriétaires réalisent des travaux non déclarés (extensions, rénovations, aménagements de sous-sol ou de garage) pour dissimuler la plus-value réelle de leur bien et ainsi frauder la taxe foncière. Lors de la vente, le prix de transaction reflète la valeur réelle du bien — y compris les améliorations clandestines — tandis que les données cadastrales officielles, elles, n'ont jamais été mises à jour.

L'objectif est de construire un modèle de régression capable d'estimer la **valeur marchande théorique** d'un bien au moment de sa transaction, à partir de ses caractéristiques techniques déclarées. En comparant cette estimation avec le prix de vente effectif, le système détecte les **plus-values injustifiées** — même sur de petits montants — qui signalent une potentielle fraude fiscale.

### Pourquoi est-ce important ?

Une estimation erronée produit deux types de conséquences critiques pour l'administration :

- Une **sous-estimation** du prix théorique laisserait passer des dossiers frauduleux sans déclenchement d'alerte, permettant à des contribuables indélicats d'échapper à leur juste imposition.
- Une **surestimation** du prix théorique générerait de **faux positifs** : des citoyens innocents seraient accusés à tort, engendrant des frais administratifs lourds, des contentieux coûteux et une perte de confiance envers l'administration fiscale.

Un modèle prédictif performant permet donc de :
- **Concentrer les audits** sur les dossiers à fort risque de fraude, en optimisant l'allocation des ressources des agents fiscaux.
- **Réduire les recours abusifs** en fondant les transmissions au service des contentieux sur une preuve chiffrée et objectivable.
- **Garantir l'équité fiscale** entre contribuables, en détectant les écarts de valeur de façon systématique et reproductible, indépendamment des biais humains d'un évaluateur individuel.

### Qui est l'utilisateur final ?

Le système est conçu pour être utilisé par des organes de l'État en charge des audits fiscaux, notamment lors des procédures de vente immobilière. L'utilisateur-type est un agent de contrôle fiscal non-spécialiste en machine learning, mais doté de compétences dans l'analyse des dossiers cadastraux et la détection de fraude. L'outil doit lui fournir :

1. Une **estimation automatique** de la valeur théorique d'un bien à partir de ses caractéristiques cadastrales déclarées.
2. Un **score d'écart** entre valeur estimée et prix de vente réel, signalant les dossiers suspects.
3. Une **justification variable par variable** (via SHAP) permettant de motiver une transmission au service des contentieux de façon défendable.

---

## Data Sources

### Quelles sont les sources de données disponibles ?

Les données mobilisées proviennent des ventes de biens immobiliers résidentiels collectées par l'État de l'Iowa. Le dataset utilisé est l'**Ames Housing Dataset**, constitué de **1 460 transactions** observées entre 2006 et 2010, comportant **80 variables** par bien.

#### Sources administratives et cadastrales
- Registres de ventes immobilières officiels de la ville d'Ames, Iowa.
- Données cadastrales : surface habitable, surface du sous-sol, type de zonage, année de construction, année de rénovation, présence et superficie du garage, de la piscine, de la véranda.
- Informations de localisation : quartier (`Neighborhood`), configuration du terrain, proximité d'axes routiers ou de voies ferrées.

#### Variables disponibles à la prédiction
Lors d'une transaction, les données suivantes sont accessibles avant connaissance du prix de vente :
- **Surface** : surface habitable (`GrLivArea`), surface du sous-sol (`TotalBsmtSF`), surface du garage (`GarageArea`), surface du terrain (`LotArea`).
- **Localisation** : quartier (`Neighborhood`), zonage (`MSZoning`), configuration de la parcelle.
- **Caractéristiques structurelles** : qualité globale (`OverallQual`), état général (`OverallCond`), type de fondation, matériaux de toiture et de façade.
- **Équipements** : nombre de salles de bains, présence d'une cheminée, type de chauffage, présence d'une climatisation centrale.
- **Historique temporel** : année de construction (`YearBuilt`), année de rénovation (`YearRemodAdd`), mois et année de vente.

---

## Prediction Task

### Type d'apprentissage

Ce projet relève de l'**apprentissage supervisé**, avec une tâche de **régression** : la variable cible est continue. Il s'agit du **logarithme du prix de vente** (`log1p(SalePrice)`), transformé pour symétriser la distribution et modéliser des erreurs relatives plutôt qu'absolues.

### Objectif de la prédiction

Prédire la **valeur marchande théorique** d'un bien immobilier à partir de ses caractéristiques cadastrales déclarées, dans le but de :
- Détecter les biens dont le **prix de vente réel dépasse significativement la valeur estimée**, signalant une potentielle fraude (travaux non déclarés, équipements dissimulés).
- Prioriser les dossiers transmis au service des contentieux en concentrant les audits sur les écarts les plus importants.

### Données en entrée

Les variables explicatives sont de nature mixte :
- **Numériques** (21 variables) : surfaces, surfaces des différents niveaux, valeurs monétaires de base, superficies des équipements.
- **Catégorielles** (53 variables) : qualité des matériaux, type de quartier, présence et type d'équipements, zonage, style architectural.
- **Temporelles** (4 variables) : années de construction, rénovation, construction du garage et de vente — transformées en **ancienneté** (différence avec l'année de vente) pour capturer l'effet de dépréciation.

### Donnée en sortie

La sortie du modèle est le **prix de vente estimé du bien**, en espace prix réel (après transformation inverse de `log1p`), directement comparable au prix de transaction déclaré.

### Justification métier

Une mauvaise prédiction a un coût asymétrique pour l'administration :
- Un **faux négatif** (fraude non détectée) prive l'État d'une recette fiscale légitime et envoie un signal de permissivité aux contribuables indélicats.
- Un **faux positif** (citoyen innocent accusé) génère des frais administratifs, des contentieux et un préjudice moral pour le contribuable, avec un risque réputationnel pour l'administration.

C'est pourquoi la métrique **RMSLE** (Root Mean Squared Logarithmic Error) est privilégiée : elle pénalise proportionnellement les erreurs relatives, sans avantager les biens de grande valeur, et s'aligne naturellement sur la transformation logarithmique de la variable cible.

---

## Features (Engineering)

### Méthodes d'extraction et de transformation

1. **Traitement des valeurs manquantes**
   - Variables numériques : imputation par **0**, car les `NaN` signifient l'absence de la caractéristique (pas de piscine → `PoolArea = 0`).
   - Variables catégorielles : imputation par la modalité **`'NA'`**, créant une catégorie explicite pour l'absence.
   - `GarageYrBlt` manquant : imputé avec `YearBuilt`, hypothèse que le garage a été construit en même temps que la maison.

2. **Transformation des variables temporelles**
   - Les quatre variables de date sont converties en **ancienneté au moment de la vente** (`YrSold - YearBuilt`, etc.), ce qui rend la variable indépendante de l'année absolue et capture l'effet de dépréciation.

3. **Encodage des variables catégorielles**
   - Application d'un **Target Mean Encoding** en section 2 : chaque modalité est remplacée par la moyenne du log-prix observé pour cette catégorie sur les données d'entraînement uniquement, afin d'éviter tout data leakage.

4. **Standardisation des variables numériques**
   - Application d'un `StandardScaler` dans le pipeline sklearn (section 3), opérant uniquement sur le fold d'entraînement à chaque pli de la validation croisée.

### Variables les plus discriminantes (issues de l'EDA)

Les variables présentant la plus forte corrélation avec le log-prix sont, dans l'ordre :
- `OverallQual` — qualité globale de construction (notée de 1 à 10)
- `GrLivArea` — surface habitable totale au-dessus du sol
- `TotalBsmtSF` — surface totale du sous-sol
- `GarageArea` / `GarageCars` — capacité et surface du garage
- `Neighborhood` — le quartier, facteur clé de valorisation

---

## Offline Evaluation

### Métriques techniques

La métrique principale est le **RMSLE** (Root Mean Squared Logarithmic Error), cohérente avec la transformation `log1p` de la variable cible et avec les exigences de la compétition Kaggle utilisée comme benchmark externe.

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\log(1 + \hat{y}_i) - \log(1 + y_i)\right)^2}$$

Elle présente deux avantages dans le contexte fiscal :
- Elle pénalise **proportionnellement** les erreurs, qu'il s'agisse de biens modestes ou de biens de valeur élevée.
- Elle est **naturellement bornée** et insensible aux valeurs extrêmes, réduisant l'influence des transactions atypiques (successions, ventes forcées).

### Protocole d'évaluation

- **Validation croisée K-Fold** à 5 plis (`random_state=42`) sur le jeu d'entraînement pour la sélection et le tuning des modèles.
- **Score Kaggle public** comme validation externe hors-distribution (jeu de test non vu, 1 459 observations).
- **Objectif de performance** : RMSLE < 0.15 en production (seuil d'alerte défini en section 6.3).

### Analyse des erreurs

| Type d'erreur | Conséquence métier |
|---|---|
| **Faux positif** (surestimation du prix théorique) | Citoyen innocent transmis au contentieux — frais administratifs, préjudice moral |
| **Faux négatif** (sous-estimation du prix théorique) | Dossier frauduleux non détecté — perte de recette fiscale |

---

## Decisions

### Utilisation des prédictions dans le processus décisionnel

Le système agit comme une **guillotine fiscale** : si le modèle confirme que le prix de vente reflète des équipements non déclarés (écart significatif entre prix estimé et prix réel), le dossier est automatiquement transmis au service des contentieux pour audit approfondi.

Le déclenchement intervient **après chaque vente de bien immobilier**, lors de la procédure d'enregistrement de la transaction auprès des organes de l'État.

### Interaction entre les agents fiscaux et les prédictions

- L'agent consulte, pour chaque dossier, le **prix estimé** par le modèle, le **prix de vente déclaré**, et **l'écart en pourcentage**.
- Pour les dossiers proches du seuil de déclenchement, une **validation humaine** est obligatoire avant transmission au contentieux — le modèle aide, il ne décide pas seul.
- Les explications **SHAP** associées à chaque prédiction permettent à l'agent de comprendre quelles caractéristiques du bien ont le plus influencé l'estimation, et donc de motiver la décision d'audit de façon défendable devant le contribuable.

### Coûts associés à la prise de décision

- **Présence d'un humain dans la boucle** obligatoire pour les biens dont l'estimation se situe dans une fenêtre de ±5% autour d'un seuil fiscal critique.
- **Risque d'iniquité** entre quartiers si les performances du modèle sont hétérogènes selon les segments de marché (à surveiller, cf. section 6.4).
- **Risque légal** : toute décision ayant des effets juridiques sur un contribuable est encadrée par le RGPD (article 22) et la nLPD suisse — le citoyen a droit à l'information, à l'intervention d'un humain et à la possibilité de contester.

---

## Making Predictions

### Quand les prédictions sont-elles réalisées ?

Les prédictions sont générées **après chaque vente de bien immobilier**, au moment de l'enregistrement de la transaction. Deux modes sont envisagés :
- **À la demande** : pour un dossier individuel signalé par un agent lors d'une mutation spécifique.
- **Par lot** : lors de campagnes de réévaluation cadastrale périodiques, où l'intégralité du parc immobilier d'une zone est réévaluée simultanément.

### Complexité calculatoire acceptable

Les pipelines CatBoost et XGBoost offrent des **temps d'inférence en quelques millisecondes** par bien, compatibles avec une utilisation à la demande. Le traitement par lot de l'ensemble du cadastre (plusieurs milliers de biens) reste faisable en quelques minutes sur infrastructure standard.

### Intervention humaine

L'intervention humaine n'est pas requise pour produire une prédiction, mais elle est **obligatoire pour valider la décision de transmission** au service des contentieux, notamment pour les dossiers proches des seuils fiscaux critiques. Un agent expert peut examiner les explications SHAP et décider, en connaissance de cause, d'engager ou non la procédure d'audit.

---

## Collecting Data

### Stratégie de collecte continue

Les données d'entraînement actuelles (Ames Housing, 2006–2010) constituent une base solide mais localisée dans le temps. Pour maintenir la pertinence du modèle, une stratégie de collecte continue est nécessaire :
- **Alimentation automatique** à partir des registres de ventes officiels, à chaque nouvelle transaction enregistrée auprès de l'État de l'Iowa.
- **Mise à jour cadastrale** : intégration des déclarations de travaux déposées par les propriétaires, permettant de détecter les mises à jour officielles et d'identifier les biens sans déclaration malgré une plus-value visible.
- **Retours des audits** : les conclusions des contentieux (fraude confirmée ou non) constituent des données d'apprentissage précieuses pour affiner le modèle sur les cas limites.

### Fréquence de mise à jour

| Source | Fréquence |
|---|---|
| Registres de ventes officiels | En continu (après chaque transaction) |
| Données cadastrales | Trimestrielle |
| Résultats d'audits contentieux | Au fil des décisions rendues |
| Réentraînement complet du modèle | Annuel, ou déclenché si RMSLE > 0.15 en production |

---

## Building Models

### Nombre et fréquence de réentraînement

Un seul modèle de production est maintenu à la fois (la combinaison CatBoost + XGBoost), avec réentraînement **annuel** aligné sur le cycle fiscal, ou déclenché par une dégradation des performances (RMSLE production > 0.15, cf. section 6.3).

### Durée et ressources

L'entraînement complet du pipeline (Optuna + validation croisée 5 plis sur 7 modèles) est estimé à moins de **4 heures** sur infrastructure CPU standard, compatible avec un traitement hors-heures ouvrées.

### Évolution du modèle

Le pipeline sklearn encapsule l'intégralité du prétraitement et du modèle dans un seul objet sérialisable (`joblib`), versionnables dans un catalogue de modèles (MLflow). Chaque version est conservée pour permettre la **rejouabilité de toute estimation passée**, exigence légale dans un contexte fiscal.
