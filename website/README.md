# Site de diagnostic de l'extraction

Site Quarto statique présentant le diagnostic des erreurs d'extraction de tableaux :
mesures, propositions d'amélioration, et comparaison tableau annoté / tableau extrait.

Déployé sur GitHub Pages à chaque push sur `main`
([`.github/workflows/publish-site.yml`](../.github/workflows/publish-site.yml)).

## Structure

| Fichier | Rôle |
|---|---|
| `_quarto.yml` | Configuration du site : navigation, thème, ressources. |
| `index.qmd` | Synthèse — chiffres-clés et orientation vers les quatre autres pages. |
| `conversion.qmd` | L'étape `json_to_csv.py` : format d'entrée par moteur, règles communes. |
| `chiffres.qmd` | Toutes les mesures et les cas particuliers rencontrés. |
| `ameliorations.qmd` | Cinq leviers classés par impact mesuré, plus les corrections de mesure. |
| `comparaison.qmd` | Visualiseur : grille annotée face à la grille extraite, cellule par cellule. |
| `build_data.py` | Produit `data/comparaisons.json` depuis S3. |
| `styles.scss`, `styles-dark.scss` | Thème clair et thème sombre. |

Les pages `.qmd` ne contiennent **aucun code exécuté** (`execute: enabled: false`) :
Quarto ne fait que rendre du markdown. Python ne sert qu'à `build_data.py`.

## Rendu local

```bash
# 1. Récupérer les données depuis S3 (nécessite les identifiants S3)
uv run --project website python website/build_data.py

# 2. Rendre le site
quarto render website

# ou, avec rechargement automatique
quarto preview website
```

Le site rendu atterrit dans `website/_site/`. `data/` et `_site/` sont ignorés par git.

L'option `--limit N` de `build_data.py` ne traite que les N premiers tableaux, pour
itérer rapidement sur la mise en page.

## Données

`data/comparaisons.json` n'est **jamais versionné** : le `.gitignore` du dépôt interdit
de committer données extraites et annotations. Le fichier est reconstruit à chaque
build, en CI comme en local.

Les grilles sont publiées **telles quelles** — raisons sociales, SIREN, montants. Le
corpus provient de comptes sociaux déposés et publiés en open data par l'INPI : rien
n'y est pseudonymisé, le site montre exactement ce que le pipeline a lu. L'identifiant
d'un tableau est le nom de fichier d'origine (`487772899_2`), ce qui permet de
remonter au PDF source.

## Déploiement

Le workflow rend le site et le publie via GitHub Pages (`actions/deploy-pages`), sans
branche `gh-pages`. Prérequis côté dépôt :

1. **Settings → Pages → Source : GitHub Actions.**
2. **Settings → Secrets and variables → Actions**, ajouter les identifiants S3 :
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ENDPOINT`, et
   `AWS_SESSION_TOKEN` si les identifiants sont temporaires.

Sans ces secrets, le workflow rend quand même le site : la page de comparaison affiche
alors un message d'indisponibilité au lieu des grilles, les trois autres pages étant
complètes.

> **Attention** — les identifiants temporaires du SSP Cloud expirent. Un secret
> `AWS_SESSION_TOKEN` périmé fait échouer l'étape de récupération des données, sans
> empêcher la publication du site.
