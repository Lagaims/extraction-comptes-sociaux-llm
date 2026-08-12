# Site de diagnostic de l'extraction

Site Quarto statique présentant le diagnostic des erreurs d'extraction de tableaux :
mesures, propositions d'amélioration, et comparaison tableau annoté / tableau extrait.

Déployé sur GitHub Pages à chaque push sur `main`
([`.github/workflows/publish-site.yml`](../.github/workflows/publish-site.yml)).

## Structure

| Fichier | Rôle |
|---|---|
| `_quarto.yml` | Configuration du site : navigation, thème, ressources. |
| `index.qmd` | Synthèse — chiffres-clés et orientation vers les trois autres pages. |
| `chiffres.qmd` | Toutes les mesures et les cas particuliers rencontrés. |
| `ameliorations.qmd` | Cinq leviers classés par impact mesuré, plus les corrections de mesure. |
| `comparaison.qmd` | Visualiseur : grille annotée face à la grille extraite, cellule par cellule. |
| `build_data.py` | Produit `data/comparaisons.json` depuis S3, en pseudonymisant les entités. |
| `styles.scss`, `styles-dark.scss` | Thème clair et thème sombre. |

Les pages `.qmd` ne contiennent **aucun code exécuté** (`execute: enabled: false`) :
Quarto ne fait que rendre du markdown. Python ne sert qu'à `build_data.py`.

## Rendu local

```bash
# 1. Récupérer et pseudonymiser les données depuis S3 (nécessite les identifiants S3)
uv run --project website python website/build_data.py

# 2. Rendre le site
quarto render website

# ou, avec rechargement automatique
quarto preview website
```

Le site rendu atterrit dans `website/_site/`. `data/` et `_site/` sont ignorés par git.

L'option `--limit N` de `build_data.py` ne traite que les N premiers tableaux, pour
itérer rapidement sur la mise en page.

## Données et pseudonymisation

`data/comparaisons.json` n'est **jamais versionné** : le `.gitignore` du dépôt interdit
de committer données extraites et annotations. Le fichier est reconstruit à chaque
build, en CI comme en local.

`build_data.py` pseudonymise les entités avant d'écrire quoi que ce soit :

- **raisons sociales, adresses** → `Entité 07`. La règle conserve un libellé seulement
  si *tous* ses mots significatifs appartiennent au vocabulaire comptable (`_VOCAB`) ;
  tout le reste est remplacé. Le sens de la règle est délibéré : conserver dès qu'un mot
  est du vocabulaire laisserait passer « Société Générale ».
- **SIREN dans les noms de fichiers** → `TAB-05_2`, le rang du tableau étant conservé.
- **montants** → inchangés. Ce sont eux que le diagnostic doit montrer, et ils ne
  réidentifient personne une fois les noms retirés.

Les variantes d'un même nom dues à l'OCR partagent le même numéro et se distinguent par
un suffixe (`Entité 07` / `Entité 07·b`), pour qu'un écart de libellé entre annotation et
prédiction reste visible sur le site.

La pseudonymisation s'applique à **toutes** les cellules, pas seulement aux colonnes
d'en-tête : une erreur d'alignement — le sujet même du site — déplace régulièrement une
raison sociale dans une colonne de montants.

### Vérifier l'absence de fuite après un changement de corpus

```bash
uv run --project website python -c "
import json, re
d = json.load(open('website/data/comparaisons.json'))
kept = set()
for t in d['tables']:
    for row in t['ann']: kept |= {c.strip() for c in row}
    for m in t['methods'].values():
        for row in m['pred']: kept |= {c.strip() for c in row}
labels = sorted({s for s in kept
                 if s and not s.startswith('Entité') and re.search(r'[A-Za-zÀ-ÿ]{3}', s)})
print(len(labels), 'libellés conservés en clair')
for s in labels: print(' ', s[:100])
"
```

Tout libellé de cette liste qui n'est pas du vocabulaire comptable est une fuite :
ajouter le mot manquant à `_VOCAB` ne suffit pas — c'est l'inverse qu'il faut faire,
vérifier pourquoi la règle l'a conservé.

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
