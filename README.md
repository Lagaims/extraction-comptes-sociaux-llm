# extraction-comptes-sociaux-llm

Extraction automatique de tableaux depuis des PDFs (comptes sociaux) via OCR et LLM.

Le pipeline repose sur [marker-pdf](https://github.com/datalab-to/marker) (OCR neuronal via [Surya](https://github.com/datalab-to/surya)) complété par un LLM pour la correction et la structuration des tableaux.

---

## Architecture

```
scripts/
  extraction_pdf_via_api.py   ← script d'orchestration (lecture S3 → API → écriture S3)

api/
  marker_proxy/   (port 1324) ← proxy LLM avec tracing Langfuse
  api_marker/     (port 8001) ← OCR + structuration JSON via marker-pdf  ← nécessite GPU
  api_opendataloader/ (port 8002) ← extraction alternative (Java/OpenDataLoader)
  api_chandra/    (port 8003) ← extraction via VLM (vllm)

libs/
  src/extraction_common/      ← utilitaires S3 partagés
  src/data_management/        ← logique marker (config, extraction, conversion)
```

`api_marker` et `marker_proxy` fonctionnent en tandem : `api_marker` appelle `marker_proxy` pour les appels LLM, qui relaie vers le LLM distant en ajoutant le tracing Langfuse.

---

## Prérequis

- **Python 3.13** (géré par `uv`)
- **[uv](https://docs.astral.sh/uv/)** (`pip install uv` ou `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **GPU NVIDIA avec ≥ 16 Go de VRAM** — voir [section GPU](#gpu--performances) ci-dessous
- Accès S3 (MinIO SSP Cloud ou AWS)
- Accès à un LLM compatible OpenAI (ex. `https://llm.lab.sspcloud.fr`)

---

## Configuration

Copier `.env.example` en `.env` à la racine (ou exporter les variables dans le shell) :

```dotenv
# LLM distant (utilisé par marker_proxy)
REAL_LLM_BASE_URL=https://llm.lab.sspcloud.fr/api
REAL_LLM_API_KEY=<votre-clé>

# Proxy marker (utilisé par api_marker pour joindre marker_proxy)
# Défaut : http://localhost:1324/v1 — ne pas changer en local
PROXY_URL=http://localhost:1324/v1

# S3
AWS_S3_BUCKET=<nom-bucket>
AWS_ACCESS_KEY_ID=<votre-clé>
AWS_SECRET_ACCESS_KEY=<votre-secret>
AWS_SESSION_TOKEN=          # optionnel
AWS_S3_ENDPOINT=<nom-endpoint>
AWS_REGION=us-east-1

# Langfuse (optionnel, désactivé si absent)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://langfuse.lab.sspcloud.fr
```

---

## Démarrage local

Les deux services doivent être lancés **avant** d'exécuter le script d'extraction.

### 1. marker_proxy (port 1324)

```bash
cd api/marker_proxy
uv run python -m uvicorn proxy:app --host 0.0.0.0 --port 1324 --app-dir src
```

Vérification : `curl http://localhost:1324/health`

### 2. api_marker (port 8001)

```bash
cd api/api_marker
uv run python -m uvicorn main_marker:app --host 0.0.0.0 --port 8001 --app-dir src
```

> **Note :** le démarrage charge tous les modèles Surya en mémoire (~3,5 Go de VRAM). Il peut prendre 1–2 minutes la première fois (téléchargement des poids depuis S3).

### 3. Script d'extraction

```bash
cd scripts

# Traiter tous les PDFs du fichier de correspondances (avec skip automatique des déjà traités)
uv run extraction_pdf_via_api.py --from-parquet

# Tester sur un PDF spécifique
uv run extraction_pdf_via_api.py --pdf-key dossier/fichier.pdf

# Lister les PDFs disponibles dans S3
uv run extraction_pdf_via_api.py --list
```

---

## GPU & Performances

### Pourquoi un GPU est indispensable

`api_marker` utilise les modèles Surya (transformers autorégessifs ~720 M de paramètres).  
Sans GPU, l'inférence OCR est **10 à 50 fois plus lente** qu'avec CUDA.

| Matériel | Vitesse (ordre de grandeur) |
|---|---|
| GPU 16 Go VRAM (ex. A10, RTX 4080) | ~30–60 s / page |
| CPU seul (même 64+ cœurs) | 10–30 min / page |

### Empreinte mémoire GPU (VRAM)

Les modèles sont tous chargés au démarrage de `api_marker` et restent en VRAM :

| Modèle | Dtype | VRAM |
|---|---|---|
| Foundation — recognition (719 M params) | bfloat16 | 1,44 Go |
| Foundation — layout (723 M params) | bfloat16 | 1,45 Go |
| OCR error detection | float16 | 0,27 Go |
| Table recognition | float16 | 0,21 Go |
| Text detection | float16 | 0,08 Go |
| **Total statique** | | **~3,5 Go** |

Pendant l'inférence OCR (KV cache + activations) :

| `recognition_batch_size` | VRAM supplémentaire (pic) | Total estimé |
|---|---|---|
| 48 (recommandé pour 16 Go) | ~2,5 Go | **~6 Go** |
| 64 (config actuelle) | ~3,5 Go | **~7–9 Go** |
| 256 (défaut CUDA Surya) | OOM sur 16 Go | — |

**Règle : sur un GPU de 16 Go, utiliser `recognition_batch_size` ≤ 48.**

### Paramètres critiques à vérifier avant de lancer

Dans `libs/src/data_management/extract_image_to_json.py` :

```python
config = {
    # 1. LLM activé/désactivé
    #    False → OCR seul, pas d'amélioration des tableaux par LLM
    #    True  → correction des tableaux par LLM (recommandé)
    "use_llm": True,

    # 2. Nom du modèle LLM — doit correspondre à un modèle disponible
    #    Vérifier avec : curl http://localhost:1324/v1/models
    "openai_model": "gemma4-26b-moe",

    # 3. Batch size OCR — à adapter selon la VRAM disponible
    #    CPU : 8–16   |   GPU 16 Go : 32–48   |   GPU 24+ Go : 64–128
    "recognition_batch_size": 48,

    # 4. force_ocr=True → tout le texte passe par le réseau de neurones
    #    Nécessaire pour les PDFs scannés/images
    #    Mettre False pour les PDFs nés-numériques (extraction native ~100× plus rapide)
    "force_ocr": True,
}
```

> **Attention :** `marker` ignore silencieusement les valeurs booléennes `False` dans sa configuration interne (`generate_config_dict` filtre toutes les valeurs falsy). `use_llm: False` est donc équivalent à ne pas spécifier la clé — le LLM est désactivé dans les deux cas.

### Checklist avant d'ouvrir le service sur GPU

- [ ] Le pod/service dispose d'un GPU avec ≥ 16 Go de VRAM
- [ ] `recognition_batch_size` ≤ 48 pour 16 Go de VRAM
- [ ] `openai_model` correspond à un modèle disponible (`curl http://localhost:1324/v1/models`)
- [ ] `REAL_LLM_API_KEY` est définie et valide
- [ ] `PROXY_URL` pointe vers `marker_proxy` (ex. `http://marker-proxy:1324/v1` en Kubernetes)
- [ ] Le démarrage de `api_marker` s'est terminé sans erreur (les modèles sont bien chargés en VRAM)
- [ ] `curl http://localhost:8001/docs` répond (swagger accessible)
- [ ] `curl http://localhost:1324/health` renvoie `{"status": "ok", "real_llm_configured": true}`

---


## Troubleshooting

| Symptôme | Cause probable | Solution |
|---|---|---|
| OCR très lent (>10 min/page) | Inférence sur CPU, pas de GPU | Vérifier que CUDA est disponible dans le pod |
| `400 - Model not found` dans les logs | `openai_model` incorrect | Vérifier les modèles disponibles via `/v1/models` |
| `use_llm: False` n'a aucun effet | Bug marker : les valeurs `False` sont filtrées | Comportement normal, `False` = LLM désactivé |
| `AssertionError: openai_api_key` au démarrage | `REAL_LLM_API_KEY` non définie | Définir la variable d'environnement |
| OOM GPU pendant l'OCR | `recognition_batch_size` trop élevé | Réduire à 32 ou 48 |
| Les tableaux ne sont pas corrigés | `use_llm: False` dans la config | Passer à `use_llm: True` |
